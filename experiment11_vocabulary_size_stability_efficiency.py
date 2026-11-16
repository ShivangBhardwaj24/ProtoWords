from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "outputs").is_dir() else SCRIPT_DIR.parent
EXPERIMENT = "exp11_vocabulary_size_stability_efficiency"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
N_CLASSES = len(CLASS_NAMES)
FEATURE_DIM = 384
DEFAULT_CACHE = Path(
    "outputs/exp1_baseline/"
    "cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz"
)
DISTANCE_C = {32: 100.0, 64: 100.0, 128: 1.0, 256: 0.1}
HARD_C = 0.01
CONTINUOUS_C = 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare global ProtoWords vocabulary sizes across random seeds."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--ks", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter-kmeans", type=int, default=300)
    parser.add_argument("--max-iter-logreg", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.data.is_absolute():
        args.data = args.data.expanduser().resolve()
    else:
        args.data = (ROOT / args.data).expanduser().resolve()

    if not args.ks or len(set(args.ks)) != len(args.ks):
        parser.error("K values must be unique")
    if any(k not in DISTANCE_C for k in args.ks):
        parser.error("Supported K values are 32, 64, 128 and 256")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide at least two different random seeds")
    if min(args.n_init, args.max_iter_kmeans, args.max_iter_logreg,
           args.batch_size, args.threads) < 1:
        parser.error("Initialization, iteration, batch and thread values must be positive")

    args.ks = sorted(args.ks)
    args.results_dir = ROOT / "results" / EXPERIMENT
    args.outputs_dir = ROOT / "outputs" / EXPERIMENT
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    return args


def normalize_features(X):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def load_features(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature cache not found: {path}\n"
            "Run Experiment 1 first or pass the cache path with --data."
        )

    with np.load(path, allow_pickle=False) as data:
        required = {"X_train", "y_train", "X_val", "y_val"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Feature cache is missing: {sorted(missing)}")
        X_train = np.asarray(data["X_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=np.int64)
        X_val = np.asarray(data["X_val"], dtype=np.float32)
        y_val = np.asarray(data["y_val"], dtype=np.int64)

    for split, X, y in (("training", X_train, y_train), ("validation", X_val, y_val)):
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(X) == 0:
            raise ValueError(f"Invalid {split} feature or label shapes")
        if X.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected {FEATURE_DIM}-dimensional features in {split} data")
        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite features found in {split} data")
        if set(np.unique(y).tolist()) != set(range(N_CLASSES)):
            raise ValueError(f"All CIFAR-10 labels 0 through 9 must appear in {split} data")

    return normalize_features(X_train), y_train, normalize_features(X_val), y_val


def make_hard_features(cluster_ids, k):
    rows = np.arange(len(cluster_ids), dtype=np.int32)
    cols = np.asarray(cluster_ids, dtype=np.int32)
    values = np.ones(len(cluster_ids), dtype=np.float32)
    return csr_matrix((values, (rows, cols)), shape=(len(cluster_ids), k), dtype=np.float32)


def make_distance_features(X, centers, batch_size, threads):
    distances = np.empty((len(X), len(centers)), dtype=np.float32)
    center_norms = np.sum(centers * centers, axis=1)[None, :]

    with threadpool_limits(limits=threads):
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch = X[start:end]
            squared = (
                np.sum(batch * batch, axis=1, keepdims=True)
                + center_norms
                - 2.0 * (batch @ centers.T)
            )
            np.maximum(squared, 0.0, out=squared)
            np.sqrt(squared, out=squared)
            distances[start:end] = squared

    return distances


def cluster_statistics(assignments, labels, k):
    counts = np.zeros((k, N_CLASSES), dtype=np.int64)
    np.add.at(counts, (assignments, labels), 1)
    sizes = counts.sum(axis=1)
    nonempty = sizes > 0

    purity = np.divide(
        counts.max(axis=1), sizes,
        out=np.zeros(k, dtype=np.float64), where=nonempty,
    )
    probabilities = np.divide(
        counts, sizes[:, None],
        out=np.zeros_like(counts, dtype=np.float64), where=nonempty[:, None],
    )
    log_probabilities = np.zeros_like(probabilities)
    positive = probabilities > 0
    log_probabilities[positive] = np.log2(probabilities[positive])
    entropy = -(probabilities * log_probabilities).sum(axis=1)

    return {
        "empty_prototypes": int(np.sum(~nonempty)),
        "smallest_cluster": int(sizes.min()),
        "largest_cluster": int(sizes.max()),
        "mean_cluster_purity": float(purity[nonempty].mean()),
        "weighted_cluster_purity": float(counts.max(axis=1).sum() / sizes.sum()),
        "mean_cluster_entropy_bits": float(entropy[nonempty].mean()),
        "weighted_cluster_entropy_bits": float(np.average(entropy[nonempty], weights=sizes[nonempty])),
    }, sizes


def fit_classifier(X_train, y_train, X_val, y_val, c_value, seed, max_iter, threads):
    model = LogisticRegression(
        C=float(c_value), solver="lbfgs", max_iter=max_iter, random_state=seed
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        with threadpool_limits(limits=threads):
            model.fit(X_train, y_train)
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    predictions = model.predict(X_val)
    metrics = {
        "validation_accuracy": float(accuracy_score(y_val, predictions)),
        "validation_macro_f1": float(f1_score(y_val, predictions, average="macro", zero_division=0)),
        "validation_weighted_f1": float(f1_score(y_val, predictions, average="weighted", zero_division=0)),
    }
    return model, metrics, convergence_warning


def measure_inference(model, method, X_val, kmeans, centers, scaler, batch_size, threads, repeats=3):
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        if method == "hard":
            with threadpool_limits(limits=threads):
                ids = kmeans.predict(X_val)
            features = make_hard_features(ids, len(centers))
            model.predict(features)
        elif method == "distance":
            distances = make_distance_features(X_val, centers, batch_size, threads)
            scaled = scaler.transform(distances)
            model.predict(scaled)
        else:
            model.predict(X_val)
        elapsed.append(time.perf_counter() - start)
    return float(np.median(elapsed) * 1000.0 / len(X_val))


def parameter_bytes(model, scaler=None):
    total = model.coef_.nbytes + model.intercept_.nbytes
    if scaler is not None:
        for name in ("mean_", "scale_", "var_"):
            value = getattr(scaler, name, None)
            if isinstance(value, np.ndarray):
                total += value.nbytes
    return int(total)


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def main():
    args = parse_args()
    start_time = time.perf_counter()
    X_train, y_train, X_val, y_val = load_features(args.data)

    if max(args.ks) > len(X_train):
        raise ValueError("Largest K cannot exceed the number of training images")

    print(f"Training features: {X_train.shape}")
    print(f"Validation features: {X_val.shape}")
    print(f"Vocabulary sizes: {args.ks}")
    print(f"Seeds: {args.seeds}")
    print("Codebooks are fitted on training features only; validation is used only for evaluation")

    baseline = LogisticRegression(C=CONTINUOUS_C, solver="lbfgs", max_iter=args.max_iter_logreg)
    with threadpool_limits(limits=args.threads):
        baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_val)
    baseline_metrics = {
        "validation_accuracy": float(accuracy_score(y_val, baseline_predictions)),
        "validation_macro_f1": float(f1_score(y_val, baseline_predictions, average="macro", zero_division=0)),
        "validation_weighted_f1": float(f1_score(y_val, baseline_predictions, average="weighted", zero_division=0)),
    }
    baseline_time = measure_inference(
        baseline, "continuous", X_val, None, None, None,
        args.batch_size, args.threads,
    )
    baseline_parameter_bytes = parameter_bytes(baseline)
    print(
        f"Continuous baseline: accuracy={baseline_metrics['validation_accuracy']:.4f}, "
        f"Macro-F1={baseline_metrics['validation_macro_f1']:.4f}"
    )

    seed_rows = [{
        "K": "baseline",
        "seed": "",
        "method": "continuous_dinov2_lr",
        "selected_C": CONTINUOUS_C,
        **baseline_metrics,
        "kmeans_fit_seconds": "",
        "classifier_fit_seconds": "",
        "inference_ms_per_image": baseline_time,
        "mean_cluster_purity": "",
        "weighted_cluster_purity": "",
        "mean_cluster_entropy_bits": "",
        "empty_prototypes": "",
    }]
    pairwise_rows = []
    storage_rows = [{
        "K": "baseline",
        "method": "continuous_dinov2_lr",
        "representation_bytes_per_image": FEATURE_DIM * 4,
        "codebook_bytes": 0,
        "scaler_bytes": 0,
        "classifier_parameter_bytes": baseline_parameter_bytes,
        "amortized_shared_bytes_per_image": baseline_parameter_bytes / 50000.0,
        "estimated_total_bytes_per_image": FEATURE_DIM * 4 + baseline_parameter_bytes / 50000.0,
        "compression_vs_continuous_percent": 0.0,
    }]

    summary_rows = []
    saved_arrays = {}

    for k in args.ks:
        print(f"\nVocabulary size K={k}")
        assignments_by_seed = {}
        seed_cluster_rows = []
        seed_metric_by_method = {"hard": [], "distance": []}
        seed_fit_times = []
        last_hard_model = None
        last_distance_model = None
        last_scaler = None
        last_centers = None
        last_kmeans = None

        for seed in args.seeds:
            print(f"  Seed {seed}: fitting K-Means")
            kmeans = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=args.n_init,
                max_iter=args.max_iter_kmeans,
                algorithm="lloyd",
            )
            fit_start = time.perf_counter()
            with threadpool_limits(limits=args.threads):
                kmeans.fit(X_train)
            kmeans_seconds = time.perf_counter() - fit_start
            centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
            train_ids = np.asarray(kmeans.labels_, dtype=np.int32)
            val_ids = np.asarray(kmeans.predict(X_val), dtype=np.int32)
            assignments_by_seed[seed] = train_ids
            stats, cluster_sizes = cluster_statistics(train_ids, y_train, k)
            seed_cluster_rows.append(stats)
            seed_fit_times.append(kmeans_seconds)

            saved_arrays[f"centers_k{k}_seed{seed}"] = centers
            saved_arrays[f"train_assignments_k{k}_seed{seed}"] = train_ids
            saved_arrays[f"validation_assignments_k{k}_seed{seed}"] = val_ids

            hard_train = make_hard_features(train_ids, k)
            hard_val = make_hard_features(val_ids, k)
            hard_fit_start = time.perf_counter()
            hard_model, hard_metrics, hard_warning = fit_classifier(
                hard_train, y_train, hard_val, y_val,
                HARD_C, seed, args.max_iter_logreg, args.threads,
            )
            hard_fit_seconds = time.perf_counter() - hard_fit_start
            hard_inference = measure_inference(
                hard_model, "hard", X_val, kmeans, centers, None,
                args.batch_size, args.threads,
            )
            seed_rows.append({
                "K": k, "seed": seed, "method": "hard",
                "selected_C": HARD_C, **hard_metrics,
                "kmeans_fit_seconds": kmeans_seconds,
                "classifier_fit_seconds": hard_fit_seconds,
                "inference_ms_per_image": hard_inference,
                "mean_cluster_purity": stats["mean_cluster_purity"],
                "weighted_cluster_purity": stats["weighted_cluster_purity"],
                "mean_cluster_entropy_bits": stats["mean_cluster_entropy_bits"],
                "empty_prototypes": stats["empty_prototypes"],
                "convergence_warning": hard_warning,
            })
            seed_metric_by_method["hard"].append(hard_metrics)

            distance_start = time.perf_counter()
            train_distances = make_distance_features(
                X_train, centers, args.batch_size, args.threads
            )
            val_distances = make_distance_features(
                X_val, centers, args.batch_size, args.threads
            )
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_distances).astype(np.float32, copy=False)
            val_scaled = scaler.transform(val_distances).astype(np.float32, copy=False)
            del train_distances, val_distances
            distance_transform_seconds = time.perf_counter() - distance_start

            distance_fit_start = time.perf_counter()
            distance_c = DISTANCE_C[k]
            distance_model, distance_metrics, distance_warning = fit_classifier(
                train_scaled, y_train, val_scaled, y_val,
                distance_c, seed, args.max_iter_logreg, args.threads,
            )
            distance_fit_seconds = time.perf_counter() - distance_fit_start
            distance_inference = measure_inference(
                distance_model, "distance", X_val, kmeans, centers, scaler,
                args.batch_size, args.threads,
            )
            seed_rows.append({
                "K": k, "seed": seed, "method": "distance",
                "selected_C": distance_c, **distance_metrics,
                "kmeans_fit_seconds": kmeans_seconds,
                "distance_transform_seconds": distance_transform_seconds,
                "classifier_fit_seconds": distance_fit_seconds,
                "inference_ms_per_image": distance_inference,
                "mean_cluster_purity": stats["mean_cluster_purity"],
                "weighted_cluster_purity": stats["weighted_cluster_purity"],
                "mean_cluster_entropy_bits": stats["mean_cluster_entropy_bits"],
                "empty_prototypes": stats["empty_prototypes"],
                "convergence_warning": distance_warning,
            })
            seed_metric_by_method["distance"].append(distance_metrics)

            last_hard_model = hard_model
            last_distance_model = distance_model
            last_scaler = scaler
            last_centers = centers
            last_kmeans = kmeans
            print(
                f"    purity={stats['mean_cluster_purity']:.4f}, "
                f"hard={hard_metrics['validation_accuracy']:.4f}, "
                f"distance={distance_metrics['validation_accuracy']:.4f}"
            )

            del hard_train, hard_val, train_scaled, val_scaled

        for seed_a, seed_b in itertools.combinations(args.seeds, 2):
            assignments_a = assignments_by_seed[seed_a]
            assignments_b = assignments_by_seed[seed_b]
            pairwise_rows.append({
                "K": k,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "ARI": float(adjusted_rand_score(assignments_a, assignments_b)),
                "NMI": float(normalized_mutual_info_score(assignments_a, assignments_b)),
            })

        k_pairs = [row for row in pairwise_rows if row["K"] == k]
        ari_mean, ari_std = summarize([row["ARI"] for row in k_pairs])
        nmi_mean, nmi_std = summarize([row["NMI"] for row in k_pairs])
        purity_mean, purity_std = summarize([row["mean_cluster_purity"] for row in seed_cluster_rows])
        weighted_purity_mean, weighted_purity_std = summarize(
            [row["weighted_cluster_purity"] for row in seed_cluster_rows]
        )
        entropy_mean, entropy_std = summarize(
            [row["mean_cluster_entropy_bits"] for row in seed_cluster_rows]
        )

        for method in ("hard", "distance"):
            accuracies = [row["validation_accuracy"] for row in seed_metric_by_method[method]]
            macro_scores = [row["validation_macro_f1"] for row in seed_metric_by_method[method]]
            accuracy_mean, accuracy_std = summarize(accuracies)
            macro_mean, macro_std = summarize(macro_scores)
            summary_rows.append({
                "K": k,
                "method": method,
                "mean_validation_accuracy": accuracy_mean,
                "std_validation_accuracy": accuracy_std,
                "mean_validation_macro_f1": macro_mean,
                "std_validation_macro_f1": macro_std,
                "mean_cluster_purity": purity_mean,
                "std_cluster_purity": purity_std,
                "mean_weighted_cluster_purity": weighted_purity_mean,
                "std_weighted_cluster_purity": weighted_purity_std,
                "mean_cluster_entropy_bits": entropy_mean,
                "std_cluster_entropy_bits": entropy_std,
                "mean_pairwise_ARI": ari_mean,
                "std_pairwise_ARI": ari_std,
                "mean_pairwise_NMI": nmi_mean,
                "std_pairwise_NMI": nmi_std,
                "mean_kmeans_fit_seconds": float(np.mean(seed_fit_times)),
            })

        codebook_bytes = int(last_centers.nbytes)
        hard_classifier_bytes = parameter_bytes(last_hard_model)
        distance_classifier_bytes = parameter_bytes(last_distance_model)
        id_dtype = np.uint8 if k <= 256 else np.uint16
        hard_payload_bytes = np.dtype(id_dtype).itemsize
        distance_payload_bytes = k * np.dtype(np.float32).itemsize

        for method, payload_bytes, scaler_bytes, classifier_bytes in (
            ("hard", hard_payload_bytes, 0, hard_classifier_bytes),
            (
                "distance", distance_payload_bytes,
                int(last_scaler.mean_.nbytes + last_scaler.scale_.nbytes + last_scaler.var_.nbytes),
                distance_classifier_bytes,
            ),
        ):
            shared_bytes = codebook_bytes + scaler_bytes + classifier_bytes
            amortized = shared_bytes / 50000.0
            total_per_image = payload_bytes + amortized
            storage_rows.append({
                "K": k,
                "method": method,
                "representation_bytes_per_image": payload_bytes,
                "codebook_bytes": codebook_bytes,
                "scaler_bytes": scaler_bytes,
                "classifier_parameter_bytes": classifier_bytes,
                "amortization_images": 50000,
                "amortized_shared_bytes_per_image": amortized,
                "estimated_total_bytes_per_image": total_per_image,
                "compression_vs_continuous_percent": 100.0 * (
                    1.0 - total_per_image / storage_rows[0]["estimated_total_bytes_per_image"]
                ),
            })

    write_csv(args.results_dir / "seed_results.csv", seed_rows)
    write_csv(args.results_dir / "pairwise_stability.csv", pairwise_rows)
    write_csv(args.results_dir / "vocabulary_size_summary.csv", summary_rows)
    write_csv(args.results_dir / "storage_efficiency.csv", storage_rows)
    np.savez_compressed(args.outputs_dir / "codebooks_and_assignments.npz", **saved_arrays)

    summary = {
        "experiment": "Vocabulary size versus stability and efficiency",
        "dataset": "CIFAR-10",
        "feature_extractor": "Frozen DINOv2 ViT-S/14 global CLS features",
        "feature_dimension": FEATURE_DIM,
        "training_images": int(len(y_train)),
        "validation_images": int(len(y_val)),
        "official_test_used": False,
        "k_values": args.ks,
        "seeds": args.seeds,
        "kmeans_n_init": args.n_init,
        "kmeans_max_iter": args.max_iter_kmeans,
        "classifier_settings": {
            "hard_assignment_C": HARD_C,
            "distance_C_by_K": {str(k): DISTANCE_C[k] for k in args.ks},
            "continuous_baseline_C": CONTINUOUS_C,
            "selection": "Fixed C values from the Experiment 3 validation selections; no hyperparameter search in this experiment.",
        },
        "measurements": [
            "validation accuracy and F1 across seeds",
            "pairwise ARI and NMI of training assignments",
            "cluster purity and entropy",
            "representation payload and amortized shared model storage",
            "median CPU encoding plus classifier prediction time",
        ],
        "storage_note": (
            "Per-image feature payload is estimated from the stored representation dtype. "
            "Shared codebook, scaler and classifier array parameters are amortized over 50,000 images. "
            "DINOv2 backbone storage and Python/joblib serialization overhead are excluded."
        ),
        "timing_note": (
            "Inference time is the median of three CPU measurements on the validation features. "
            "DINOv2 feature extraction and model training are excluded."
        ),
        "continuous_baseline": baseline_metrics,
        "total_runtime_seconds": float(time.perf_counter() - start_time),
        "results": {
            "seed_results": "seed_results.csv",
            "pairwise_stability": "pairwise_stability.csv",
            "vocabulary_size_summary": "vocabulary_size_summary.csv",
            "storage_efficiency": "storage_efficiency.csv",
            "codebooks_and_assignments": "codebooks_and_assignments.npz",
        },
    }
    with (args.results_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nVocabulary size summary")
    print("K   method     accuracy (mean +/- std)   purity   ARI     NMI")
    for row in summary_rows:
        print(
            f"{row['K']:<3} {row['method']:<9} "
            f"{row['mean_validation_accuracy']:.4f} +/- {row['std_validation_accuracy']:.4f}   "
            f"{row['mean_cluster_purity']:.4f}   "
            f"{row['mean_pairwise_ARI']:.4f}   {row['mean_pairwise_NMI']:.4f}"
        )
    print(f"\nResults: {args.results_dir}")
    print(f"Outputs: {args.outputs_dir}")
    print("The official CIFAR-10 test split was not loaded or evaluated.")


if __name__ == "__main__":
    main()
