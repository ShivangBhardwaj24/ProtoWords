from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
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
EXPERIMENT = "exp10_prototype_stability"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
N_CLASSES = len(CLASS_NAMES)
DEFAULT_CACHE = Path(
    "outputs/exp1_baseline/"
    "cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure the stability of shared global ProtoWords across random seeds."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--temperature-multiplier", type=float, default=2.0)
    parser.add_argument("--soft-weight", type=float, default=20.0)
    parser.add_argument("--distance-c", type=float, default=0.1)
    parser.add_argument("--hybrid-c", type=float, default=0.1)
    parser.add_argument("--continuous-c", type=float, default=1.0)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter-kmeans", type=int, default=300)
    parser.add_argument("--max-iter-logreg", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.data.is_absolute():
        args.data = args.data.resolve()
    else:
        args.data = (ROOT / args.data).resolve()

    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide at least two different random seeds")
    if args.k < 2 or args.m < 1 or args.m > args.k:
        parser.error("K must be at least 2 and m must be between 1 and K")
    if args.temperature_multiplier <= 0 or args.soft_weight < 0:
        parser.error("Temperature multiplier must be positive and soft weight non-negative")
    if min(args.distance_c, args.hybrid_c, args.continuous_c) <= 0:
        parser.error("Logistic Regression C values must be positive")
    if min(args.n_init, args.max_iter_kmeans, args.max_iter_logreg,
           args.batch_size, args.threads) < 1:
        parser.error("Iteration counts, batch size and threads must be positive")

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

    required = {"X_train", "y_train", "X_val", "y_val"}
    with np.load(path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Feature cache is missing: {sorted(missing)}")
        X_train = np.asarray(data["X_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=np.int64)
        X_val = np.asarray(data["X_val"], dtype=np.float32)
        y_val = np.asarray(data["y_val"], dtype=np.int64)

    for split, X, y in (("train", X_train, y_train), ("validation", X_val, y_val)):
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(X) == 0:
            raise ValueError(f"Invalid {split} feature or label shape")
        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite features found in {split} data")
        if not np.isin(y, np.arange(N_CLASSES)).all():
            raise ValueError(f"Invalid CIFAR-10 labels in {split} data")
        if set(np.unique(y).tolist()) != set(range(N_CLASSES)):
            raise ValueError(f"All 10 classes must appear in {split} data")

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Train and validation feature dimensions do not match")
    return normalize_features(X_train), y_train, normalize_features(X_val), y_val


def build_features(X, centers, m, batch_size, threads):
    n_samples = len(X)
    k = len(centers)
    distances = np.empty((n_samples, k), dtype=np.float32)
    nearest_ids = np.empty((n_samples, m), dtype=np.int32)
    nearest_squared = np.empty((n_samples, m), dtype=np.float32)
    center_norms = np.sum(centers * centers, axis=1)[None, :]

    with threadpool_limits(limits=threads):
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = X[start:end]
            squared = (
                np.sum(batch * batch, axis=1, keepdims=True)
                + center_norms
                - 2.0 * (batch @ centers.T)
            )
            np.maximum(squared, 0.0, out=squared)
            ids = np.argpartition(squared, kth=m - 1, axis=1)[:, :m]
            selected = np.take_along_axis(squared, ids, axis=1)
            order = np.argsort(selected, axis=1)
            nearest_ids[start:end] = np.take_along_axis(ids, order, axis=1)
            nearest_squared[start:end] = np.take_along_axis(selected, order, axis=1)
            np.sqrt(squared, out=squared)
            distances[start:end] = squared

    return distances, nearest_ids, nearest_squared


def make_soft_features(ids, squared_distances, k, temperature):
    shifted = squared_distances - squared_distances[:, :1]
    weights = np.exp(-shifted / max(float(temperature), 1e-12)).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    output = np.zeros((len(ids), k), dtype=np.float32)
    rows = np.arange(len(ids), dtype=np.int32)[:, None]
    output[rows, ids] = weights
    if not np.allclose(output.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Soft-assignment weights do not sum to one")
    return output


def cluster_statistics(assignments, labels, k):
    counts = np.zeros((k, N_CLASSES), dtype=np.int64)
    np.add.at(counts, (assignments, labels), 1)
    sizes = counts.sum(axis=1)
    nonempty = sizes > 0
    purity_by_cluster = np.divide(
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
    entropy_by_cluster = -(probabilities * log_probabilities).sum(axis=1)

    result = {
        "empty_prototypes": int(np.sum(~nonempty)),
        "smallest_cluster": int(sizes.min()),
        "largest_cluster": int(sizes.max()),
        "mean_cluster_purity": float(purity_by_cluster[nonempty].mean()),
        "weighted_cluster_purity": float(counts.max(axis=1).sum() / max(sizes.sum(), 1)),
        "mean_cluster_entropy_bits": float(entropy_by_cluster[nonempty].mean()),
        "mean_normalized_entropy": float((entropy_by_cluster[nonempty] / np.log2(N_CLASSES)).mean()),
    }
    return result, counts, sizes


def fit_classifier(X_train, y_train, X_val, y_val, c_value, max_iter, seed, threads):
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


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    start_time = time.perf_counter()
    X_train, y_train, X_val, y_val = load_features(args.data)

    if args.k > len(X_train):
        raise ValueError("K cannot exceed the number of training images")

    print(f"Training features: {X_train.shape}")
    print(f"Validation features: {X_val.shape}")
    print(f"K={args.k}, m={args.m}, seeds={args.seeds}")
    print("Fitting K-Means on training features only; validation is used only for evaluation")

    continuous_model, continuous_metrics, continuous_warning = fit_classifier(
        X_train, y_train, X_val, y_val,
        args.continuous_c, args.max_iter_logreg, args.seeds[0], args.threads,
    )
    print(
        f"Continuous baseline: accuracy={continuous_metrics['validation_accuracy']:.4f}, "
        f"Macro-F1={continuous_metrics['validation_macro_f1']:.4f}"
    )

    assignments_by_seed = {}
    seed_rows = []
    model_bundles = {}
    pairwise_rows = []
    seed_assignments = {}

    for seed in args.seeds:
        print(f"\nSeed {seed}: fitting global codebook")
        kmeans = KMeans(
            n_clusters=args.k,
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

        train_distances, train_ids, train_near_sq = build_features(
            X_train, centers, args.m, args.batch_size, args.threads
        )
        val_distances, val_ids, val_near_sq = build_features(
            X_val, centers, args.m, args.batch_size, args.threads
        )

        temperature_reference = float(np.median(train_near_sq[:, 1] - train_near_sq[:, 0])) if args.m > 1 else 1.0
        temperature_reference = max(temperature_reference, 1e-12)
        temperature = temperature_reference * args.temperature_multiplier
        train_soft = make_soft_features(train_ids, train_near_sq, args.k, temperature)
        val_soft = make_soft_features(val_ids, val_near_sq, args.k, temperature)

        scaler = StandardScaler()
        train_distance_scaled = scaler.fit_transform(train_distances).astype(np.float32, copy=False)
        val_distance_scaled = scaler.transform(val_distances).astype(np.float32, copy=False)
        train_hybrid = np.hstack((train_distance_scaled, train_soft * args.soft_weight))
        val_hybrid = np.hstack((val_distance_scaled, val_soft * args.soft_weight))

        distance_model, distance_metrics, distance_warning = fit_classifier(
            train_distance_scaled, y_train, val_distance_scaled, y_val,
            args.distance_c, args.max_iter_logreg, seed, args.threads,
        )
        hybrid_model, hybrid_metrics, hybrid_warning = fit_classifier(
            train_hybrid, y_train, val_hybrid, y_val,
            args.hybrid_c, args.max_iter_logreg, seed, args.threads,
        )

        train_assignments = kmeans.labels_.astype(np.int32, copy=False)
        val_assignments = kmeans.predict(X_val).astype(np.int32, copy=False)
        all_assignments = np.concatenate((train_assignments, val_assignments))
        all_labels = np.concatenate((y_train, y_val))
        cluster_metrics, class_counts, cluster_sizes = cluster_statistics(
            all_assignments, all_labels, args.k
        )

        assignments_by_seed[seed] = all_assignments
        seed_assignments[f"assignments_seed_{seed}"] = all_assignments
        seed_rows.append({
            "seed": seed,
            "k": args.k,
            "m": args.m,
            "temperature_reference": temperature_reference,
            "temperature": temperature,
            "soft_weight": args.soft_weight,
            "kmeans_fit_seconds": round(kmeans_seconds, 4),
            **cluster_metrics,
            "continuous_validation_accuracy": continuous_metrics["validation_accuracy"],
            "continuous_validation_macro_f1": continuous_metrics["validation_macro_f1"],
            "distance_validation_accuracy": distance_metrics["validation_accuracy"],
            "distance_validation_macro_f1": distance_metrics["validation_macro_f1"],
            "distance_convergence_warning": distance_warning,
            "hybrid_validation_accuracy": hybrid_metrics["validation_accuracy"],
            "hybrid_validation_macro_f1": hybrid_metrics["validation_macro_f1"],
            "hybrid_convergence_warning": hybrid_warning,
        })
        model_bundles[str(seed)] = {
            "seed": seed,
            "codebook_centers": centers,
            "distance_scaler": scaler,
            "distance_model": distance_model,
            "hybrid_model": hybrid_model,
            "temperature_reference": temperature_reference,
            "temperature": temperature,
            "K": args.k,
            "m": args.m,
            "soft_weight": args.soft_weight,
            "distance_C": args.distance_c,
            "hybrid_C": args.hybrid_c,
            "cluster_class_counts": class_counts,
            "cluster_sizes": cluster_sizes,
        }

        print(
            f"Seed {seed}: purity={cluster_metrics['mean_cluster_purity']:.4f}, "
            f"entropy={cluster_metrics['mean_cluster_entropy_bits']:.4f}, "
            f"distance acc={distance_metrics['validation_accuracy']:.4f}, "
            f"hybrid acc={hybrid_metrics['validation_accuracy']:.4f}"
        )

        del train_distances, val_distances, train_near_sq, val_near_sq
        del train_soft, val_soft, train_distance_scaled, val_distance_scaled
        del train_hybrid, val_hybrid

    seed_values = list(args.seeds)
    for i, seed_a in enumerate(seed_values):
        for seed_b in seed_values[i + 1:]:
            labels_a = assignments_by_seed[seed_a]
            labels_b = assignments_by_seed[seed_b]
            pairwise_rows.append({
                "seed_a": seed_a,
                "seed_b": seed_b,
                "adjusted_rand_index": float(adjusted_rand_score(labels_a, labels_b)),
                "normalized_mutual_information": float(
                    normalized_mutual_info_score(labels_a, labels_b, average_method="arithmetic")
                ),
            })

    write_csv(args.results_dir / "seed_results.csv", seed_rows)
    write_csv(args.results_dir / "pairwise_stability.csv", pairwise_rows)

    distance_accuracies = np.asarray([row["distance_validation_accuracy"] for row in seed_rows])
    hybrid_accuracies = np.asarray([row["hybrid_validation_accuracy"] for row in seed_rows])
    purity_values = np.asarray([row["mean_cluster_purity"] for row in seed_rows])
    entropy_values = np.asarray([row["mean_cluster_entropy_bits"] for row in seed_rows])
    ari_values = np.asarray([row["adjusted_rand_index"] for row in pairwise_rows])
    nmi_values = np.asarray([row["normalized_mutual_information"] for row in pairwise_rows])

    def mean_std(values):
        return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}

    summary = {
        "experiment": "Shared global ProtoWords stability across random seeds",
        "dataset": "CIFAR-10",
        "feature_extractor": "Frozen DINOv2 ViT-S/14 global CLS features",
        "feature_cache": str(args.data),
        "training_images": int(len(y_train)),
        "validation_images": int(len(y_val)),
        "official_test_used": False,
        "fit_protocol": "A separate global K-Means codebook was fitted on the same training features for each seed; validation features were used only for evaluation.",
        "k": args.k,
        "seeds": [int(seed) for seed in args.seeds],
        "n_init": args.n_init,
        "max_iter_kmeans": args.max_iter_kmeans,
        "soft_assignment": {
            "active_prototypes": args.m,
            "temperature_multiplier": args.temperature_multiplier,
            "temperature_reference": "Median training-set gap between squared distances to the nearest and second-nearest prototypes, recomputed for each seed.",
            "hybrid_soft_weight": args.soft_weight,
        },
        "fixed_classifier_settings": {
            "continuous_dinov2_lr_C": args.continuous_c,
            "distance_C": args.distance_c,
            "hybrid_C": args.hybrid_c,
            "selection": "No hyperparameter search. C values and representation settings were fixed before this experiment.",
        },
        "continuous_validation_baseline": {
            **continuous_metrics,
            "C": args.continuous_c,
            "convergence_warning": continuous_warning,
        },
        "stability_summary": {
            "pairwise_ARI": mean_std(ari_values),
            "pairwise_NMI": mean_std(nmi_values),
            "mean_cluster_purity": mean_std(purity_values),
            "mean_cluster_entropy_bits": mean_std(entropy_values),
        },
        "classification_summary": {
            "distance_validation_accuracy": mean_std(distance_accuracies),
            "hybrid_validation_accuracy": mean_std(hybrid_accuracies),
            "distance_validation_macro_f1": mean_std(np.asarray([r["distance_validation_macro_f1"] for r in seed_rows])),
            "hybrid_validation_macro_f1": mean_std(np.asarray([r["hybrid_validation_macro_f1"] for r in seed_rows])),
        },
        "total_runtime_seconds": round(time.perf_counter() - start_time, 3),
        "results_directory": str(args.results_dir),
        "outputs_directory": str(args.outputs_dir),
    }
    with (args.results_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    seed_assignments["y_train_val"] = np.concatenate((y_train, y_val))
    np.savez_compressed(args.outputs_dir / "seed_assignments.npz", **seed_assignments)
    joblib.dump({
        "experiment": EXPERIMENT,
        "settings": {
            "K": args.k,
            "m": args.m,
            "temperature_multiplier": args.temperature_multiplier,
            "soft_weight": args.soft_weight,
            "distance_C": args.distance_c,
            "hybrid_C": args.hybrid_c,
            "seeds": args.seeds,
        },
        "continuous_baseline": continuous_model,
        "seed_models": model_bundles,
    }, args.outputs_dir / "seed_models.joblib", compress=3)

    print("\nPairwise cluster stability")
    print(f"ARI: {ari_values.mean():.4f} +/- {ari_values.std(ddof=1):.4f}")
    print(f"NMI: {nmi_values.mean():.4f} +/- {nmi_values.std(ddof=1):.4f}")
    print("\nAcross-seed summary")
    print(f"Mean purity: {purity_values.mean():.4f} +/- {purity_values.std(ddof=1):.4f}")
    print(f"Mean entropy: {entropy_values.mean():.4f} +/- {entropy_values.std(ddof=1):.4f} bits")
    print(f"Distance validation accuracy: {distance_accuracies.mean():.4f} +/- {distance_accuracies.std(ddof=1):.4f}")
    print(f"Hybrid validation accuracy: {hybrid_accuracies.mean():.4f} +/- {hybrid_accuracies.std(ddof=1):.4f}")
    print(f"Continuous baseline validation accuracy: {continuous_metrics['validation_accuracy']:.4f}")
    print(f"Results: {args.results_dir}")
    print(f"Outputs: {args.outputs_dir}")
    print("The official CIFAR-10 test split was not loaded or evaluated.")


if __name__ == "__main__":
    main()
