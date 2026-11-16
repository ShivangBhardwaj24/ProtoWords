from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import Normalizer, StandardScaler
from threadpoolctl import threadpool_limits


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "outputs").is_dir() else SCRIPT_DIR.parent
EXPERIMENT = "exp3_shared_global_protowords"
CACHE_NAME = "cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
DEFAULT_KS = [32, 64, 128, 256]
DEFAULT_C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("outputs/exp1_baseline") / CACHE_NAME)
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--c-values", type=float, nargs="+", default=DEFAULT_C_VALUES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter-kmeans", type=int, default=300)
    parser.add_argument("--max-iter-logreg", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if not args.ks or any(k < 2 for k in args.ks) or len(set(args.ks)) != len(args.ks):
        parser.error("K values must be unique integers greater than or equal to 2")
    if not args.c_values or any(c <= 0 for c in args.c_values):
        parser.error("All C values must be positive")
    if args.n_init < 1 or args.max_iter_kmeans < 1 or args.max_iter_logreg < 1 or args.threads < 1:
        parser.error("Initializations, iteration limits, and threads must be positive")

    args.data = args.data.expanduser()
    if not args.data.is_absolute():
        args.data = ROOT / args.data
    args.results_dir = ROOT / "results" / EXPERIMENT
    args.outputs_dir = ROOT / "outputs" / EXPERIMENT
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    return args


def load_features(path):
    if not path.is_file():
        raise FileNotFoundError(f"DINOv2 feature cache not found: {path}")

    with np.load(path, allow_pickle=False) as cache:
        required = {"X_train", "y_train", "X_val", "y_val"}
        missing = required.difference(cache.files)
        if missing:
            raise KeyError(f"Feature cache is missing: {sorted(missing)}")
        X_train = np.asarray(cache["X_train"], dtype=np.float32)
        y_train = np.asarray(cache["y_train"], dtype=np.int64)
        X_val = np.asarray(cache["X_val"], dtype=np.float32)
        y_val = np.asarray(cache["y_val"], dtype=np.int64)

    for split, X, y in (("train", X_train, y_train), ("validation", X_val, y_val)):
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(X) == 0:
            raise ValueError(f"Invalid {split} feature or label shape")
        if not np.isfinite(X).all() or not np.isin(y, np.arange(10)).all():
            raise ValueError(f"Invalid values or labels in {split} split")
        if set(np.unique(y)) != set(range(10)):
            raise ValueError(f"All ten classes must be present in {split}")

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Train and validation feature dimensions differ")

    normalizer = Normalizer(norm="l2")
    X_train = normalizer.fit_transform(X_train).astype(np.float32, copy=False)
    X_val = normalizer.transform(X_val).astype(np.float32, copy=False)
    return X_train, y_train, X_val, y_val


def make_hard_features(cluster_ids, k):
    rows = np.arange(len(cluster_ids), dtype=np.int32)
    cols = np.asarray(cluster_ids, dtype=np.int32)
    values = np.ones(len(cluster_ids), dtype=np.float32)
    return csr_matrix((values, (rows, cols)), shape=(len(cluster_ids), k), dtype=np.float32)


def make_distance_features(X, centers, batch_size=8192):
    center_norms = np.sum(centers * centers, axis=1)[None, :]
    features = np.empty((len(X), len(centers)), dtype=np.float32)
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        batch_norms = np.sum(batch * batch, axis=1, keepdims=True)
        squared = batch_norms + center_norms - 2.0 * (batch @ centers.T)
        np.maximum(squared, 0.0, out=squared)
        np.sqrt(squared, out=squared)
        features[start:start + len(batch)] = squared
    return features


def fit_classifier(X_train, y_train, X_val, y_val, c_values, max_iter, seed, threads):
    rows = []
    best_model = None
    best_key = None
    best_c = None
    best_predictions = None
    best_metrics = None

    for c_value in c_values:
        model = LogisticRegression(
            C=float(c_value), solver="lbfgs", max_iter=max_iter, random_state=seed
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            with threadpool_limits(limits=threads):
                model.fit(X_train, y_train)
        convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        predictions = model.predict(X_val)
        accuracy = float(accuracy_score(y_val, predictions))
        macro_f1 = float(f1_score(y_val, predictions, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_val, predictions, average="weighted", zero_division=0))
        rows.append({
            "C": float(c_value),
            "validation_accuracy": accuracy,
            "validation_macro_f1": macro_f1,
            "validation_weighted_f1": weighted_f1,
            "convergence_warning": bool(convergence_warning),
        })
        key = (accuracy, macro_f1, -float(c_value))
        if best_key is None or key > best_key:
            best_key = key
            best_model = model
            best_c = float(c_value)
            best_predictions = predictions
            best_metrics = {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "convergence_warning": bool(convergence_warning),
            }

    return best_model, best_c, best_metrics, rows, best_predictions


def load_baseline_reference():
    metrics_path = ROOT / "results" / "exp1_baseline" / "test_metrics.json"
    search_path = ROOT / "results" / "exp1_baseline" / "validation_c_search.csv"
    if not metrics_path.is_file() or not search_path.is_file():
        return None

    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    selected_c = float(metrics["selected_C"])
    with search_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    match = next((row for row in rows if np.isclose(float(row["C"]), selected_c)), None)
    if match is None:
        return None
    return {
        "selected_C": selected_c,
        "validation_accuracy": float(match["validation_accuracy"]),
        "validation_macro_f1": float(match["validation_macro_f1"]),
    }


def main():
    args = parse_args()
    start_time = time.perf_counter()
    X_train, y_train, X_val, y_val = load_features(args.data)

    if max(args.ks) > len(X_train):
        raise ValueError("Largest K cannot exceed the number of training images")

    results = []
    search_results = []
    prediction_rows = []
    confusion_matrices = {}
    codebooks = {}
    model_bundles = {}

    with threadpool_limits(limits=args.threads):
        for k in sorted(args.ks):
            kmeans = KMeans(
                n_clusters=k,
                random_state=args.seed,
                n_init=args.n_init,
                max_iter=args.max_iter_kmeans,
                algorithm="lloyd",
            )
            kmeans.fit(X_train)
            centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
            train_ids = kmeans.labels_.astype(np.int32, copy=False)
            val_ids = kmeans.predict(X_val).astype(np.int32, copy=False)
            cluster_sizes = np.bincount(train_ids, minlength=k)
            codebooks[f"centers_k{k}"] = centers
            codebooks[f"cluster_sizes_k{k}"] = cluster_sizes.astype(np.int64)

            representations = {
                "hard": (make_hard_features(train_ids, k), make_hard_features(val_ids, k)),
            }
            distance_train = make_distance_features(X_train, centers)
            distance_val = make_distance_features(X_val, centers)
            scaler = StandardScaler()
            distance_train = scaler.fit_transform(distance_train).astype(np.float32, copy=False)
            distance_val = scaler.transform(distance_val).astype(np.float32, copy=False)
            representations["distance"] = (distance_train, distance_val)

            for representation, (train_repr, val_repr) in representations.items():
                model, selected_c, metrics, c_rows, predictions = fit_classifier(
                    train_repr, y_train, val_repr, y_val,
                    args.c_values, args.max_iter_logreg, args.seed, args.threads,
                )
                for row in c_rows:
                    search_results.append({"K": k, "representation": representation, **row})

                matrix = confusion_matrix(y_val, predictions, labels=np.arange(10))
                confusion_matrices[f"{representation}_k{k}"] = matrix
                results.append({
                    "K": k,
                    "representation": representation,
                    "selected_C": selected_c,
                    "validation_accuracy": metrics["accuracy"],
                    "validation_macro_f1": metrics["macro_f1"],
                    "validation_weighted_f1": metrics["weighted_f1"],
                    "inertia": float(kmeans.inertia_),
                    "min_cluster_size": int(cluster_sizes.min()),
                    "max_cluster_size": int(cluster_sizes.max()),
                    "empty_clusters": int(np.sum(cluster_sizes == 0)),
                    "convergence_warning": metrics["convergence_warning"],
                })
                for index, (true_label, predicted_label) in enumerate(zip(y_val, predictions)):
                    prediction_rows.append({
                        "validation_index": index,
                        "K": k,
                        "representation": representation,
                        "true_label": int(true_label),
                        "predicted_label": int(predicted_label),
                    })

                model_bundles[f"{representation}_k{k}"] = {
                    "K": k,
                    "representation": representation,
                    "centers": centers,
                    "distance_scaler": scaler if representation == "distance" else None,
                    "classifier": model,
                    "selected_C": selected_c,
                    "validation_metrics": metrics,
                    "class_names": CLASS_NAMES,
                }

            print(f"K={k}: codebook fitted; hard and distance representations evaluated")
            del distance_train, distance_val, representations

    results.sort(key=lambda row: (row["K"], row["representation"]))
    best = max(results, key=lambda row: (row["validation_accuracy"], row["validation_macro_f1"], -row["selected_C"]))
    baseline = load_baseline_reference()

    result_path = args.results_dir / "validation_results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    search_path = args.results_dir / "c_value_search.csv"
    with search_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(search_results[0]))
        writer.writeheader()
        writer.writerows(search_results)

    predictions_path = args.results_dir / "validation_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    np.savez_compressed(args.results_dir / "validation_confusion_matrices.npz", **confusion_matrices)
    np.savez_compressed(args.outputs_dir / "global_codebooks.npz", **codebooks)
    joblib.dump(model_bundles, args.outputs_dir / "global_protowords_models.joblib", compress=3)

    best_key = f"{best['representation']}_k{best['K']}"
    best_predictions = np.asarray([
        row["predicted_label"] for row in prediction_rows
        if row["K"] == best["K"] and row["representation"] == best["representation"]
    ], dtype=np.int64)
    best_report = classification_report(
        y_val, best_predictions, labels=np.arange(10), target_names=CLASS_NAMES,
        digits=4, zero_division=0,
    )
    (args.results_dir / "best_validation_report.txt").write_text(best_report, encoding="utf-8")

    summary = {
        "experiment": EXPERIMENT,
        "dataset": "CIFAR-10",
        "feature_extractor": "frozen DINOv2 ViT-S/14 global CLS features",
        "feature_dimension": int(X_train.shape[1]),
        "training_images": int(len(y_train)),
        "validation_images": int(len(y_val)),
        "normalization": "L2 per image feature vector",
        "codebook_training": "one global Euclidean K-Means codebook fitted on training features only",
        "representations": ["hard one-hot nearest-centroid assignment", "Euclidean distances to all centroids"],
        "distance_scaling": "StandardScaler fitted on training distances only",
        "K_values": sorted(args.ks),
        "C_values": args.c_values,
        "seed": args.seed,
        "kmeans_n_init": args.n_init,
        "kmeans_max_iter": args.max_iter_kmeans,
        "logistic_regression_max_iter": args.max_iter_logreg,
        "official_test_used": False,
        "baseline_validation_reference": baseline,
        "best_validation_configuration": best,
        "total_runtime_seconds": round(time.perf_counter() - start_time, 3),
        "results_file": str(result_path.relative_to(ROOT)),
        "model_file": str((args.outputs_dir / "global_protowords_models.joblib").relative_to(ROOT)),
    }
    with (args.results_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("method       K    accuracy    macro-F1    C")
    for row in results:
        print(f"{row['representation']:<11} {row['K']:<4} {row['validation_accuracy']:.4f}      {row['validation_macro_f1']:.4f}    {row['selected_C']:g}")
    print(f"Best configuration: {best_key}, validation accuracy={best['validation_accuracy']:.4f}")
    if baseline is not None:
        print(f"DINOv2 + LR validation reference: {baseline['validation_accuracy']:.4f}")
    print(f"Results: {args.results_dir}")
    print(f"Outputs: {args.outputs_dir}")
    print("Official CIFAR-10 test data was not loaded or evaluated.")


if __name__ == "__main__":
    main()
