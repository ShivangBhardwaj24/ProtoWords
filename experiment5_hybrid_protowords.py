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
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "outputs").is_dir() else SCRIPT_DIR.parent
EXPERIMENT = "exp5_hybrid_protowords"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
N_CLASSES = len(CLASS_NAMES)
FEATURE_DIM = 384
BASELINE_C = 1.0
DEFAULT_CACHE = Path(
    "outputs/exp1_baseline/"
    "cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--temperature-multiplier", type=float, default=2.0)
    parser.add_argument("--soft-weights", type=float, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1, 10, 100])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter-kmeans", type=int, default=300)
    parser.add_argument("--max-iter-logreg", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.data.is_absolute():
        args.data = args.data.resolve()
    else:
        args.data = (ROOT / args.data).resolve()

    args.soft_weights = sorted(set(args.soft_weights))
    args.c_values = sorted(set(args.c_values))
    if args.k < 2 or args.m < 1 or args.m > args.k:
        parser.error("K must be at least 2 and m must be between 1 and K")
    if args.temperature_multiplier <= 0:
        parser.error("Temperature multiplier must be positive")
    if not args.soft_weights or any(value <= 0 for value in args.soft_weights):
        parser.error("Soft weights must be positive")
    if not args.c_values or any(value <= 0 for value in args.c_values):
        parser.error("C values must be positive")
    if min(args.n_init, args.max_iter_kmeans, args.max_iter_logreg,
           args.batch_size, args.threads) < 1:
        parser.error("Iteration limits, batch size, and threads must be positive")

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
        raise FileNotFoundError(f"DINOv2 feature cache not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {"X_train", "y_train", "X_val", "y_val"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Feature cache is missing: {sorted(missing)}")
        X_train = np.asarray(data["X_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=np.int64)
        X_val = np.asarray(data["X_val"], dtype=np.float32)
        y_val = np.asarray(data["y_val"], dtype=np.int64)

    for split, X, y in (("train", X_train, y_train), ("validation", X_val, y_val)):
        if X.ndim != 2 or X.shape[1] != FEATURE_DIM:
            raise ValueError(f"Unexpected {split} feature shape: {X.shape}")
        if y.ndim != 1 or len(X) != len(y) or len(X) == 0:
            raise ValueError(f"Invalid {split} labels or feature count")
        if not np.isfinite(X).all() or not np.isin(y, np.arange(N_CLASSES)).all():
            raise ValueError(f"Invalid values or labels in {split} data")
        if set(np.unique(y).tolist()) != set(range(N_CLASSES)):
            raise ValueError(f"All CIFAR-10 classes must occur in {split} data")

    return normalize_features(X_train), y_train, normalize_features(X_val), y_val


def make_features(X, centers, m, batch_size, threads):
    n_samples = len(X)
    n_centers = len(centers)
    distances = np.empty((n_samples, n_centers), dtype=np.float32)
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

    rows = np.repeat(np.arange(len(ids), dtype=np.int32), ids.shape[1])
    return csr_matrix(
        (weights.reshape(-1), (rows, ids.reshape(-1))),
        shape=(len(ids), k),
        dtype=np.float32,
    )


def calculate_metrics(y_true, y_pred):
    return {
        "validation_accuracy": float(accuracy_score(y_true, y_pred)),
        "validation_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "validation_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_representation(name, weight, X_train, y_train, X_val, y_val, c_values,
                             max_iter, seed, threads):
    search_rows = []
    best = None

    for c_value in c_values:
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
        metrics = calculate_metrics(y_val, predictions)
        row = {
            "representation": name,
            "soft_weight": "" if weight is None else float(weight),
            "C": float(c_value),
            **metrics,
            "convergence_warning": convergence_warning,
        }
        search_rows.append(row)
        key = (metrics["validation_accuracy"], metrics["validation_macro_f1"], -float(c_value))
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "model": model,
                "predictions": predictions,
                "C": float(c_value),
                "metrics": metrics,
                "convergence_warning": convergence_warning,
            }

    result = {
        "representation": name,
        "soft_weight": "" if weight is None else float(weight),
        "selected_C": best["C"],
        **best["metrics"],
        "convergence_warning": best["convergence_warning"],
    }
    return result, search_rows, best


def save_confusion_matrix(path, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(N_CLASSES))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + CLASS_NAMES)
        for name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([name] + row.tolist())


def save_predictions(path, y_true, y_pred):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["validation_index", "true_label", "true_class", "predicted_label", "predicted_class"])
        for index, (true_label, predicted_label) in enumerate(zip(y_true, y_pred)):
            writer.writerow([
                index, int(true_label), CLASS_NAMES[int(true_label)],
                int(predicted_label), CLASS_NAMES[int(predicted_label)],
            ])


def main():
    args = parse_args()
    start_time = time.perf_counter()
    X_train, y_train, X_val, y_val = load_features(args.data)

    if args.k > len(X_train):
        raise ValueError("K cannot exceed the number of training images")

    print(f"Training features: {X_train.shape}")
    print(f"Validation features: {X_val.shape}")
    print(f"Vocabulary size: {args.k}")

    kmeans = KMeans(
        n_clusters=args.k,
        random_state=args.seed,
        n_init=args.n_init,
        max_iter=args.max_iter_kmeans,
        algorithm="lloyd",
    )
    with threadpool_limits(limits=args.threads):
        kmeans.fit(X_train)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)

    train_distances, train_ids, train_squared = make_features(
        X_train, centers, args.m, args.batch_size, args.threads
    )
    val_distances, val_ids, val_squared = make_features(
        X_val, centers, args.m, args.batch_size, args.threads
    )

    if args.m == 1:
        temperature_reference = 1.0
    else:
        temperature_reference = float(np.median(train_squared[:, 1] - train_squared[:, 0]))
        temperature_reference = max(temperature_reference, 1e-12)
    temperature = temperature_reference * args.temperature_multiplier

    train_soft = make_soft_features(train_ids, train_squared, args.k, temperature)
    val_soft = make_soft_features(val_ids, val_squared, args.k, temperature)

    baseline_model = LogisticRegression(
        C=BASELINE_C, solver="lbfgs", max_iter=args.max_iter_logreg, random_state=args.seed
    )
    with threadpool_limits(limits=args.threads):
        baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_val)
    baseline_metrics = calculate_metrics(y_val, baseline_predictions)
    baseline_row = {
        "representation": "continuous_dinov2_lr_baseline",
        "soft_weight": "",
        "selected_C": BASELINE_C,
        **baseline_metrics,
        "convergence_warning": False,
    }

    scaler = StandardScaler()
    train_distance_scaled = scaler.fit_transform(train_distances).astype(np.float32, copy=False)
    val_distance_scaled = scaler.transform(val_distances).astype(np.float32, copy=False)

    configurations = []
    searches = []
    best_overall = None

    result, rows, best = evaluate_representation(
        "distance", None, train_distance_scaled, y_train,
        val_distance_scaled, y_val, args.c_values,
        args.max_iter_logreg, args.seed, args.threads,
    )
    configurations.append(result)
    searches.extend(rows)
    best_overall = {"result": result, "best": best, "weight": None}

    result, rows, best = evaluate_representation(
        "soft", None, train_soft, y_train, val_soft, y_val,
        args.c_values, args.max_iter_logreg, args.seed, args.threads,
    )
    configurations.append(result)
    searches.extend(rows)
    key = (result["validation_accuracy"], result["validation_macro_f1"], 0, -result["selected_C"])
    current_key = (
        best_overall["result"]["validation_accuracy"],
        best_overall["result"]["validation_macro_f1"],
        0, -best_overall["result"]["selected_C"],
    )
    if key > current_key:
        best_overall = {"result": result, "best": best, "weight": None}

    for weight in args.soft_weights:
        train_hybrid = np.hstack((train_distance_scaled, train_soft.toarray() * weight))
        val_hybrid = np.hstack((val_distance_scaled, val_soft.toarray() * weight))
        result, rows, best = evaluate_representation(
            "hybrid", weight, train_hybrid, y_train, val_hybrid, y_val,
            args.c_values, args.max_iter_logreg, args.seed, args.threads,
        )
        configurations.append(result)
        searches.extend(rows)
        key = (
            result["validation_accuracy"], result["validation_macro_f1"],
            -float(weight), -result["selected_C"],
        )
        current = best_overall["result"]
        current_key = (
            current["validation_accuracy"], current["validation_macro_f1"],
            -float(best_overall["weight"] or 0), -current["selected_C"],
        )
        if key > current_key:
            best_overall = {"result": result, "best": best, "weight": float(weight)}
        del train_hybrid, val_hybrid

    configurations.append(baseline_row)
    configurations.sort(key=lambda row: (
        -row["validation_accuracy"], -row["validation_macro_f1"],
        row["representation"], float(row["soft_weight"] or 0),
    ))
    searches.sort(key=lambda row: (
        row["representation"], float(row["soft_weight"] or 0), row["C"]
    ))

    results_path = args.results_dir / "validation_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(configurations[0]))
        writer.writeheader()
        writer.writerows(configurations)

    search_path = args.results_dir / "hyperparameter_search.csv"
    with search_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(searches[0]))
        writer.writeheader()
        writer.writerows(searches)

    best_result = best_overall["result"]
    best_predictions = best_overall["best"]["predictions"]
    report = classification_report(
        y_val, best_predictions, labels=np.arange(N_CLASSES),
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    )
    (args.results_dir / "best_validation_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(
        args.results_dir / "best_validation_confusion_matrix.csv", y_val, best_predictions
    )
    save_predictions(
        args.results_dir / "best_validation_predictions.csv", y_val, best_predictions
    )

    bundle = {
        "experiment": EXPERIMENT,
        "method": best_result["representation"],
        "K": args.k,
        "m": args.m,
        "temperature_multiplier": args.temperature_multiplier,
        "temperature_reference": temperature_reference,
        "temperature": temperature,
        "soft_weight": best_overall["weight"],
        "distance_scaler": scaler,
        "codebook_centers": centers,
        "classifier": best_overall["best"]["model"],
        "selected_C": best_result["selected_C"],
        "validation_metrics": {
            "accuracy": best_result["validation_accuracy"],
            "macro_f1": best_result["validation_macro_f1"],
            "weighted_f1": best_result["validation_weighted_f1"],
        },
        "class_names": CLASS_NAMES,
    }
    joblib.dump(bundle, args.outputs_dir / "best_validation_model.joblib", compress=3)

    summary = {
        "experiment": "Hybrid Global ProtoWords",
        "dataset": "CIFAR-10",
        "feature_extractor": "Frozen DINOv2 ViT-S/14 CLS features",
        "feature_dimension": FEATURE_DIM,
        "feature_cache": str(args.data),
        "training_images": int(len(y_train)),
        "validation_images": int(len(y_val)),
        "official_test_used": False,
        "normalization": "L2 normalization per image feature",
        "codebook": "One global K-Means codebook fitted on training features only",
        "K": args.k,
        "soft_assignment": "Softmax over negative squared distances to the m nearest prototypes",
        "m": args.m,
        "temperature_reference": "Median training-set gap between the nearest and second-nearest squared distances",
        "temperature_multiplier": args.temperature_multiplier,
        "soft_block_weights": args.soft_weights,
        "distance_processing": "StandardScaler fitted on training distances only",
        "hybrid_representation": "Concatenation of standardized centroid distances and weighted localized soft assignments",
        "C_values": args.c_values,
        "continuous_baseline_C": BASELINE_C,
        "continuous_baseline_validation": baseline_metrics,
        "seed": args.seed,
        "kmeans_n_init": args.n_init,
        "best_validation_configuration": best_result,
        "total_runtime_seconds": round(time.perf_counter() - start_time, 3),
        "results_file": "validation_results.csv",
        "search_file": "hyperparameter_search.csv",
    }
    with (args.results_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nValidation results")
    print("method    soft_weight   accuracy   macro-F1   C")
    for row in configurations:
        weight = "-" if row["soft_weight"] == "" else f"{row['soft_weight']:g}"
        print(
            f"{row['representation']:<9} {weight:<12} "
            f"{row['validation_accuracy']:.4f}     "
            f"{row['validation_macro_f1']:.4f}   {row['selected_C']:g}"
        )
    print(
        f"Best ProtoWords configuration: {best_result['representation']}, "
        f"soft_weight={best_result['soft_weight']}, "
        f"accuracy={best_result['validation_accuracy']:.4f}, "
        f"Macro-F1={best_result['validation_macro_f1']:.4f}"
    )
    print(
        f"DINOv2 + LR baseline: {baseline_metrics['validation_accuracy']:.4f} "
        f"accuracy, {baseline_metrics['validation_macro_f1']:.4f} Macro-F1 (C={BASELINE_C})"
    )
    print(f"Results: {args.results_dir}")
    print(f"Outputs: {args.outputs_dir}")
    print("Official CIFAR-10 test data was not loaded or evaluated.")


if __name__ == "__main__":
    main()
