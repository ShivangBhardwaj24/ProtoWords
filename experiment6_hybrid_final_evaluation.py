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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "outputs").is_dir() else SCRIPT_DIR.parent
EXPERIMENT = "exp6_hybrid_final_evaluation"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
N_CLASSES = len(CLASS_NAMES)
DEFAULT_CACHE = Path(
    "outputs/exp1_baseline/"
    "cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz"
)
E5_RESULTS = Path("results/exp5_hybrid_protowords/validation_results.csv")
E5_SUMMARY = Path("results/exp5_hybrid_protowords/experiment_summary.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-iter-kmeans", type=int, default=300)
    parser.add_argument("--max-iter-logreg", type=int, default=2000)
    args = parser.parse_args()

    if args.data.is_absolute():
        args.data = args.data.resolve()
    else:
        args.data = (ROOT / args.data).resolve()
    if min(args.threads, args.batch_size, args.max_iter_kmeans, args.max_iter_logreg) < 1:
        parser.error("Threads, batch size, and iteration limits must be positive")

    args.e5_results = ROOT / E5_RESULTS
    args.e5_summary = ROOT / E5_SUMMARY
    args.results_dir = ROOT / "results" / EXPERIMENT
    args.outputs_dir = ROOT / "outputs" / EXPERIMENT
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    return args


def normalize_features(X):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def load_development_features(path):
    if not path.is_file():
        raise FileNotFoundError(f"DINOv2 feature cache not found: {path}")

    required = {"X_train", "y_train", "X_val", "y_val"}
    with np.load(path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Feature cache is missing: {sorted(missing)}")
        X_train = np.asarray(data["X_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=np.int64)
        X_val = np.asarray(data["X_val"], dtype=np.float32)
        y_val = np.asarray(data["y_val"], dtype=np.int64)

    feature_dim = X_train.shape[1] if X_train.ndim == 2 else 0
    for split, X, y in (("train", X_train, y_train), ("validation", X_val, y_val)):
        if X.ndim != 2 or feature_dim == 0 or X.shape[1] != feature_dim:
            raise ValueError(f"Invalid {split} feature shape: {X.shape}")
        if y.ndim != 1 or len(X) != len(y) or len(X) == 0:
            raise ValueError(f"Invalid {split} labels or feature count")
        if not np.isfinite(X).all() or not np.isin(y, np.arange(N_CLASSES)).all():
            raise ValueError(f"Invalid features or labels in {split} data")
        if set(np.unique(y).tolist()) != set(range(N_CLASSES)):
            raise ValueError(f"All 10 classes must occur in {split} data")

    return normalize_features(X_train), y_train, normalize_features(X_val), y_val


def load_test_features(path, feature_dim):
    with np.load(path, allow_pickle=False) as data:
        required = {"X_test", "y_test"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Feature cache is missing: {sorted(missing)}")
        X_test = np.asarray(data["X_test"], dtype=np.float32)
        y_test = np.asarray(data["y_test"], dtype=np.int64)

    if X_test.ndim != 2 or X_test.shape[1] != feature_dim:
        raise ValueError(f"Invalid test feature shape: {X_test.shape}")
    if y_test.ndim != 1 or len(X_test) != len(y_test) or len(X_test) == 0:
        raise ValueError("Invalid test labels or feature count")
    if not np.isfinite(X_test).all() or not np.isin(y_test, np.arange(N_CLASSES)).all():
        raise ValueError("Invalid features or labels in test data")
    if set(np.unique(y_test).tolist()) != set(range(N_CLASSES)):
        raise ValueError("All 10 classes must occur in test data")
    return normalize_features(X_test), y_test


def load_e5_selection(results_path, summary_path):
    if not results_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "Experiment 5 results were not found. Run Experiment 5 first. "
            f"Expected: {results_path} and {summary_path}"
        )

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    with results_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    best = summary.get("best_validation_configuration", {})
    if best.get("representation") != "hybrid":
        raise ValueError("Experiment 5 summary does not identify a hybrid model as best")

    k = int(summary["K"])
    m = int(summary["m"])
    temperature_multiplier = float(summary["temperature_multiplier"])
    soft_weight = float(best["soft_weight"])

    def find_row(method, weight=None):
        for row in rows:
            if row.get("representation") != method:
                continue
            value = row.get("soft_weight", "").strip()
            if weight is None and value == "":
                return row
            if weight is not None and value and np.isclose(float(value), weight):
                return row
        raise ValueError(f"Could not find Experiment 5 validation row: {method}, {weight}")

    selected = {
        "continuous_dinov2_lr": find_row("continuous_dinov2_lr_baseline"),
        "distance": find_row("distance"),
        "soft": find_row("soft"),
        "hybrid": find_row("hybrid", soft_weight),
    }
    selected_c = {name: float(row["selected_C"]) for name, row in selected.items()}
    if not np.isclose(selected_c["hybrid"], float(best["selected_C"])):
        raise ValueError("Experiment 5 summary and validation CSV disagree on the hybrid C value")

    configurations = {}
    for name, row in selected.items():
        configurations[name] = {
            "selected_C": selected_c[name],
            "validation_accuracy": float(row["validation_accuracy"]),
            "validation_macro_f1": float(row["validation_macro_f1"]),
        }
    configurations["hybrid"]["soft_weight"] = soft_weight

    return {
        "K": k,
        "m": m,
        "temperature_multiplier": temperature_multiplier,
        "seed": int(summary.get("seed", 42)),
        "n_init": int(summary.get("kmeans_n_init", 10)),
        "configurations": configurations,
    }


def build_features(X, centers, m, batch_size, threads):
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


def fit_classifier(X_train, y_train, c_value, max_iter, seed, threads):
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
    return model, convergence_warning


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_predictions(path, y_true, predictions):
    fields = ["true_label", "true_class"]
    fields.extend(f"{name}_prediction" for name in predictions)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, true_label in enumerate(y_true):
            row = {
                "true_label": int(true_label),
                "true_class": CLASS_NAMES[int(true_label)],
            }
            for name, values in predictions.items():
                row[f"{name}_prediction"] = int(values[index])
            writer.writerow(row)


def main():
    args = parse_args()
    start_time = time.perf_counter()
    selection = load_e5_selection(args.e5_results, args.e5_summary)
    k = selection["K"]
    m = selection["m"]
    temperature_multiplier = selection["temperature_multiplier"]
    seed = selection["seed"]
    n_init = selection["n_init"]
    configs = selection["configurations"]

    X_train, y_train, X_val, y_val = load_development_features(args.data)
    X_dev = np.concatenate((X_train, X_val), axis=0)
    y_dev = np.concatenate((y_train, y_val), axis=0)
    if k > len(X_dev) or m > k:
        raise ValueError("Experiment 5 K or m is incompatible with the development data")

    print(f"Training + validation features: {X_dev.shape}")
    print(f"Using E5 selection: K={k}, m={m}, temperature multiplier={temperature_multiplier}")
    print("Fitting a new codebook on all development features")

    kmeans = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
        max_iter=args.max_iter_kmeans,
        algorithm="lloyd",
    )
    with threadpool_limits(limits=args.threads):
        kmeans.fit(X_dev)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)

    dev_distances, dev_ids, dev_squared = build_features(
        X_dev, centers, m, args.batch_size, args.threads
    )
    temperature_reference = float(np.median(dev_squared[:, 1] - dev_squared[:, 0]))
    temperature_reference = max(temperature_reference, 1e-12)
    temperature = temperature_reference * temperature_multiplier
    dev_soft = make_soft_features(dev_ids, dev_squared, k, temperature)

    distance_scaler = StandardScaler()
    dev_distance_scaled = distance_scaler.fit_transform(dev_distances).astype(np.float32, copy=False)
    dev_hybrid = np.hstack((dev_distance_scaled, dev_soft.toarray() * configs["hybrid"]["soft_weight"]))

    models = {}
    convergence_warnings = {}
    for name, X_fit in (
        ("continuous_dinov2_lr", X_dev),
        ("distance", dev_distance_scaled),
        ("soft", dev_soft),
        ("hybrid", dev_hybrid),
    ):
        model, convergence_warning = fit_classifier(
            X_fit, y_dev, configs[name]["selected_C"],
            args.max_iter_logreg, seed, args.threads,
        )
        models[name] = model
        convergence_warnings[name] = convergence_warning
        print(f"Fitted {name} model (C={configs[name]['selected_C']:g})")

    del X_train, X_val, dev_distances, dev_ids, dev_squared, dev_soft
    del dev_distance_scaled, dev_hybrid

    # The official test split is loaded only after all model choices and fitting are complete.
    X_test, y_test = load_test_features(args.data, X_dev.shape[1])
    print(f"Official test features: {X_test.shape}")

    test_distances, test_ids, test_squared = build_features(
        X_test, centers, m, args.batch_size, args.threads
    )
    test_soft = make_soft_features(test_ids, test_squared, k, temperature)
    test_distance_scaled = distance_scaler.transform(test_distances).astype(np.float32, copy=False)
    test_hybrid = np.hstack((test_distance_scaled, test_soft.toarray() * configs["hybrid"]["soft_weight"]))

    test_representations = {
        "continuous_dinov2_lr": X_test,
        "distance": test_distance_scaled,
        "soft": test_soft,
        "hybrid": test_hybrid,
    }
    predictions = {}
    metrics_rows = []
    reports = []
    confusion_matrices = {}

    for name, X_eval in test_representations.items():
        y_pred = models[name].predict(X_eval)
        predictions[name] = y_pred
        metrics = calculate_metrics(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred, labels=np.arange(N_CLASSES))
        confusion_matrices[name] = matrix
        report = classification_report(
            y_test, y_pred, labels=np.arange(N_CLASSES),
            target_names=CLASS_NAMES, digits=4, zero_division=0,
        )
        reports.append(f"{name}\n{'-' * len(name)}\n{report}")
        config = configs[name]
        weight = config.get("soft_weight", "")
        metrics_rows.append({
            "method": name,
            "selected_C": config["selected_C"],
            "soft_weight": weight,
            "validation_accuracy": config["validation_accuracy"],
            "validation_macro_f1": config["validation_macro_f1"],
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "test_micro_f1": metrics["micro_f1"],
            "test_weighted_f1": metrics["weighted_f1"],
            "correct_predictions": int(np.sum(y_test == y_pred)),
            "test_images": int(len(y_test)),
            "convergence_warning": convergence_warnings[name],
        })
        print(
            f"{name:<22} accuracy={metrics['accuracy']:.4f}, "
            f"macro-F1={metrics['macro_f1']:.4f}, C={config['selected_C']:g}"
        )

    metrics_path = args.results_dir / "final_test_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_rows)
    (args.results_dir / "final_test_classification_reports.txt").write_text(
        "\n\n".join(reports), encoding="utf-8"
    )
    np.savez_compressed(
        args.results_dir / "final_test_confusion_matrices.npz",
        **confusion_matrices,
    )
    save_predictions(args.results_dir / "final_test_predictions.csv", y_test, predictions)

    baseline_accuracy = next(
        row["test_accuracy"] for row in metrics_rows if row["method"] == "continuous_dinov2_lr"
    )
    test_results = {}
    for row in metrics_rows:
        test_results[row["method"]] = {
            key: row[key] for key in (
                "test_accuracy", "test_macro_f1", "test_weighted_f1", "correct_predictions"
            )
        }
        test_results[row["method"]]["accuracy_difference_from_continuous_pp"] = round(
            100.0 * (row["test_accuracy"] - baseline_accuracy), 4
        )

    model_bundle = {
        "experiment": EXPERIMENT,
        "feature_extractor": "Frozen DINOv2 ViT-S/14 CLS features",
        "normalization": "L2 per feature vector",
        "K": k,
        "m": m,
        "temperature_multiplier": temperature_multiplier,
        "temperature_reference": temperature_reference,
        "temperature": temperature,
        "soft_weight": configs["hybrid"]["soft_weight"],
        "codebook_centers": centers,
        "distance_scaler": distance_scaler,
        "models": models,
        "class_names": CLASS_NAMES,
        "selected_configs": configs,
    }
    joblib.dump(model_bundle, args.outputs_dir / "final_models.joblib", compress=3)

    summary = {
        "experiment": "Final test evaluation of E5 Hybrid Global ProtoWords",
        "dataset": "CIFAR-10",
        "feature_cache": str(args.data),
        "training_images": int(len(y_train)),
        "validation_images": int(len(y_val)),
        "development_images": int(len(y_dev)),
        "official_test_images": int(len(y_test)),
        "test_use": "Loaded only after all configurations and classifiers were fixed and fitted using development data",
        "refit_protocol": "Fresh K-Means codebook, distance scaler, and classifiers fitted on train + validation data",
        "K": k,
        "m": m,
        "temperature_multiplier": temperature_multiplier,
        "temperature_reference": temperature_reference,
        "temperature": temperature,
        "hybrid_soft_weight": configs["hybrid"]["soft_weight"],
        "selected_configurations_from_E5": configs,
        "test_results": test_results,
        "total_runtime_seconds": round(time.perf_counter() - start_time, 3),
        "results_directory": str(args.results_dir),
        "outputs_directory": str(args.outputs_dir),
    }
    with (args.results_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nFinal test comparison")
    for row in metrics_rows:
        difference = 100.0 * (row["test_accuracy"] - baseline_accuracy)
        print(f"{row['method']:<22} {row['test_accuracy']:.4f} "
              f"({difference:+.2f} percentage points vs continuous baseline)")
    print(f"Results: {args.results_dir}")
    print(f"Outputs: {args.outputs_dir}")
    print("E5 validation selections were fixed before test evaluation.")


if __name__ == "__main__":
    main()
