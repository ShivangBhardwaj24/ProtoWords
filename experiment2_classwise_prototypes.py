from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "outputs").is_dir() else SCRIPT_DIR.parent
EXPERIMENT = "exp2_classwise_prototypes"
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    if args.data.is_absolute():
        args.data = args.data.resolve()
    else:
        args.data = (ROOT / args.data).resolve()

    args.ks = sorted(set(args.ks))
    if not args.ks or any(k < 1 for k in args.ks):
        parser.error("K values must be positive integers")
    if args.n_init < 1 or args.max_iter < 1 or args.batch_size < 1:
        parser.error("n-init, max-iter, and batch-size must be positive")

    args.results_dir = ROOT / "results" / EXPERIMENT
    args.outputs_dir = ROOT / "outputs" / EXPERIMENT
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    return args


def l2_normalize(X):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def load_features(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature cache not found: {path}\n"
            "Run Experiment 1 first, or pass the cache path with --data."
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

    expected_shapes = {
        "X_train": (45000, FEATURE_DIM),
        "X_val": (5000, FEATURE_DIM),
        "y_train": (45000,),
        "y_val": (5000,),
    }
    actual_arrays = {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }
    for name, expected in expected_shapes.items():
        if actual_arrays[name].shape != expected:
            raise ValueError(
                f"Expected {name} shape {expected}, got {actual_arrays[name].shape}"
            )

    for name in ("X_train", "X_val"):
        if not np.isfinite(actual_arrays[name]).all():
            raise ValueError(f"{name} contains NaN or infinite values")

    for name in ("y_train", "y_val"):
        if not np.isin(actual_arrays[name], np.arange(N_CLASSES)).all():
            raise ValueError(f"{name} must contain labels from 0 to 9")

    train_counts = np.bincount(y_train, minlength=N_CLASSES)
    val_counts = np.bincount(y_val, minlength=N_CLASSES)
    if not np.all(train_counts == 4500):
        raise ValueError("Expected 4,500 training examples per class")
    if not np.all(val_counts == 500):
        raise ValueError("Expected 500 validation examples per class")

    return l2_normalize(X_train), y_train, l2_normalize(X_val), y_val


def load_test_features(path):
    with np.load(path, allow_pickle=False) as data:
        required = {"X_test", "y_test"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Feature cache is missing: {sorted(missing)}")
        X_test = np.asarray(data["X_test"], dtype=np.float32)
        y_test = np.asarray(data["y_test"], dtype=np.int64)

    if X_test.shape != (10000, FEATURE_DIM) or y_test.shape != (10000,):
        raise ValueError("Expected official test features (10000, 384) and labels (10000,)")
    if not np.isfinite(X_test).all():
        raise ValueError("Test features contain NaN or infinite values")
    if not np.isin(y_test, np.arange(N_CLASSES)).all():
        raise ValueError("Test labels must be integers from 0 to 9")
    if not np.all(np.bincount(y_test, minlength=N_CLASSES) == 1000):
        raise ValueError("Expected 1,000 official test examples per class")

    return l2_normalize(X_test), y_test


def fit_classwise_prototypes(X_train, y_train, k, seed, n_init, max_iter):
    centers_by_class = []
    labels_by_class = []
    warning_count = 0
    start = time.perf_counter()

    for class_id in range(N_CLASSES):
        X_class = X_train[y_train == class_id]
        if len(X_class) < k:
            raise ValueError(
                f"Class {CLASS_NAMES[class_id]} has fewer than K={k} samples"
            )

        if k == 1:
            centers = X_class.mean(axis=0, keepdims=True)
        else:
            model = KMeans(
                n_clusters=k,
                random_state=seed + class_id,
                n_init=n_init,
                max_iter=max_iter,
                algorithm="lloyd",
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(X_class)
            warning_count += len(caught)
            centers = model.cluster_centers_

        centers_by_class.append(np.asarray(centers, dtype=np.float32))
        labels_by_class.extend([class_id] * len(centers))

    prototypes = np.vstack(centers_by_class).astype(np.float32, copy=False)
    prototype_labels = np.asarray(labels_by_class, dtype=np.int64)
    return prototypes, prototype_labels, time.perf_counter() - start, warning_count


def predict_euclidean(X, prototypes, prototype_labels, batch_size):
    prototype_norms = np.sum(prototypes * prototypes, axis=1)[None, :]
    predictions = []
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        batch_norms = np.sum(batch * batch, axis=1, keepdims=True)
        distances = batch_norms + prototype_norms - 2.0 * (batch @ prototypes.T)
        np.maximum(distances, 0.0, out=distances)
        predictions.append(prototype_labels[np.argmin(distances, axis=1)])
    return np.concatenate(predictions)


def predict_cosine(X, prototypes, prototype_labels, batch_size):
    normalized_prototypes = l2_normalize(prototypes)
    predictions = []
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        similarities = batch @ normalized_prototypes.T
        predictions.append(prototype_labels[np.argmax(similarities, axis=1)])
    return np.concatenate(predictions)


def get_metrics(y_true, y_pred, split):
    return {
        f"{split}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{split}_macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        f"{split}_weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def save_predictions(path, y_true, predictions):
    methods = list(predictions)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["image_index", "true_label", "true_class"]
        for method in methods:
            header.extend([f"{method}_predicted_label", f"{method}_predicted_class"])
        writer.writerow(header)

        for index, true_label in enumerate(y_true):
            row = [index, int(true_label), CLASS_NAMES[int(true_label)]]
            for method in methods:
                pred_label = int(predictions[method][index])
                row.extend([pred_label, CLASS_NAMES[pred_label]])
            writer.writerow(row)


def save_confusion_matrix(path, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(N_CLASSES))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + CLASS_NAMES)
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name] + row.tolist())


def load_baseline_reference(path):
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None
    best = max(
        rows,
        key=lambda row: (
            float(row["validation_accuracy"]),
            float(row.get("validation_macro_f1", 0.0)),
            -float(row["C"]),
        ),
    )
    return {
        "method": "DINOv2 + Logistic Regression",
        "selected_C": float(best["C"]),
        "validation_accuracy": float(best["validation_accuracy"]),
        "validation_macro_f1": float(best.get("validation_macro_f1", 0.0)),
    }


def main():
    args = parse_args()
    X_train, y_train, X_val, y_val = load_features(args.data)

    if max(args.ks) > int(np.bincount(y_train, minlength=N_CLASSES).min()):
        raise ValueError(
            "A requested K exceeds the number of training examples in at least one class"
        )

    baseline_path = ROOT / "results" / "exp1_baseline" / "validation_c_search.csv"
    baseline = load_baseline_reference(baseline_path)

    result_rows = []
    saved_prototypes = {}
    best_predictions = {"euclidean": None, "cosine": None}
    best_rows = {"euclidean": None, "cosine": None}

    for k in args.ks:
        prototypes, prototype_labels, fit_seconds, warning_count = (
            fit_classwise_prototypes(
                X_train, y_train, k, args.seed, args.n_init, args.max_iter
            )
        )
        saved_prototypes[f"prototypes_k{k}"] = prototypes
        saved_prototypes[f"labels_k{k}"] = prototype_labels

        predictions = {
            "euclidean": predict_euclidean(
                X_val, prototypes, prototype_labels, args.batch_size
            ),
            "cosine": predict_cosine(
                X_val, prototypes, prototype_labels, args.batch_size
            ),
        }

        for method, y_pred in predictions.items():
            metrics = get_metrics(y_val, y_pred, "validation")
            row = {
                "method": method,
                "prototypes_per_class": k,
                "total_prototypes": len(prototypes),
                **metrics,
                "prototype_fit_seconds": round(fit_seconds, 4),
                "kmeans_warning_count": warning_count,
            }
            result_rows.append(row)

            selection_key = (
                metrics["validation_accuracy"],
                metrics["validation_macro_f1"],
                -k,
            )
            current = best_rows[method]
            if current is None or selection_key > current["selection_key"]:
                best_rows[method] = {
                    **row,
                    "selection_key": selection_key,
                }
                best_predictions[method] = y_pred.copy()

    if baseline is not None:
        result_rows.append({
            "method": "DINOv2 + Logistic Regression",
            "prototypes_per_class": "",
            "total_prototypes": "",
            "validation_accuracy": baseline["validation_accuracy"],
            "validation_macro_f1": baseline["validation_macro_f1"],
            "validation_weighted_f1": "",
            "prototype_fit_seconds": "",
            "kmeans_warning_count": "",
        })

    with (args.results_dir / "validation_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    best_configurations = {}
    for method in ("euclidean", "cosine"):
        best = best_rows[method].copy()
        best.pop("selection_key")
        best_configurations[method] = best

        y_pred = best_predictions[method]
        report = classification_report(
            y_val,
            y_pred,
            labels=np.arange(N_CLASSES),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
        (args.results_dir / f"{method}_best_classification_report.txt").write_text(
            report, encoding="utf-8"
        )
        save_confusion_matrix(
            args.results_dir / f"{method}_best_confusion_matrix.csv", y_val, y_pred
        )
        save_predictions(
            args.results_dir / f"{method}_best_validation_predictions.csv",
            y_val,
            {method: y_pred},
        )

        if baseline is not None:
            best_configurations[method]["accuracy_difference_from_baseline_pp"] = round(
                100.0 * (
                    best_configurations[method]["validation_accuracy"]
                    - baseline["validation_accuracy"]
                ),
                4,
            )

    np.savez_compressed(
        args.outputs_dir / "classwise_prototypes_validation.npz",
        **saved_prototypes,
    )

    # Load the test split only after the validation configurations have been selected.
    X_test, y_test = load_test_features(args.data)

    # Refit the selected K for each prediction method on train + validation.
    X_development = np.concatenate([X_train, X_val], axis=0)
    y_development = np.concatenate([y_train, y_val], axis=0)
    final_test_predictions = {}
    final_test_rows = []
    final_prototypes = {}
    fitted_for_k = {}

    for method in ("euclidean", "cosine"):
        selected_k = int(best_configurations[method]["prototypes_per_class"])

        if selected_k not in fitted_for_k:
            prototypes, prototype_labels, fit_seconds, warning_count = (
                fit_classwise_prototypes(
                    X_development,
                    y_development,
                    selected_k,
                    args.seed,
                    args.n_init,
                    args.max_iter,
                )
            )
            fitted_for_k[selected_k] = (
                prototypes, prototype_labels, fit_seconds, warning_count
            )

        prototypes, prototype_labels, fit_seconds, warning_count = (
            fitted_for_k[selected_k]
        )
        final_prototypes[f"{method}_prototypes"] = prototypes
        final_prototypes[f"{method}_prototype_labels"] = prototype_labels

        if method == "euclidean":
            y_pred = predict_euclidean(
                X_test, prototypes, prototype_labels, args.batch_size
            )
        else:
            y_pred = predict_cosine(
                X_test, prototypes, prototype_labels, args.batch_size
            )

        final_test_predictions[method] = y_pred
        metrics = get_metrics(y_test, y_pred, "test")
        row = {
            "method": method,
            "selected_K": selected_k,
            "total_prototypes": len(prototypes),
            **metrics,
            "final_prototype_fit_seconds": round(fit_seconds, 4),
            "kmeans_warning_count": warning_count,
        }
        final_test_rows.append(row)

        report = classification_report(
            y_test,
            y_pred,
            labels=np.arange(N_CLASSES),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
        (args.results_dir / f"{method}_final_test_report.txt").write_text(
            report, encoding="utf-8"
        )
        save_confusion_matrix(
            args.results_dir / f"{method}_final_test_confusion_matrix.csv",
            y_test,
            y_pred,
        )

    with (args.results_dir / "final_test_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(final_test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_test_rows)

    save_predictions(
        args.outputs_dir / "final_test_predictions.csv",
        y_test,
        final_test_predictions,
    )
    np.savez_compressed(
        args.outputs_dir / "classwise_prototypes_final.npz",
        **final_prototypes,
    )

    final_test_summary = {
        "dataset": "CIFAR-10",
        "feature_cache": str(args.data),
        "development_images": len(y_development),
        "official_test_images": len(y_test),
        "model_selection": "Best K selected using validation accuracy, with macro-F1 as tie-breaker",
        "final_fit": "Selected prototypes refitted on the combined 45,000 train and 5,000 validation images",
        "official_test_evaluated_once": True,
        "results": final_test_rows,
    }
    with (args.results_dir / "final_test_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(final_test_summary, file, indent=2)

    summary = {
        "experiment": "Class-wise prototypes on frozen DINOv2 ViT-S/14 features",
        "dataset": "CIFAR-10",
        "feature_cache": str(args.data),
        "feature_dimension": FEATURE_DIM,
        "train_images": len(y_train),
        "validation_images": len(y_val),
        "official_test_images": len(y_test),
        "official_test_used": True,
        "normalization": "L2 per image feature vector",
        "prototype_learning": (
            "K=1 uses class mean; K>1 uses Euclidean K-Means independently per class"
        ),
        "prediction_variants": {
            "euclidean": "Nearest unnormalized K-Means centroid by Euclidean distance",
            "cosine": "Nearest L2-normalized centroid by cosine similarity",
        },
        "k_values": args.ks,
        "seed": args.seed,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "baseline_validation_reference": baseline,
        "best_validation_configurations": best_configurations,
        "validation_results": result_rows,
        "final_test_results": final_test_rows,
    }
    with (args.results_dir / "experiment_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("Validation results")
    print("method      K   accuracy   macro-F1")
    for row in result_rows:
        if row["method"] == "DINOv2 + Logistic Regression":
            print(
                f"{row['method']:<28} "
                f"{row['validation_accuracy']:.4f}     "
                f"{row['validation_macro_f1']:.4f}"
            )
        else:
            print(
                f"{row['method']:<10} {row['prototypes_per_class']:<3} "
                f"{row['validation_accuracy']:.4f}     "
                f"{row['validation_macro_f1']:.4f}"
            )

    print("\nFinal test results")
    print("method      selected K   accuracy   macro-F1")
    for row in final_test_rows:
        print(
            f"{row['method']:<10} {row['selected_K']:<11} "
            f"{row['test_accuracy']:.4f}     {row['test_macro_f1']:.4f}"
        )

    print(f"\nResults: {args.results_dir}")
    print(f"Outputs: {args.outputs_dir}")


if __name__ == "__main__":
    main()
