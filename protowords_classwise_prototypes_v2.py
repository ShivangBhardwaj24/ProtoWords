#!/usr/bin/env python3
""" 
ProtoWords — Experiment 2: Class-wise prototype baseline

Purpose
-------
Compare nearest-prototype classifiers with K prototypes per CIFAR-10 class:
K = 1, 2, 4, 8 (by default).

Input
-----
The cached DINOv2 ViT-S/14 feature file created by the full baseline script.
Only X_train, y_train, X_val, and y_val are read. The official test set is
intentionally not loaded or used during this model-selection experiment.

Method
------
1. L2-normalize each image feature vector.
2. For each class independently:
   - K=1: use the class mean feature vector as its prototype.
   - K>1: fit scikit-learn K-Means to that class's training features.
3. Classify each validation image using its nearest prototype by Euclidean
   distance. This matches the Euclidean K-Means objective. Centroids are NOT
   normalized after K-Means, because doing so and then using cosine similarity
   would change the assignment metric used to learn the centroids.
4. Compare validation metrics and save prototypes for later refitting.

This script selects a candidate K using validation data only. It does not
report final test performance. After selecting K, refit prototypes on the full
50,000-image training set (train + validation) and evaluate once on the
10,000-image official test set in a separate final-evaluation step.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
EXPECTED_CLASSES = np.arange(len(CLASS_NAMES), dtype=np.int64)
DEFAULT_DATA = (
    "outputs/cifar10_dinov2_vits14_224px_"
    "train45000_val5000_test10000_seed42.npz"
)
DEFAULT_BASELINE_METRICS = "outputs/baseline_dinov2_cifar10_metrics.json"


def l2_normalize(features: np.ndarray) -> np.ndarray:
    """L2-normalize rows, safely handling zero-length vectors."""
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def load_train_validation_cache(
    data_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load only train/validation arrays; do not access official test arrays."""
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Feature cache not found: {data_path}\n"
            f"Run from your ~/ProtoWords directory or pass --data with the correct path."
        )

    required = {"X_train", "y_train", "X_val", "y_val"}
    with np.load(data_path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(
                f"Feature cache is missing keys: {sorted(missing)}. "
                f"Available keys: {archive.files}"
            )
        # Explicitly copy the arrays out of the NPZ context.
        X_train = np.asarray(archive["X_train"], dtype=np.float32).copy()
        y_train = np.asarray(archive["y_train"], dtype=np.int64).copy()
        X_val = np.asarray(archive["X_val"], dtype=np.float32).copy()
        y_val = np.asarray(archive["y_val"], dtype=np.int64).copy()

    for name, X, y in (
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
    ):
        if X.ndim != 2:
            raise ValueError(f"{name} features must be 2-D; got shape {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"{name} labels must be 1-D; got shape {y.shape}.")
        if len(X) != len(y):
            raise ValueError(
                f"{name} feature/label count mismatch: {len(X)} vs {len(y)}."
            )
        if not np.isfinite(X).all():
            raise ValueError(f"{name} features contain NaN or infinite values.")
        if not np.isin(y, EXPECTED_CLASSES).all():
            raise ValueError(f"{name} labels must be integers from 0 to 9.")

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            f"Train and validation feature dimensions differ: "
            f"{X_train.shape[1]} vs {X_val.shape[1]}."
        )

    missing_train = sorted(set(EXPECTED_CLASSES) - set(np.unique(y_train)))
    missing_val = sorted(set(EXPECTED_CLASSES) - set(np.unique(y_val)))
    if missing_train or missing_val:
        raise ValueError(
            f"All 10 classes must be present. Missing in train: {missing_train}; "
            f"missing in validation: {missing_val}."
        )

    # Normalize both splits using the same row-wise operation.
    X_train = l2_normalize(X_train)
    X_val = l2_normalize(X_val)
    return X_train, y_train, X_val, y_val


def learn_classwise_prototypes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Learn K Euclidean prototypes separately for each class.

    Input features are L2-normalized. K=1 is the class mean. For K>1,
    standard Euclidean K-Means is fitted independently within each class.
    Centroids are kept unnormalized so prediction uses the same Euclidean
    geometry that K-Means optimized.
    """
    if k < 1:
        raise ValueError(f"K must be >= 1; received {k}.")

    prototype_vectors = []
    prototype_labels = []

    for class_id in EXPECTED_CLASSES:
        class_features = X_train[y_train == class_id]
        if len(class_features) < k:
            raise ValueError(
                f"Class {CLASS_NAMES[class_id]} has only {len(class_features)} "
                f"training examples, fewer than K={k}."
            )

        if k == 1:
            centers = class_features.mean(axis=0, keepdims=True)
        else:
            model = KMeans(
                n_clusters=k,
                random_state=random_state + int(class_id),
                n_init=n_init,
                max_iter=max_iter,
                algorithm="lloyd",
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(class_features)
            for warning in caught:
                print(
                    f"Warning for class {CLASS_NAMES[class_id]} (K={k}): "
                    f"{warning.message}"
                )
            centers = model.cluster_centers_

        prototype_vectors.append(np.asarray(centers, dtype=np.float32))
        prototype_labels.extend([int(class_id)] * len(centers))

    prototypes = np.vstack(prototype_vectors).astype(np.float32, copy=False)
    labels = np.asarray(prototype_labels, dtype=np.int64)
    return prototypes, labels


def predict_nearest_prototype(
    X: np.ndarray,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    """Predict by nearest prototype under squared Euclidean distance.

    The distance calculation is vectorized and batched to keep memory use low.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if prototypes.ndim != 2 or X.ndim != 2:
        raise ValueError("X and prototypes must both be 2-D arrays.")
    if X.shape[1] != prototypes.shape[1]:
        raise ValueError("X and prototype feature dimensions do not match.")
    if len(prototypes) != len(prototype_labels):
        raise ValueError("Each prototype must have exactly one class label.")

    prototype_norms = np.sum(prototypes * prototypes, axis=1)[None, :]
    predictions = []

    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        batch_norms = np.sum(batch * batch, axis=1, keepdims=True)
        # ||x-p||^2 = ||x||^2 + ||p||^2 - 2*x.p
        distances = batch_norms + prototype_norms - 2.0 * (batch @ prototypes.T)
        np.maximum(distances, 0.0, out=distances)  # numerical safety
        nearest_indices = np.argmin(distances, axis=1)
        predictions.append(prototype_labels[nearest_indices])

    return np.concatenate(predictions).astype(np.int64, copy=False)


def load_baseline_validation_accuracy(metrics_path: Path):
    """Read the already-recorded logistic-regression validation score, if found."""
    if not metrics_path.is_file():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        selected_c = metrics.get("selected_C")
        search_rows = metrics.get("validation_search", [])
        for row in search_rows:
            if selected_c is not None and float(row.get("C")) == float(selected_c):
                return float(row["validation_accuracy"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"Could not read baseline validation metric: {exc}")
    return None


def save_confusion_outputs(
    output_dir: Path,
    k: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Save count matrix, row-normalized matrix, and a readable heatmap."""
    matrix = confusion_matrix(y_true, y_pred, labels=EXPECTED_CLASSES)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_totals, 1)

    np.savetxt(
        output_dir / f"confusion_matrix_k{k}_counts.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )
    np.savetxt(
        output_dir / f"confusion_matrix_k{k}_row_normalized.csv",
        normalized,
        delimiter=",",
        fmt="%.6f",
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(normalized, interpolation="nearest", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=EXPECTED_CLASSES,
        yticks=EXPECTED_CLASSES,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title=f"Class-wise prototypes: validation confusion matrix (K={k})",
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

    # Annotate each cell; values are row-normalized proportions.
    for row in range(len(CLASS_NAMES)):
        for col in range(len(CLASS_NAMES)):
            value = normalized[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value >= 0.55 else "black",
            )

    fig.tight_layout()
    fig.savefig(
        output_dir / f"confusion_matrix_k{k}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate class-wise DINOv2 prototypes on CIFAR-10."
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to the DINOv2 NPZ feature cache.")
    parser.add_argument(
        "--output-dir",
        default="outputs/classwise_prototype_validation",
        help="Directory for validation results and learned prototypes.",
    )
    parser.add_argument(
        "--ks", nargs="+", type=int, default=[1, 2, 4, 8],
        help="Number of prototypes per class to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="K-Means random seed.")
    parser.add_argument("--n-init", type=int, default=10, help="K-Means initializations for each class.")
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum K-Means iterations.")
    parser.add_argument("--prediction-batch-size", type=int, default=2048, help="Validation prediction batch size.")
    parser.add_argument(
        "--baseline-metrics",
        default=DEFAULT_BASELINE_METRICS,
        help="Optional baseline JSON, used only to show the logistic-regression validation reference.",
    )
    args = parser.parse_args()

    ks = sorted(set(args.ks))
    if not ks or any(k < 1 for k in ks):
        raise ValueError("Provide one or more positive K values, e.g. --ks 1 2 4 8.")
    if args.n_init < 1 or args.max_iter < 1:
        raise ValueError("--n-init and --max-iter must be positive.")

    data_path = Path(args.data).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ProtoWords — Experiment 2: Class-wise prototype validation")
    print("=" * 72)
    print(f"Feature cache: {data_path.resolve()}")
    print("The official test arrays will NOT be loaded or evaluated in this script.")

    start_total = time.time()
    X_train, y_train, X_val, y_val = load_train_validation_cache(data_path)

    class_counts = {CLASS_NAMES[c]: int(np.sum(y_train == c)) for c in EXPECTED_CLASSES}
    min_class_count = min(class_counts.values())
    if max(ks) > min_class_count:
        raise ValueError(
            f"Largest requested K={max(ks)} exceeds the smallest class training "
            f"count ({min_class_count})."
        )

    print(f"\nTraining features:   {X_train.shape}")
    print(f"Validation features: {X_val.shape}")
    print(f"Feature dimension:   {X_train.shape[1]}")
    print(f"Training examples per class: {class_counts}")
    print(f"K values: {ks}")
    print(f"K-Means n_init: {args.n_init}; max_iter: {args.max_iter}")

    baseline_val_accuracy = load_baseline_validation_accuracy(Path(args.baseline_metrics))
    if baseline_val_accuracy is not None:
        print(f"Logistic-regression validation reference: {baseline_val_accuracy:.4%}")
    else:
        print("Logistic-regression validation reference not found; continuing without it.")

    rows = []
    all_saved = {}

    for k in ks:
        print("\n" + "-" * 72)
        print(f"Fitting K={k} prototype(s) per class ({10 * k} total prototypes)")
        fit_start = time.time()
        prototypes, prototype_labels = learn_classwise_prototypes(
            X_train=X_train,
            y_train=y_train,
            k=k,
            random_state=args.seed,
            n_init=args.n_init,
            max_iter=args.max_iter,
        )
        fit_seconds = time.time() - fit_start

        pred_start = time.time()
        val_predictions = predict_nearest_prototype(
            X=X_val,
            prototypes=prototypes,
            prototype_labels=prototype_labels,
            batch_size=args.prediction_batch_size,
        )
        prediction_seconds = time.time() - pred_start

        accuracy = float(accuracy_score(y_val, val_predictions))
        macro_f1 = float(f1_score(y_val, val_predictions, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_val, val_predictions, average="weighted", zero_division=0))

        rows.append({
            "method": "classwise_nearest_prototype_euclidean",
            "prototypes_per_class": int(k),
            "total_prototypes": int(len(prototypes)),
            "validation_accuracy": accuracy,
            "validation_macro_f1": macro_f1,
            "validation_weighted_f1": weighted_f1,
            "prototype_fit_seconds": fit_seconds,
            "validation_prediction_seconds": prediction_seconds,
        })

        report_text = classification_report(
            y_val,
            val_predictions,
            labels=EXPECTED_CLASSES,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
        (output_dir / f"classification_report_k{k}.txt").write_text(
            report_text, encoding="utf-8"
        )
        save_confusion_outputs(output_dir, k, y_val, val_predictions)

        pd.DataFrame({
            "true_label": y_val,
            "true_class": [CLASS_NAMES[i] for i in y_val],
            "predicted_label": val_predictions,
            "predicted_class": [CLASS_NAMES[i] for i in val_predictions],
        }).to_csv(output_dir / f"validation_predictions_k{k}.csv", index=False)

        all_saved[f"prototypes_k{k}"] = prototypes
        all_saved[f"prototype_labels_k{k}"] = prototype_labels

        print(f"Total prototypes:       {len(prototypes)}")
        print(f"Prototype fitting time: {fit_seconds:.2f} seconds")
        print(f"Prediction time:        {prediction_seconds:.2f} seconds")
        print(f"Validation accuracy:    {accuracy:.4%}")
        print(f"Validation macro-F1:    {macro_f1:.4f}")
        print(f"Validation weighted-F1: {weighted_f1:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "validation_results.csv", index=False)
    np.savez_compressed(output_dir / "learned_prototypes_validation.npz", **all_saved)

    # Select a candidate using validation macro-F1, breaking ties with accuracy.
    best_row = results.sort_values(
        ["validation_macro_f1", "validation_accuracy"],
        ascending=[False, False],
    ).iloc[0]
    best_k = int(best_row["prototypes_per_class"])

    # Make a visual comparison; the LR baseline is a reference, not a fitted method here.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        results["prototypes_per_class"],
        results["validation_accuracy"],
        marker="o",
        linewidth=2,
        label="Class-wise prototype accuracy",
    )
    if baseline_val_accuracy is not None:
        ax.axhline(
            baseline_val_accuracy,
            linestyle="--",
            linewidth=1.8,
            label=f"DINOv2 + logistic regression ({baseline_val_accuracy:.2%})",
        )
    ax.set_xlabel("Prototypes per class (K)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Validation comparison: class-wise prototypes")
    ax.set_xticks(ks)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_accuracy_comparison.png", dpi=180)
    plt.close(fig)

    summary: Dict[str, object] = {
        "experiment": "Class-wise K-Means prototypes on frozen DINOv2 ViT-S/14 features",
        "dataset": "CIFAR-10",
        "feature_cache": str(data_path),
        "feature_dimension": int(X_train.shape[1]),
        "training_images": int(len(y_train)),
        "validation_images": int(len(y_val)),
        "official_test_used": False,
        "normalization": "L2 per image feature vector",
        "clustering": "class-wise Euclidean K-Means on normalized training features",
        "prediction": "nearest prototype by Euclidean distance",
        "k_values": ks,
        "kmeans_n_init": int(args.n_init),
        "kmeans_max_iter": int(args.max_iter),
        "random_seed": int(args.seed),
        "logistic_regression_validation_reference_accuracy": baseline_val_accuracy,
        "best_k_by_validation_macro_f1_then_accuracy": best_k,
        "best_validation_accuracy": float(best_row["validation_accuracy"]),
        "best_validation_macro_f1": float(best_row["validation_macro_f1"]),
        "results": rows,
        "total_elapsed_seconds": float(time.time() - start_total),
        "next_step": (
            "After selecting K using validation only, refit the chosen class-wise "
            "prototype model on all 50,000 train+validation examples and evaluate "
            "once on the untouched 10,000-image official test set."
        ),
    }
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 72)
    print("VALIDATION RESULTS")
    print("=" * 72)
    print(results.to_string(index=False, formatters={
        "validation_accuracy": "{:.4%}".format,
        "validation_macro_f1": "{:.4f}".format,
        "validation_weighted_f1": "{:.4f}".format,
    }))
    if baseline_val_accuracy is not None:
        print(f"\nLogistic-regression validation reference: {baseline_val_accuracy:.4%}")
    print(f"Best candidate by validation macro-F1, then accuracy: K={best_k}")
    print(f"\nSaved outputs in: {output_dir.resolve()}")
    print("Official test set was not used. Do not choose K using test results.")


if __name__ == "__main__":
    main()
