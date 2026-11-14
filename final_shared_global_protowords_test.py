#!/usr/bin/env python3
"""Final held-out CIFAR-10 evaluation for shared global ProtoWords.

Protocol
--------
1. Read the existing frozen DINOv2 ViT-S/14 feature cache.
2. L2-normalize each image feature (same transform as validation experiments).
3. Concatenate the 45,000 development-train and 5,000 validation examples.
4. Fit a fresh global K-Means vocabulary with K=128 on those 50,000 features.
5. Compute Euclidean distances from each image to all 128 centroids.
6. Fit StandardScaler on the 50,000 development distance vectors only.
7. Fit LogisticRegression(C=1.0) on the 50,000 labeled development examples.
8. Evaluate once on the official 10,000-image CIFAR-10 test split.

K=128 and C=1.0 were selected earlier using validation results. This script
never uses official test labels for model selection or fitting.

Run from the ProtoWords project root:
    python3 final_shared_global_protowords_test.py

Optional paths:
    python3 final_shared_global_protowords_test.py \
        --data outputs/cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz \
        --output-dir outputs/final_shared_global_protowords_k128_distance
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import Normalizer, StandardScaler
from threadpoolctl import threadpool_limits

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
DEFAULT_CACHE = (
    "outputs/cifar10_dinov2_vits14_224px_"
    "train45000_val5000_test10000_seed42.npz"
)
DEFAULT_BASELINE = "outputs/baseline_dinov2_cifar10_metrics.json"
DEFAULT_OUTPUT_DIR = "outputs/final_shared_global_protowords_k128_distance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final official-test evaluation for selected ProtoWords K=128 distance model."
    )
    parser.add_argument("--data", default=DEFAULT_CACHE, help="Feature-cache NPZ path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--baseline-metrics", default=DEFAULT_BASELINE,
                        help="Baseline metrics JSON used only for comparison.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter-kmeans", type=int, default=300)
    parser.add_argument("--max-iter-logreg", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def check_split(name: str, X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError(f"{name} features must be a 2D matrix; got {X.shape}.")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"{name} features/labels have incompatible shapes: {X.shape}, {y.shape}.")
    if X.shape[1] != 384:
        raise ValueError(f"Expected 384-dimensional DINOv2 features; got {X.shape[1]}.")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} features contain NaN or infinite values.")
    if set(np.unique(y).tolist()) != set(range(10)):
        raise ValueError(f"{name} labels must contain CIFAR-10 class IDs 0 through 9.")


def load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature cache not found: {path}\n"
            "Run from the ProtoWords project root or pass its path using --data."
        )
    with np.load(path, allow_pickle=False) as data:
        required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"Feature cache is missing required arrays: {missing}")
        result = {key: np.asarray(data[key]) for key in required}
        metadata = data["metadata"].item() if "metadata" in data.files else ""
        train_indices = np.asarray(data["train_indices"]) if "train_indices" in data.files else None
        val_indices = np.asarray(data["val_indices"]) if "val_indices" in data.files else None
    for split in ("train", "val", "test"):
        result[f"X_{split}"] = result[f"X_{split}"].astype(np.float32, copy=False)
        result[f"y_{split}"] = result[f"y_{split}"].astype(np.int64, copy=False)
        check_split(split, result[f"X_{split}"], result[f"y_{split}"])
    if len(result["X_train"]) != 45000 or len(result["X_val"]) != 5000 or len(result["X_test"]) != 10000:
        raise ValueError("Split sizes differ from the intended 45k/5k/10k CIFAR-10 protocol.")
    if train_indices is not None and val_indices is not None:
        if np.intersect1d(train_indices, val_indices).size:
            raise ValueError("Training and validation indices overlap.")
    result["metadata"] = metadata
    return result


def load_baseline(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        test = payload.get("test_metrics", {})
        if "accuracy" in test:
            return {
                "accuracy": float(test["accuracy"]),
                "macro_f1": float(test.get("macro_f1", float("nan"))),
                "micro_f1": float(test.get("micro_f1", float("nan"))),
            }
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def main() -> None:
    args = parse_args()
    if args.threads < 1 or args.n_init < 1 or args.max_iter_kmeans < 1 or args.max_iter_logreg < 1:
        raise ValueError("Thread and iteration values must be positive.")

    data_path = Path(args.data).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.perf_counter()
    print("=" * 72)
    print("FINAL SHARED-GLOBAL PROTOWORDS EVALUATION")
    print("Selected validation configuration: K=128, distance representation, C=1.0")
    print("Official test: held out until the final evaluation")
    print("=" * 72)

    cache = load_cache(data_path)
    X_train_raw, y_train = cache["X_train"], cache["y_train"]
    X_val_raw, y_val = cache["X_val"], cache["y_val"]
    X_test_raw, y_test = cache["X_test"], cache["y_test"]
    print(f"Loaded splits: train={X_train_raw.shape}, validation={X_val_raw.shape}, test={X_test_raw.shape}")

    # Keep the same per-image L2 normalization used in the earlier ProtoWords runs.
    normalizer = Normalizer(norm="l2")
    X_train = normalizer.fit_transform(X_train_raw).astype(np.float32, copy=False)
    X_val = normalizer.transform(X_val_raw).astype(np.float32, copy=False)
    X_test = normalizer.transform(X_test_raw).astype(np.float32, copy=False)
    del X_train_raw, X_val_raw, X_test_raw

    # Match the baseline's final-refit protocol: combine train + validation.
    X_dev = np.concatenate([X_train, X_val], axis=0)
    y_dev = np.concatenate([y_train, y_val], axis=0)
    del X_train, X_val, y_train, y_val
    print(f"Final development pool: {X_dev.shape}; class counts={np.bincount(y_dev, minlength=10).tolist()}")

    # Fit a new codebook on all 50,000 development images, not the earlier 45k codebook.
    print("Fitting fresh K-Means codebook (K=128) on the 50,000 development features...")
    kmeans = KMeans(
        n_clusters=128,
        random_state=args.seed,
        n_init=args.n_init,
        max_iter=args.max_iter_kmeans,
        algorithm="lloyd",
    )
    t0 = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        kmeans.fit(X_dev)
    kmeans_seconds = time.perf_counter() - t0
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    cluster_sizes = np.bincount(kmeans.labels_, minlength=128).astype(np.int64)
    print(f"K-Means finished in {kmeans_seconds:.2f}s; inertia={kmeans.inertia_:.4f}")
    print(f"Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, empty={(cluster_sizes == 0).sum()}")

    # Distance representation: one Euclidean distance per codeword.
    print("Building 128-dimensional Euclidean-distance representations...")
    t0 = time.perf_counter()
    D_dev = euclidean_distances(X_dev, centers, squared=False).astype(np.float32, copy=False)
    D_test = euclidean_distances(X_test, centers, squared=False).astype(np.float32, copy=False)
    scaler = StandardScaler()
    D_dev_scaled = scaler.fit_transform(D_dev).astype(np.float32, copy=False)
    D_test_scaled = scaler.transform(D_test).astype(np.float32, copy=False)
    distance_scaling_seconds = time.perf_counter() - t0
    del D_dev, D_test, X_dev, X_test

    # C=1.0 was selected on validation before this test is run; do not retune on test.
    print("Fitting Logistic Regression (C=1.0) on all 50,000 development examples...")
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=args.max_iter_logreg,
        random_state=args.seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        classifier.fit(D_dev_scaled, y_dev)
        classifier_seconds = time.perf_counter() - t0
    convergence_warnings = [str(w.message) for w in caught if "converg" in str(w.message).lower()]
    del D_dev_scaled, y_dev

    print("Evaluating once on the official 10,000-image test split...")
    t0 = time.perf_counter()
    y_pred = classifier.predict(D_test_scaled)
    y_proba = classifier.predict_proba(D_test_scaled)
    prediction_seconds = time.perf_counter() - t0
    del D_test_scaled

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_test, y_pred, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    report_dict = classification_report(
        y_test, y_pred, labels=list(range(10)), target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        y_test, y_pred, labels=list(range(10)), target_names=CLASS_NAMES,
        digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
    print("\nFINAL OFFICIAL-TEST RESULTS")
    print(f"Accuracy : {accuracy:.4%}")
    print(f"Macro-F1 : {macro_f1:.4%}")
    print(f"Micro-F1 : {micro_f1:.4%}")
    print(f"Weighted-F1: {weighted_f1:.4%}")
    print("\nPer-class report:\n" + report_text)
    if convergence_warnings:
        print("WARNING: Logistic Regression emitted convergence warning(s):")
        for item in convergence_warnings:
            print(" -", item)
    else:
        print("Logistic Regression convergence: no convergence warning")

    baseline = load_baseline(Path(args.baseline_metrics).expanduser())
    comparison = None
    if baseline is not None:
        comparison = {
            "baseline_test_accuracy": baseline["accuracy"],
            "protowords_test_accuracy": accuracy,
            "accuracy_difference_percentage_points": (accuracy - baseline["accuracy"]) * 100.0,
            "baseline_test_macro_f1": baseline["macro_f1"],
            "protowords_test_macro_f1": macro_f1,
            "macro_f1_difference_percentage_points": (macro_f1 - baseline["macro_f1"]) * 100.0,
        }
        print("\nComparison with DINOv2 + Logistic Regression baseline:")
        print(f"Accuracy: baseline={baseline['accuracy']:.4%}, ProtoWords={accuracy:.4%}, "
              f"difference={(accuracy - baseline['accuracy']) * 100:+.3f} percentage points")
        print(f"Macro-F1: baseline={baseline['macro_f1']:.4%}, ProtoWords={macro_f1:.4%}, "
              f"difference={(macro_f1 - baseline['macro_f1']) * 100:+.3f} percentage points")

    # Persist all evaluation outputs and the final reusable model components.
    (output_dir / "classification_report.txt").write_text(
        "ProtoWords: Shared Global Vocabulary (K=128, Euclidean distances)\n"
        "===============================================================\n\n"
        "Protocol: select K=128 and C=1.0 using validation; refit K-Means, scaler, "
        "and classifier on train+validation (50,000 images); evaluate once on the "
        "official CIFAR-10 test set (10,000 images).\n\n"
        f"Test accuracy: {accuracy:.6f}\nTest macro-F1: {macro_f1:.6f}\n"
        f"Test micro-F1: {micro_f1:.6f}\nTest weighted-F1: {weighted_f1:.6f}\n\n"
        + report_text,
        encoding="utf-8",
    )
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix_counts.csv", index_label="true_label"
    )
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title("ProtoWords K=128 — CIFAR-10 Test Confusion Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(ticks, CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(ticks, CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    predictions = pd.DataFrame({
        "test_index": np.arange(len(y_test)),
        "true_label_id": y_test,
        "true_label": [CLASS_NAMES[i] for i in y_test],
        "predicted_label_id": y_pred,
        "predicted_label": [CLASS_NAMES[i] for i in y_pred],
        "correct": (y_test == y_pred),
        "max_class_probability": y_proba.max(axis=1),
    })
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    np.savez_compressed(
        output_dir / "final_codebook_k128.npz",
        centers=centers,
        cluster_sizes=cluster_sizes,
        n_clusters=np.asarray(128),
        random_state=np.asarray(args.seed),
        feature_dimension=np.asarray(384),
    )
    model_bundle = {
        "method": "shared_global_protowords_euclidean_distance_logistic_regression",
        "K": 128,
        "representation": "Euclidean distances to all K-Means centroids, standardized",
        "feature_extractor": "frozen DINOv2 ViT-S/14 global CLS features",
        "feature_dimension": 384,
        "feature_normalizer": normalizer,
        "kmeans": kmeans,
        "distance_scaler": scaler,
        "classifier": classifier,
        "selected_C_from_validation": 1.0,
        "class_names": CLASS_NAMES,
        "random_seed": args.seed,
    }
    joblib.dump(model_bundle, output_dir / "final_model_k128_distance.joblib")

    total_seconds = time.perf_counter() - start_total
    metrics = {
        "experiment": "final_shared_global_protowords_test_evaluation",
        "dataset": "CIFAR-10",
        "feature_extractor": "frozen DINOv2 ViT-S/14 global CLS features",
        "feature_dimension": 384,
        "feature_normalization": "L2 per image using sklearn Normalizer",
        "protocol": {
            "development_train": 45000,
            "validation_for_model_selection": 5000,
            "final_refit_train_plus_validation": 50000,
            "official_test": 10000,
            "split_seed": 42,
            "selected_K": 128,
            "representation": "Euclidean distances to all K-Means centroids, standardized",
            "selected_C_from_validation": 1.0,
            "kmeans_random_state": args.seed,
            "kmeans_n_init": args.n_init,
            "kmeans_max_iter": args.max_iter_kmeans,
            "logistic_regression_solver": "lbfgs",
            "logistic_regression_max_iter": args.max_iter_logreg,
            "official_test_used_for_selection_or_fitting": False,
        },
        "test_metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "weighted_f1": weighted_f1,
        },
        "classification_report": report_dict,
        "baseline_comparison": comparison,
        "cluster_statistics": {
            "inertia": float(kmeans.inertia_),
            "minimum_cluster_size": int(cluster_sizes.min()),
            "maximum_cluster_size": int(cluster_sizes.max()),
            "empty_clusters": int((cluster_sizes == 0).sum()),
        },
        "timing_seconds": {
            "kmeans_fit": kmeans_seconds,
            "distance_transform_and_scaling": distance_scaling_seconds,
            "classifier_fit": classifier_seconds,
            "test_prediction": prediction_seconds,
            "total_script": total_seconds,
        },
        "convergence_warnings": convergence_warnings,
        "feature_cache": str(data_path),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nSaved final evaluation artifacts to: {output_dir.resolve()}")
    print(f"Total script runtime: {total_seconds:.2f}s")


if __name__ == "__main__":
    main()
