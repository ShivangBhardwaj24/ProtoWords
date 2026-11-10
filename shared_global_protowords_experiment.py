#!/usr/bin/env python3
"""
ProtoWords Experiment: Shared Global Prototype Vocabulary
=========================================================

Runs six validation experiments using an existing frozen DINOv2 ViT-S/14
CIFAR-10 feature cache:

    K in {32, 64, 128}
    Representation in {hard one-hot assignment, distances to all prototypes}

For each K:
  1. Fit ONE global K-Means codebook on normalized TRAIN features only.
  2. Convert train/validation features to hard one-hot assignments.
  3. Train/evaluate a Logistic Regression classifier on that representation.
  4. Convert train/validation features to distances to all codebook centroids.
  5. Standardize distance columns using TRAIN data only, then train/evaluate
     another Logistic Regression classifier.
  6. Select C for each classifier using VALIDATION accuracy (Macro-F1 breaks ties).

The official CIFAR-10 test split is intentionally NOT loaded or evaluated here.
Choose configurations using validation results; evaluate the selected
configuration on the official test set in a separate final-evaluation step.

Expected cache (default):
outputs/cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz

Required cache keys: X_train, y_train, X_val, y_val
Optional cache keys such as X_test/y_test are ignored.

Example:
python3 shared_global_protowords_experiment.py

Optional:
python3 shared_global_protowords_experiment.py --ks 32 64 128 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
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
N_CLASSES = len(CLASS_NAMES)
DEFAULT_CACHE = (
    "outputs/cifar10_dinov2_vits14_224px_"
    "train45000_val5000_test10000_seed42.npz"
)
DEFAULT_OUTPUT_DIR = "outputs/shared_global_vocabulary"
DEFAULT_C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run six shared-global-ProtoWords validation experiments."
    )
    parser.add_argument(
        "--data", type=str, default=DEFAULT_CACHE,
        help=f"Path to the cached DINOv2 feature NPZ (default: {DEFAULT_CACHE})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for experiment outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--ks", nargs="+", type=int, default=[32, 64, 128],
        help="Shared vocabulary sizes. Default: 32 64 128",
    )
    parser.add_argument(
        "--c-values", nargs="+", type=float, default=DEFAULT_C_VALUES,
        help="Logistic Regression C values selected on validation data.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter-kmeans", type=int, default=300)
    parser.add_argument("--max-iter-logreg", type=int, default=2000)
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Maximum native CPU threads used by sklearn operations.",
    )
    parser.add_argument(
        "--baseline-metrics", type=str,
        default="outputs/baseline_dinov2_cifar10_metrics.json",
        help="Optional baseline JSON for drawing reference lines in plots.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.ks or any(k <= 1 for k in args.ks):
        raise ValueError("All K values must be integers greater than 1.")
    if len(set(args.ks)) != len(args.ks):
        raise ValueError("K values must not contain duplicates.")
    if not args.c_values or any(c <= 0 for c in args.c_values):
        raise ValueError("All Logistic Regression C values must be positive.")
    if args.n_init < 1 or args.max_iter_kmeans < 1 or args.max_iter_logreg < 1:
        raise ValueError("n_init and iteration limits must be positive.")
    if args.threads < 1:
        raise ValueError("threads must be at least 1.")


def load_feature_cache(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature cache not found: {path}\n"
            "Run this script from ~/ProtoWords, or pass the correct path with --data."
        )

    print(f"Loading feature cache: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = ["X_train", "y_train", "X_val", "y_val"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"Feature cache is missing required arrays: {missing}")
        X_train = np.asarray(data["X_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=np.int64)
        X_val = np.asarray(data["X_val"], dtype=np.float32)
        y_val = np.asarray(data["y_val"], dtype=np.int64)

    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("X_train and X_val must be two-dimensional feature matrices.")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Train and validation feature dimensions do not match.")
    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise ValueError("Feature and label counts do not match.")
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Training and validation arrays must not be empty.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Feature cache contains NaN or infinite values.")

    expected = set(range(N_CLASSES))
    if set(np.unique(y_train).tolist()) != expected:
        raise ValueError("Training labels must contain all CIFAR-10 labels 0 through 9.")
    if set(np.unique(y_val).tolist()) != expected:
        raise ValueError("Validation labels must contain all CIFAR-10 labels 0 through 9.")

    return X_train, y_train, X_val, y_val


def load_baseline_reference(path: Path) -> Dict[str, float]:
    """Read the completed baseline's validation score if its JSON is available."""
    fallback = {"validation_accuracy": 0.9772, "validation_macro_f1": 0.9772272516012757}
    if not path.is_file():
        print("Baseline metrics JSON not found; using the recorded baseline reference "
              "(97.72% validation accuracy) for plots only.")
        return fallback

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        selected_c = float(payload.get("selected_C", 1.0))
        search = payload.get("validation_search", [])
        match = next((row for row in search if np.isclose(float(row.get("C", -1)), selected_c)), None)
        if match is not None:
            return {
                "validation_accuracy": float(match["validation_accuracy"]),
                "validation_macro_f1": float(match["validation_macro_f1"]),
            }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"Could not read baseline reference ({exc}); using recorded values.")
    return fallback


def one_hot_assignments(cluster_ids: np.ndarray, n_clusters: int) -> csr_matrix:
    rows = np.arange(cluster_ids.shape[0], dtype=np.int32)
    cols = cluster_ids.astype(np.int32, copy=False)
    values = np.ones(cluster_ids.shape[0], dtype=np.float32)
    return csr_matrix((values, (rows, cols)), shape=(len(cluster_ids), n_clusters), dtype=np.float32)


def make_distance_features(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return Euclidean distances from every row in X to every codebook center."""
    distances = euclidean_distances(X, centers, squared=False)
    return np.asarray(distances, dtype=np.float32)


def fit_and_select_classifier(
    X_train_repr,
    y_train: np.ndarray,
    X_val_repr,
    y_val: np.ndarray,
    c_values: List[float],
    max_iter: int,
    seed: int,
    representation: str,
) -> Tuple[LogisticRegression, float, Dict[str, float], List[Dict[str, float]], float]:
    """Select C using validation accuracy; Macro-F1 breaks exact ties."""
    search_rows: List[Dict[str, float]] = []
    best_model = None
    best_c = None
    best_metrics = None
    best_key = (-1.0, -1.0)
    total_fit_start = time.perf_counter()

    for c_value in c_values:
        model = LogisticRegression(
            C=float(c_value),
            solver="lbfgs",
            max_iter=max_iter,
            random_state=seed,
        )
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train_repr, y_train)
        fit_seconds = time.perf_counter() - start
        convergence_warning = any("converg" in str(w.message).lower() for w in caught)

        val_pred = model.predict(X_val_repr)
        accuracy = float(accuracy_score(y_val, val_pred))
        macro_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_val, val_pred, average="weighted", zero_division=0))
        row = {
            "representation": representation,
            "C": float(c_value),
            "validation_accuracy": accuracy,
            "validation_macro_f1": macro_f1,
            "validation_weighted_f1": weighted_f1,
            "fit_seconds": float(fit_seconds),
            "convergence_warning": bool(convergence_warning),
        }
        search_rows.append(row)

        key = (accuracy, macro_f1)
        if key > best_key:
            best_key = key
            best_model = model
            best_c = float(c_value)
            best_metrics = {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
            }

    total_fit_seconds = time.perf_counter() - total_fit_start
    if best_model is None or best_c is None or best_metrics is None:
        raise RuntimeError(f"Failed to fit classifier for {representation} representation.")
    return best_model, best_c, best_metrics, search_rows, total_fit_seconds


def save_classifier_outputs(
    output_dir: Path,
    k: int,
    representation: str,
    model: LogisticRegression,
    y_val: np.ndarray,
    val_predictions: np.ndarray,
    best_c: float,
    metrics: Dict[str, float],
) -> None:
    stem = f"{representation}_k{k}"
    report = classification_report(
        y_val,
        val_predictions,
        labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    (output_dir / f"classification_report_{stem}.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(y_val, val_predictions, labels=list(range(N_CLASSES)))
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / f"confusion_matrix_{stem}.csv", index=True
    )

    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(N_CLASSES),
        yticks=np.arange(N_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title=f"Validation confusion matrix — {representation}, K={k}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({
        "true_label": y_val,
        "true_class": [CLASS_NAMES[int(v)] for v in y_val],
        "predicted_label": val_predictions,
        "predicted_class": [CLASS_NAMES[int(v)] for v in val_predictions],
        "correct": (y_val == val_predictions),
    }).to_csv(output_dir / f"validation_predictions_{stem}.csv", index=False)

    joblib.dump(
        {
            "representation": representation,
            "K": int(k),
            "best_C": float(best_c),
            "classifier": model,
            "class_names": CLASS_NAMES,
            "validation_metrics": metrics,
        },
        output_dir / f"classifier_{stem}.joblib",
    )


def plot_comparisons(
    results: pd.DataFrame,
    output_dir: Path,
    baseline: Dict[str, float],
) -> None:
    order = [(k, rep) for k in sorted(results["K"].unique())
             for rep in ("hard", "distance")]
    labels, accuracies, macro_scores = [], [], []
    for k, rep in order:
        row = results[(results["K"] == k) & (results["representation"] == rep)].iloc[0]
        labels.append(f"K={k}\n{rep}")
        accuracies.append(float(row["validation_accuracy"]) * 100)
        macro_scores.append(float(row["validation_macro_f1"]) * 100)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(x, accuracies, color=["#3977A8" if "hard" in s else "#65A88A" for s in labels])
    ax.axhline(baseline["validation_accuracy"] * 100, color="#C44E52", linestyle="--",
               linewidth=1.8, label=f"Logistic Regression baseline ({baseline['validation_accuracy']*100:.2f}%)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Shared ProtoWords vocabulary — validation accuracy")
    ax.set_xticks(x, labels)
    ax.set_ylim(max(0, min(accuracies + [baseline["validation_accuracy"]*100]) - 4), 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    for bar, value in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_accuracy_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(x, macro_scores, color=["#3977A8" if "hard" in s else "#65A88A" for s in labels])
    ax.axhline(baseline["validation_macro_f1"] * 100, color="#C44E52", linestyle="--",
               linewidth=1.8, label=f"Logistic Regression baseline ({baseline['validation_macro_f1']*100:.2f}%)")
    ax.set_ylabel("Validation Macro F1 (%)")
    ax.set_title("Shared ProtoWords vocabulary — validation Macro F1")
    ax.set_xticks(x, labels)
    ax.set_ylim(max(0, min(macro_scores + [baseline["validation_macro_f1"]*100]) - 4), 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    for bar, value in zip(bars, macro_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_macro_f1_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)

    data_path = Path(args.data).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("ProtoWords — Shared Global Prototype Vocabulary")
    print("Six validation experiments: K=32/64/128 × hard/distance")
    print("=" * 76)
    print(f"Feature cache : {data_path}")
    print(f"Output folder : {output_dir.resolve()}")
    print(f"K values      : {args.ks}")
    print(f"C candidates  : {args.c_values}")
    print(f"Random seed   : {args.seed}")
    print("Official test : NOT loaded or evaluated")

    X_train_raw, y_train, X_val_raw, y_val = load_feature_cache(data_path)
    print(f"Training data : {X_train_raw.shape}")
    print(f"Validation data: {X_val_raw.shape}")
    print(f"Feature dim   : {X_train_raw.shape[1]}")

    if max(args.ks) > len(X_train_raw):
        raise ValueError("Largest K cannot exceed the number of training images.")

    # Match the previous ProtoWords scripts: L2-normalize each image feature.
    normalizer = Normalizer(norm="l2")
    X_train = normalizer.fit_transform(X_train_raw).astype(np.float32, copy=False)
    X_val = normalizer.transform(X_val_raw).astype(np.float32, copy=False)
    del X_train_raw, X_val_raw

    baseline = load_baseline_reference(Path(args.baseline_metrics).expanduser())
    print(f"Baseline reference: accuracy={baseline['validation_accuracy']:.4f}, "
          f"macro-F1={baseline['validation_macro_f1']:.4f}")

    summary_rows: List[Dict[str, float]] = []
    c_search_rows: List[Dict[str, float]] = []
    run_start = time.perf_counter()

    for k in args.ks:
        print("\n" + "=" * 76)
        print(f"FITTING SHARED GLOBAL K-MEANS CODEBOOK: K={k}")
        print("=" * 76)

        kmeans = KMeans(
            n_clusters=int(k),
            random_state=int(args.seed),
            n_init=int(args.n_init),
            max_iter=int(args.max_iter_kmeans),
            algorithm="lloyd",
        )
        km_start = time.perf_counter()
        with threadpool_limits(limits=args.threads):
            kmeans.fit(X_train)
        kmeans_seconds = time.perf_counter() - km_start
        centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)

        # The codebook is learned from training features only.
        train_cluster_ids = kmeans.labels_.astype(np.int32, copy=False)
        val_cluster_ids = kmeans.predict(X_val).astype(np.int32, copy=False)
        cluster_sizes = np.bincount(train_cluster_ids, minlength=k).astype(np.int64)
        empty_clusters = int(np.sum(cluster_sizes == 0))

        codebook_path = output_dir / f"global_codebook_k{k}.npz"
        np.savez_compressed(
            codebook_path,
            centers=centers,
            cluster_sizes=cluster_sizes,
            n_clusters=np.asarray(k),
            random_state=np.asarray(args.seed),
            feature_dimension=np.asarray(X_train.shape[1]),
        )
        joblib.dump(kmeans, output_dir / f"kmeans_model_k{k}.joblib")

        print(f"Codebook fit time : {kmeans_seconds:.2f} seconds")
        print(f"Codebook shape    : {centers.shape}")
        print(f"Smallest cluster  : {cluster_sizes.min()} samples")
        print(f"Largest cluster   : {cluster_sizes.max()} samples")
        if empty_clusters:
            print(f"WARNING: {empty_clusters} clusters have no assigned training samples.")

        # ------------------------------------------------------------------
        # Representation 1: hard nearest-centroid assignment (one-hot).
        # ------------------------------------------------------------------
        print(f"\n[K={k}] Training HARD one-hot assignment classifier...")
        hard_train = one_hot_assignments(train_cluster_ids, k)
        hard_val = one_hot_assignments(val_cluster_ids, k)
        hard_model, hard_c, hard_metrics, hard_search, hard_fit_seconds = fit_and_select_classifier(
            hard_train, y_train, hard_val, y_val,
            args.c_values, args.max_iter_logreg, args.seed, "hard",
        )
        c_search_rows.extend([{**row, "K": int(k)} for row in hard_search])
        hard_pred = hard_model.predict(hard_val)
        save_classifier_outputs(
            output_dir, k, "hard", hard_model, y_val, hard_pred, hard_c, hard_metrics
        )
        hard_row = {
            "K": int(k),
            "representation": "hard",
            "total_prototypes": int(k),
            "selected_C": hard_c,
            "validation_accuracy": hard_metrics["accuracy"],
            "validation_macro_f1": hard_metrics["macro_f1"],
            "validation_weighted_f1": hard_metrics["weighted_f1"],
            "kmeans_fit_seconds": float(kmeans_seconds),
            "classifier_search_seconds": float(hard_fit_seconds),
            "inertia": float(kmeans.inertia_),
            "smallest_cluster_size": int(cluster_sizes.min()),
            "largest_cluster_size": int(cluster_sizes.max()),
            "empty_clusters": empty_clusters,
        }
        summary_rows.append(hard_row)
        joblib.dump(
            {
                "method": "global_codebook_hard_one_hot_logistic_regression",
                "K": int(k),
                "kmeans": kmeans,
                "classifier": hard_model,
                "best_C": hard_c,
                "class_names": CLASS_NAMES,
                "validation_metrics": hard_metrics,
            },
            output_dir / f"complete_model_hard_k{k}.joblib",
        )
        print(f"Hard assignment: best C={hard_c:g}, "
              f"accuracy={hard_metrics['accuracy']:.4%}, "
              f"Macro-F1={hard_metrics['macro_f1']:.4%}")

        # ------------------------------------------------------------------
        # Representation 2: Euclidean distance to every codebook centroid.
        # StandardScaler is fit on TRAIN distances only, then applied to val.
        # ------------------------------------------------------------------
        print(f"\n[K={k}] Building DISTANCE representation...")
        distance_start = time.perf_counter()
        dist_train = make_distance_features(X_train, centers)
        dist_val = make_distance_features(X_val, centers)
        scaler = StandardScaler()
        dist_train_scaled = scaler.fit_transform(dist_train).astype(np.float32, copy=False)
        dist_val_scaled = scaler.transform(dist_val).astype(np.float32, copy=False)
        distance_transform_seconds = time.perf_counter() - distance_start
        del dist_train, dist_val

        print(f"[K={k}] Training DISTANCE representation classifier...")
        distance_model, distance_c, distance_metrics, distance_search, distance_fit_seconds = fit_and_select_classifier(
            dist_train_scaled, y_train, dist_val_scaled, y_val,
            args.c_values, args.max_iter_logreg, args.seed, "distance",
        )
        c_search_rows.extend([{**row, "K": int(k)} for row in distance_search])
        distance_pred = distance_model.predict(dist_val_scaled)
        save_classifier_outputs(
            output_dir, k, "distance", distance_model, y_val,
            distance_pred, distance_c, distance_metrics,
        )
        distance_row = {
            "K": int(k),
            "representation": "distance",
            "total_prototypes": int(k),
            "selected_C": distance_c,
            "validation_accuracy": distance_metrics["accuracy"],
            "validation_macro_f1": distance_metrics["macro_f1"],
            "validation_weighted_f1": distance_metrics["weighted_f1"],
            "kmeans_fit_seconds": float(kmeans_seconds),
            "distance_transform_seconds": float(distance_transform_seconds),
            "classifier_search_seconds": float(distance_fit_seconds),
            "inertia": float(kmeans.inertia_),
            "smallest_cluster_size": int(cluster_sizes.min()),
            "largest_cluster_size": int(cluster_sizes.max()),
            "empty_clusters": empty_clusters,
        }
        summary_rows.append(distance_row)
        joblib.dump(
            {
                "method": "global_codebook_euclidean_distances_standardized_logistic_regression",
                "K": int(k),
                "kmeans": kmeans,
                "distance_scaler": scaler,
                "classifier": distance_model,
                "best_C": distance_c,
                "class_names": CLASS_NAMES,
                "validation_metrics": distance_metrics,
            },
            output_dir / f"complete_model_distance_k{k}.joblib",
        )
        print(f"Distance representation: best C={distance_c:g}, "
              f"accuracy={distance_metrics['accuracy']:.4%}, "
              f"Macro-F1={distance_metrics['macro_f1']:.4%}")

        del hard_train, hard_val, dist_train_scaled, dist_val_scaled

    # ----------------------------------------------------------------------
    # Save consolidated results and plots.
    # ----------------------------------------------------------------------
    results = pd.DataFrame(summary_rows).sort_values(
        by=["K", "representation"], ascending=[True, True]
    ).reset_index(drop=True)
    results.to_csv(output_dir / "validation_results.csv", index=False)
    pd.DataFrame(c_search_rows).to_csv(output_dir / "c_value_search.csv", index=False)
    plot_comparisons(results, output_dir, baseline)

    best_row = results.sort_values(
        by=["validation_accuracy", "validation_macro_f1"], ascending=False
    ).iloc[0]
    best_summary = {
        "experiment": "shared_global_protowords_vocabulary",
        "feature_extractor": "frozen DINOv2 ViT-S/14 global CLS features",
        "feature_dimension": int(X_train.shape[1]),
        "normalization": "L2 per image",
        "training_images": int(len(y_train)),
        "validation_images": int(len(y_val)),
        "official_test_used": False,
        "k_values": [int(k) for k in args.ks],
        "representations": ["hard one-hot assignment", "Euclidean distances to all prototypes"],
        "c_values": [float(c) for c in args.c_values],
        "random_seed": int(args.seed),
        "kmeans_n_init": int(args.n_init),
        "baseline_validation_accuracy": float(baseline["validation_accuracy"]),
        "baseline_validation_macro_f1": float(baseline["validation_macro_f1"]),
        "best_validation_configuration": {
            "K": int(best_row["K"]),
            "representation": str(best_row["representation"]),
            "selected_C": float(best_row["selected_C"]),
            "validation_accuracy": float(best_row["validation_accuracy"]),
            "validation_macro_f1": float(best_row["validation_macro_f1"]),
        },
        "total_runtime_seconds": float(time.perf_counter() - run_start),
        "results_csv": "validation_results.csv",
    }
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    print("\n" + "=" * 76)
    print("VALIDATION RESULTS — SIX CONFIGURATIONS")
    print("=" * 76)
    print(results[[
        "K", "representation", "selected_C", "validation_accuracy",
        "validation_macro_f1", "classifier_search_seconds"
    ]].to_string(index=False, formatters={
        "validation_accuracy": "{:.4%}".format,
        "validation_macro_f1": "{:.4%}".format,
    }))
    print("\nBest validation configuration:")
    print(json.dumps(best_summary["best_validation_configuration"], indent=2))
    print(f"\nAll outputs saved to: {output_dir.resolve()}")
    print("The official CIFAR-10 test split was not loaded or evaluated.")
    print("Select a configuration using validation results, then evaluate it once on test.")


if __name__ == "__main__":
    main()
