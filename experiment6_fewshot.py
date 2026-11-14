#!/usr/bin/env python3
"""Experiment 6: strict few-shot recognition for ProtoWords.

Primary protocol (strict few-shot):
- CIFAR-10 train split supplies only n labelled images per class.
- The same sampled labelled subset is used for every compared method.
- Any supervised classifier and the shared ProtoWords codebook are fitted
  using only that labelled subset.
- Validation is used to select Logistic Regression C.
- The official test split is used only for final evaluation.

Methods:
1. Continuous DINOv2 features + Logistic Regression.
2. One class-mean prototype per class (Euclidean nearest mean).
3. One class-mean prototype per class (cosine nearest mean).
4. Shared global ProtoWords: K-Means on the few-shot subset, Euclidean
   distances to centroids, StandardScaler + Logistic Regression.

This is deliberately separate from a few-labels-plus-extra-unlabeled-data
setting, where a vocabulary learned from all 45,000 training images is reused.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import Normalizer, StandardScaler

CACHE_NAME = "cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz"
DEFAULT_SHOTS = [5, 10, 25, 50, 100, 4500]  # 4500/class is the full 45,000-image training split
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def find_project_root(start: Path) -> Path:
    """Find the nearest parent containing the expected outputs directory/cache."""
    for candidate in [start, *start.parents]:
        if (candidate / "outputs" / CACHE_NAME).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find outputs/{CACHE_NAME}. Run from inside ProtoWords or use --project-root."
    )


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return Normalizer(norm="l2").fit_transform(x)


def sample_balanced_subset(
    y_train: np.ndarray,
    shots_per_class: int,
    seed: int,
) -> np.ndarray:
    """Return a balanced, reproducible, nested subset of training-row indices."""
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    classes = np.unique(y_train)
    for cls in classes:
        cls_idx = np.flatnonzero(y_train == cls)
        if len(cls_idx) < shots_per_class:
            raise ValueError(
                f"Class {cls} has only {len(cls_idx)} train examples; requested {shots_per_class}."
            )
        shuffled = rng.permutation(cls_idx)
        selected.extend(shuffled[:shots_per_class].tolist())
    # Shuffle the combined subset so rows are not class-ordered.
    return rng.permutation(np.asarray(selected, dtype=np.int64))


def choose_c_and_evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_values: list[float],
    seed: int,
    method_name: str,
) -> tuple[dict, list[dict]]:
    """Select C on validation; evaluate the selected training-only model on test."""
    best_model = None
    best_c = None
    best_val_acc = -1.0
    search_rows: list[dict] = []
    convergence_messages: list[str] = []

    for c_value in c_values:
        model = LogisticRegression(
            C=float(c_value),
            solver="lbfgs",
            max_iter=3000,
            random_state=seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_train, y_train)
        warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        if warned:
            convergence_messages.append(f"{method_name}: C={c_value} convergence warning")

        val_pred = model.predict(x_val)
        val_acc = float(accuracy_score(y_val, val_pred))
        search_rows.append({
            "method": method_name,
            "C": float(c_value),
            "validation_accuracy": val_acc,
            "convergence_warning": bool(warned),
        })
        # Sorted C grid means ties retain the smaller C.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_c = float(c_value)
            best_model = model

    assert best_model is not None and best_c is not None
    test_pred = best_model.predict(x_test)
    result = {
        "method": method_name,
        "selected_C": best_c,
        "validation_accuracy": best_val_acc,
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_macro_f1": float(f1_score(y_test, test_pred, average="macro")),
        "test_micro_f1": float(f1_score(y_test, test_pred, average="micro")),
        "convergence_warning": bool(convergence_messages),
    }
    return result, search_rows


def class_prototype_predictions(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
    classes: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Nearest one-mean-per-class prototype."""
    prototypes = np.stack([x_fit[y_fit == cls].mean(axis=0) for cls in classes]).astype(np.float32)
    if metric == "euclidean":
        distances = euclidean_distances(x_eval, prototypes)
        return classes[np.argmin(distances, axis=1)]
    if metric == "cosine":
        proto_norm = l2_normalize(prototypes)
        similarities = cosine_similarity(x_eval, proto_norm)
        return classes[np.argmax(similarities, axis=1)]
    raise ValueError(f"Unknown prototype metric: {metric}")


def prototype_metric_result(
    method: str,
    y_val: np.ndarray,
    val_pred: np.ndarray,
    y_test: np.ndarray,
    test_pred: np.ndarray,
) -> dict:
    return {
        "method": method,
        "selected_C": np.nan,
        "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_macro_f1": float(f1_score(y_test, test_pred, average="macro")),
        "test_micro_f1": float(f1_score(y_test, test_pred, average="micro")),
        "convergence_warning": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict few-shot ProtoWords Experiment 6.")
    parser.add_argument("--project-root", type=Path, default=None,
                        help="ProtoWords directory; auto-detected if omitted.")
    parser.add_argument("--cache", type=Path, default=None,
                        help="Optional path to the existing DINOv2 feature .npz cache.")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Output directory; default: outputs/e6_fewshot.")
    parser.add_argument("--shots", type=int, nargs="+", default=DEFAULT_SHOTS,
                        help="Labelled examples per class (default: 5 10 25 50 100).")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds (default: 42 43 44).")
    parser.add_argument("--k", type=int, default=32,
                        help="Shared K-Means vocabulary size for strict ProtoWords (default: 32).")
    parser.add_argument("--cs", type=float, nargs="+", default=DEFAULT_C_VALUES,
                        help="Logistic Regression C values searched on validation.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = args.project_root.resolve() if args.project_root else find_project_root(script_dir)
    cache_path = args.cache.resolve() if args.cache else project_root / "outputs" / CACHE_NAME
    outdir = args.outdir.resolve() if args.outdir else project_root / "outputs" / "e6_fewshot"
    outdir.mkdir(parents=True, exist_ok=True)

    if not cache_path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")
    if args.k < 2:
        raise ValueError("--k must be at least 2.")
    if any(s <= 0 for s in args.shots):
        raise ValueError("All --shots values must be positive.")
    if sorted(args.cs) != args.cs or any(c <= 0 for c in args.cs):
        raise ValueError("--cs must be positive and sorted in ascending order.")

    print("=" * 76)
    print("EXPERIMENT 6: STRICT FEW-SHOT RECOGNITION")
    print("=" * 76)
    print("project root:", project_root)
    print("feature cache:", cache_path)
    print("output directory:", outdir)
    print("shots per class:", args.shots)
    print("seeds:", args.seeds)
    print("strict ProtoWords codebook K:", args.k)
    print("C values:", args.cs)

    t_total = time.time()
    z = np.load(cache_path)
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    missing = [key for key in required if key not in z.files]
    if missing:
        raise KeyError(f"Feature cache is missing arrays: {missing}. Available: {z.files}")

    x_train_all = l2_normalize(z["X_train"])
    y_train_all = np.asarray(z["y_train"]).astype(int)
    x_val = l2_normalize(z["X_val"])
    y_val = np.asarray(z["y_val"]).astype(int)
    x_test = l2_normalize(z["X_test"])
    y_test = np.asarray(z["y_test"]).astype(int)
    classes = np.unique(y_train_all)

    if len(classes) != 10:
        raise ValueError(f"Expected 10 CIFAR-10 classes; found {len(classes)}: {classes}")
    class_counts = {int(cls): int((y_train_all == cls).sum()) for cls in classes}
    print("loaded feature shapes:", x_train_all.shape, x_val.shape, x_test.shape)
    print("training class counts:", class_counts)
    print("validation/test labels remain fixed for all runs.")

    original_train_indices = (
        np.asarray(z["train_indices"]).astype(int)
        if "train_indices" in z.files else np.arange(len(y_train_all))
    )

    run_rows: list[dict] = []
    c_search_rows: list[dict] = []
    sampled_rows: list[dict] = []

    for seed in args.seeds:
        print(f"\n################ SEED {seed} ################")
        # A single shuffled order per class means smaller budgets are nested
        # within larger budgets for a given seed.
        class_orders: dict[int, np.ndarray] = {}
        rng = np.random.default_rng(seed)
        for cls in classes:
            class_orders[int(cls)] = rng.permutation(np.flatnonzero(y_train_all == cls))

        for shots in args.shots:
            selected = np.concatenate([class_orders[int(cls)][:shots] for cls in classes])
            # Shuffle the combined subset, but preserve the same rows for all methods.
            selected = rng.permutation(selected)
            x_few = x_train_all[selected]
            y_few = y_train_all[selected]
            expected_total = shots * len(classes)
            if len(selected) != expected_total:
                raise RuntimeError("Incorrect few-shot subset size.")
            if args.k >= len(selected):
                raise ValueError(
                    f"K={args.k} must be smaller than the number of selected images ({len(selected)})."
                )

            print(f"\n--- {shots}-shot: {len(selected)} labelled images; seed={seed} ---")
            for row_idx, label in zip(selected, y_few):
                sampled_rows.append({
                    "seed": seed,
                    "shots_per_class": shots,
                    "class_id": int(label),
                    "feature_train_row": int(row_idx),
                    "cifar_original_index": int(original_train_indices[row_idx]),
                })

            # Method 1: Continuous DINOv2 + Logistic Regression.
            start = time.time()
            baseline_result, c_rows = choose_c_and_evaluate(
                x_few, y_few, x_val, y_val, x_test, y_test,
                args.cs, seed, "continuous_dinov2_logreg",
            )
            baseline_result.update({"seed": seed, "shots_per_class": shots,
                                    "train_images": len(selected), "vocabulary_K": np.nan,
                                    "runtime_seconds": time.time() - start})
            run_rows.append(baseline_result)
            c_search_rows.extend([dict(r, seed=seed, shots_per_class=shots) for r in c_rows])
            print(f"  continuous LR: val={baseline_result['validation_accuracy']:.4f}, "
                  f"test={baseline_result['test_accuracy']:.4f}, C={baseline_result['selected_C']}")

            # Methods 2 and 3: one class mean prototype per class.
            for metric in ("euclidean", "cosine"):
                start = time.time()
                val_pred = class_prototype_predictions(x_few, y_few, x_val, classes, metric)
                test_pred = class_prototype_predictions(x_few, y_few, x_test, classes, metric)
                result = prototype_metric_result(
                    f"class_mean_prototype_{metric}", y_val, val_pred, y_test, test_pred
                )
                result.update({"seed": seed, "shots_per_class": shots,
                               "train_images": len(selected), "vocabulary_K": np.nan,
                               "runtime_seconds": time.time() - start})
                run_rows.append(result)
                print(f"  class mean ({metric}): val={result['validation_accuracy']:.4f}, "
                      f"test={result['test_accuracy']:.4f}")

            # Method 4: Strict shared ProtoWords. The codebook sees only the
            # selected few-shot feature vectors, not the remaining train pool.
            start = time.time()
            km = KMeans(
                n_clusters=args.k,
                n_init=10,
                max_iter=300,
                random_state=seed,
                algorithm="lloyd",
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                km.fit(x_few)
            km_convergence_warning = any(
                issubclass(w.category, ConvergenceWarning) for w in caught
            )
            d_train = euclidean_distances(x_few, km.cluster_centers_).astype(np.float32)
            d_val = euclidean_distances(x_val, km.cluster_centers_).astype(np.float32)
            d_test = euclidean_distances(x_test, km.cluster_centers_).astype(np.float32)
            scaler = StandardScaler().fit(d_train)
            d_train = scaler.transform(d_train).astype(np.float32)
            d_val = scaler.transform(d_val).astype(np.float32)
            d_test = scaler.transform(d_test).astype(np.float32)
            proto_result, c_rows = choose_c_and_evaluate(
                d_train, y_few, d_val, y_val, d_test, y_test,
                args.cs, seed, f"strict_global_protowords_distance_K{args.k}",
            )
            proto_result.update({"seed": seed, "shots_per_class": shots,
                                 "train_images": len(selected), "vocabulary_K": args.k,
                                 "runtime_seconds": time.time() - start,
                                 "convergence_warning": bool(proto_result["convergence_warning"] or km_convergence_warning)})
            run_rows.append(proto_result)
            c_search_rows.extend([dict(r, seed=seed, shots_per_class=shots) for r in c_rows])
            np.savez_compressed(
                outdir / f"strict_codebook_seed{seed}_shots{shots}_K{args.k}.npz",
                centroids=km.cluster_centers_.astype(np.float32),
                scaler_mean=scaler.mean_.astype(np.float32),
                scaler_scale=scaler.scale_.astype(np.float32),
                selected_train_rows=selected,
                selected_labels=y_few,
            )
            print(f"  strict ProtoWords K={args.k}: val={proto_result['validation_accuracy']:.4f}, "
                  f"test={proto_result['test_accuracy']:.4f}, C={proto_result['selected_C']}")

    runs = pd.DataFrame(run_rows)
    c_search = pd.DataFrame(c_search_rows)
    sampled = pd.DataFrame(sampled_rows)
    runs.to_csv(outdir / "fewshot_runs.csv", index=False)
    c_search.to_csv(outdir / "fewshot_c_search.csv", index=False)
    sampled.drop_duplicates().to_csv(outdir / "fewshot_sampled_indices.csv", index=False)

    summary = (
        runs.groupby(["shots_per_class", "method"], as_index=False)
        .agg(
            mean_validation_accuracy=("validation_accuracy", "mean"),
            std_validation_accuracy=("validation_accuracy", "std"),
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            mean_test_macro_f1=("test_macro_f1", "mean"),
            std_test_macro_f1=("test_macro_f1", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
    )
    summary.to_csv(outdir / "fewshot_summary.csv", index=False)

    # Learning curves: mean +/- standard deviation over seeds.
    for metric, ylabel, filename, mean_col, std_col in [
        ("test_accuracy", "Test accuracy", "fewshot_learning_curve_accuracy.png",
         "mean_test_accuracy", "std_test_accuracy"),
        ("test_macro_f1", "Test macro-F1", "fewshot_learning_curve_macro_f1.png",
         "mean_test_macro_f1", "std_test_macro_f1"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for method in summary["method"].unique():
            part = summary[summary["method"] == method].sort_values("shots_per_class")
            ax.errorbar(
                part["shots_per_class"], part[mean_col], yerr=part[std_col].fillna(0),
                marker="o", capsize=3, label=method,
            )
        ax.set_xlabel("Labeled training images per class")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Experiment 6: Few-shot learning curve ({metric})")
        ax.set_xscale("log")
        ax.set_xticks(sorted(summary["shots_per_class"].unique()))
        ax.set_xticklabels([str(x) for x in sorted(summary["shots_per_class"].unique())])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=180)
        plt.close(fig)

    metadata = {
        "experiment": "E6_strict_few_shot_recognition",
        "dataset": "CIFAR-10",
        "feature_extractor": "Frozen DINOv2 ViT-S/14 global CLS features",
        "feature_dimension": int(x_train_all.shape[1]),
        "train_split_size": int(len(y_train_all)),
        "validation_size": int(len(y_val)),
        "official_test_size": int(len(y_test)),
        "shots_per_class": args.shots,
        "full_training_budget_per_class": int(min(class_counts.values())),
        "total_labeled_images_per_budget": {str(s): int(s * len(classes)) for s in args.shots},
        "seeds": args.seeds,
        "strict_protocol": True,
        "strict_protocol_definition": (
            "Classifiers and the shared K-Means codebook are fitted using only the balanced "
            "few-shot labeled subset. The same selected examples are used by every method."
        ),
        "methods": [
            "continuous_dinov2_logreg",
            "class_mean_prototype_euclidean",
            "class_mean_prototype_cosine",
            f"strict_global_protowords_distance_K{args.k}",
        ],
        "protowords": {
            "K": args.k,
            "representation": "Euclidean distances to K centroids",
            "codebook_fit_data": "few-shot labeled training subset only",
            "scaler_fit_data": "few-shot training distance representations only",
        },
        "logistic_regression_C_values": args.cs,
        "model_selection": "Validation accuracy; official test used only after C selection.",
        "note_on_unlabeled_data": (
            "This is the strict labeled-only setting. Reusing a codebook learned from all 45,000 "
            "training images would be a separate few-labels-plus-unlabeled-data experiment and is not mixed here."
        ),
        "feature_cache": str(cache_path),
        "runtime_seconds": time.time() - t_total,
    }
    with open(outdir / "fewshot_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 76)
    print("EXPERIMENT 6 COMPLETE")
    print("=" * 76)
    print(summary.to_string(index=False))
    print("\nSaved outputs to:", outdir)
    print("Runtime: {:.1f}s".format(time.time() - t_total))


if __name__ == "__main__":
    main()
