#!/usr/bin/env python3
"""
ProtoWords - full CIFAR-10 baseline
=================================

Baseline:
    CIFAR-10 images
        -> frozen DINOv2 ViT-S/14
        -> image-level CLS feature (384 dimensions expected)
        -> per-image L2 normalization
        -> multinomial logistic regression
        -> final evaluation on the official 10,000-image test set

Protocol:
    * Use all 50,000 official CIFAR-10 training images as development data.
    * Split development data into 45,000 train + 5,000 validation images
      using a deterministic, stratified split.
    * Select logistic-regression C using validation accuracy (macro-F1 as
      a tie-breaker).
    * Refit the selected classifier on all 50,000 development images.
    * Evaluate once on all 10,000 official test images.

DINOv2 is frozen; this script does NOT fine-tune the vision model.
Features are cached so future experiments can reuse them without running
DINOv2 again. The previous 1,000/1,000 experiment cache is not overwritten.

Run from ~/ProtoWords, for example:
    python baseline_dinov2_cifar10_full.py --batch-size 16 --num-workers 4

If GPU memory is insufficient, rerun with --batch-size 8. The cache is only
written after all three feature splits have been extracted successfully.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


# CIFAR-10's official class order.
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

MODEL_HUB = "facebookresearch/dinov2"
MODEL_NAME = "dinov2_vits14"
EXPECTED_FEATURE_DIM = 384  # ViT-S/14 CLS feature dimension.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full CIFAR-10 baseline using frozen DINOv2 ViT-S/14 + logistic regression."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Directory for the CIFAR-10 download (default: ./data).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Directory for feature caches and results (default: ./outputs).")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="DINOv2 feature-extraction batch size. Start with 16 on a 6-GB GPU.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers. Use 2 if your laptop becomes unresponsive.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the train/validation split and classifier.")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Square DINOv2 input resolution (224 is the planned baseline).")
    parser.add_argument("--max-iter", type=int, default=2000,
                        help="Maximum iterations for each logistic-regression fit.")
    parser.add_argument("--c-values", type=float, nargs="+",
                        default=[0.01, 0.1, 1.0, 10.0],
                        help="C values selected using validation data.")
    parser.add_argument("--force-extract", action="store_true",
                        help="Ignore the full-dataset cache and extract all features again.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU execution (normally leave this off to use the RTX GPU).")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.image_size < 14 or args.image_size % 14 != 0:
        parser.error("--image-size must be a positive multiple of 14; use 224 for this baseline")
    if args.max_iter < 1:
        parser.error("--max-iter must be at least 1")
    if not args.c_values or any(c <= 0 for c in args.c_values):
        parser.error("All --c-values must be positive")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transform(image_size: int) -> transforms.Compose:
    # CIFAR-10 images are square, so direct bicubic resizing avoids an
    # unnecessary center crop while producing the ViT-S/14 input size.
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_datasets(data_dir: Path, image_size: int):
    data_dir.mkdir(parents=True, exist_ok=True)
    transform = make_transform(image_size)

    print("\n[1/7] Loading CIFAR-10 (downloads it if it is not already present)...")
    train_full = datasets.CIFAR10(
        root=str(data_dir), train=True, transform=transform, download=True
    )
    test_full = datasets.CIFAR10(
        root=str(data_dir), train=False, transform=transform, download=True
    )

    if len(train_full) != 50000 or len(test_full) != 10000:
        raise RuntimeError(
            f"Unexpected CIFAR-10 sizes: train={len(train_full)}, test={len(test_full)}"
        )
    return train_full, test_full


def make_splits(train_full, seed: int):
    """Deterministic, stratified 45k/5k split of CIFAR-10's 50k train set."""
    all_indices = np.arange(len(train_full))
    labels = np.asarray(train_full.targets, dtype=np.int64)

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=5000,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    # Sanity checks: CIFAR-10 is balanced, so this split should have
    # exactly 4,500 train and 500 validation examples per class.
    train_counts = np.bincount(labels[train_indices], minlength=10)
    val_counts = np.bincount(labels[val_indices], minlength=10)
    if not np.all(train_counts == 4500) or not np.all(val_counts == 500):
        raise RuntimeError(
            f"Unexpected class counts. Train={train_counts.tolist()}, Val={val_counts.tolist()}"
        )

    train_subset = Subset(train_full, train_indices.tolist())
    val_subset = Subset(train_full, val_indices.tolist())
    return train_subset, val_subset, train_indices, val_indices


def load_dinov2(device: torch.device):
    print("\n[2/7] Loading pretrained DINOv2 ViT-S/14...")
    print("The first use may download the official model code and pretrained weights.")
    model = torch.hub.load(MODEL_HUB, MODEL_NAME)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract_features(
    model,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract one normalized-by-model CLS feature vector per image.

    The returned vectors are not L2-normalized here. L2 normalization is
    part of the saved sklearn pipeline, ensuring the classifier and any
    later inference use exactly the same preprocessing.
    """
    use_cuda = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )

    features_parts = []
    labels_parts = []
    start = time.time()

    print(f"\nExtracting {split_name} features ({len(dataset):,} images)...")
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=f"{split_name} features", unit="batch"):
            images = images.to(device, non_blocking=use_cuda)

            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(images)
            else:
                output = model(images)

            # DINOv2 Hub model normally returns the CLS feature tensor.
            # Handle a feature dictionary as well, for compatibility.
            if isinstance(output, dict):
                if "x_norm_clstoken" not in output:
                    raise RuntimeError(
                        "DINOv2 returned a dictionary without 'x_norm_clstoken'."
                    )
                output = output["x_norm_clstoken"]

            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                shape = getattr(output, "shape", None)
                raise RuntimeError(
                    f"Expected a 2-D image-level feature tensor; got {type(output)} shape={shape}"
                )

            features_parts.append(output.float().cpu().numpy())
            labels_parts.append(labels.numpy().astype(np.int64, copy=False))

    X = np.concatenate(features_parts, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(labels_parts, axis=0).astype(np.int64, copy=False)

    if X.shape[0] != len(dataset):
        raise RuntimeError(f"Extracted {X.shape[0]} {split_name} features for {len(dataset)} images")
    if X.shape[1] != EXPECTED_FEATURE_DIM:
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURE_DIM} DINOv2-S features, got {X.shape[1]}. "
            "Check that the loaded model is dinov2_vits14."
        )

    print(f"{split_name} feature matrix: {X.shape}; elapsed: {time.time() - start:.1f}s")
    return X, y


def cache_metadata(args: argparse.Namespace) -> Dict:
    return {
        "dataset": "CIFAR-10",
        "model_hub": MODEL_HUB,
        "model_name": MODEL_NAME,
        "feature_type": "DINOv2 normalized CLS token returned by Hub model",
        "image_size": args.image_size,
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "split_seed": args.seed,
        "split_sizes": {"train": 45000, "validation": 5000, "test": 10000},
    }


def get_feature_cache_path(output_dir: Path, args: argparse.Namespace) -> Path:
    # This unique name prevents the older 1k/1k cache from being reused.
    name = (
        f"cifar10_dinov2_vits14_{args.image_size}px_"
        f"train45000_val5000_test10000_seed{args.seed}.npz"
    )
    return output_dir / name


def extract_or_load_features(args, device, output_dir):
    cache_path = get_feature_cache_path(output_dir, args)
    expected_metadata = cache_metadata(args)

    if cache_path.exists() and not args.force_extract:
        print(f"\n[3/7] Found full-dataset feature cache:\n{cache_path}")
        print("Loading cached features; DINOv2 extraction will be skipped.")
        with np.load(cache_path, allow_pickle=False) as cache:
            X_train = cache["X_train"].astype(np.float32, copy=False)
            y_train = cache["y_train"].astype(np.int64, copy=False)
            X_val = cache["X_val"].astype(np.float32, copy=False)
            y_val = cache["y_val"].astype(np.int64, copy=False)
            X_test = cache["X_test"].astype(np.float32, copy=False)
            y_test = cache["y_test"].astype(np.int64, copy=False)
            train_indices = cache["train_indices"].astype(np.int64, copy=False)
            val_indices = cache["val_indices"].astype(np.int64, copy=False)
            stored_metadata = json.loads(str(cache["metadata"].item()))
        if stored_metadata != expected_metadata:
            raise RuntimeError(
                "The cache metadata does not match this run's settings. "
                "Use --force-extract to create a fresh cache."
            )
    else:
        train_full, test_full = load_datasets(args.data_dir, args.image_size)
        train_subset, val_subset, train_indices, val_indices = make_splits(
            train_full, args.seed
        )
        model = load_dinov2(device)

        print("\n[3/7] Extracting all three splits. This is the GPU-heavy stage.")
        X_train, y_train = extract_features(
            model, train_subset, device, args.batch_size,
            args.num_workers, "Train",
        )
        X_val, y_val = extract_features(
            model, val_subset, device, args.batch_size,
            args.num_workers, "Validation",
        )
        X_test, y_test = extract_features(
            model, test_full, device, args.batch_size,
            args.num_workers, "Test",
        )

        # Ensure labels and split sizes are as expected.
        expected_shapes = {
            "train": (45000, EXPECTED_FEATURE_DIM),
            "validation": (5000, EXPECTED_FEATURE_DIM),
            "test": (10000, EXPECTED_FEATURE_DIM),
        }
        actual_shapes = {
            "train": X_train.shape,
            "validation": X_val.shape,
            "test": X_test.shape,
        }
        if actual_shapes != expected_shapes:
            raise RuntimeError(f"Unexpected feature shapes: {actual_shapes}")

        print("\nSaving the complete feature cache...")
        metadata_string = json.dumps(expected_metadata, sort_keys=True)
        # Write to a temporary path, then rename. This avoids leaving a
        # seemingly valid cache if the machine is interrupted while saving.
        temp_cache = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temp_cache,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            train_indices=train_indices,
            val_indices=val_indices,
            metadata=np.asarray(metadata_string),
        )
        temp_cache.replace(cache_path)
        print(f"Feature cache saved: {cache_path}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Strong consistency checks also apply when loading an existing cache.
    expected = ((45000, 384), (5000, 384), (10000, 384))
    actual = (X_train.shape, X_val.shape, X_test.shape)
    if actual != expected:
        raise RuntimeError(f"Cached feature shapes are incorrect: {actual}; expected {expected}")
    if len(y_train) != 45000 or len(y_val) != 5000 or len(y_test) != 10000:
        raise RuntimeError("Cached label counts do not match the 45k/5k/10k protocol")

    return X_train, y_train, X_val, y_val, X_test, y_test, train_indices, val_indices, cache_path


def make_classifier(C: float, max_iter: int, seed: int) -> Pipeline:
    # Normalizer applies L2 normalization independently to each sample.
    # It has no learned dataset-level statistics, so it is safe inside a pipeline.
    return Pipeline([
        ("l2_normalizer", Normalizer(norm="l2")),
        ("logistic_regression", LogisticRegression(
            C=float(C),
            solver="lbfgs",
            max_iter=max_iter,
            random_state=seed,
            tol=1e-4,
        )),
    ])


def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def save_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("ProtoWords | Full CIFAR-10 DINOv2 Baseline")
    print("=" * 72)
    print(f"Environment Python: {__import__('sys').version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {__import__('torchvision').__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        print("WARNING: CUDA is not active. Feature extraction will run on CPU and may be slow.")

    print("\nExperimental protocol:")
    print("  CIFAR-10 development split: 45,000 train + 5,000 validation")
    print("  Official CIFAR-10 test split: 10,000 test images")
    print(f"  Split seed: {args.seed}; image size: {args.image_size}; batch size: {args.batch_size}")
    print(f"  Validation C candidates: {args.c_values}")

    total_start = time.time()
    (X_train, y_train, X_val, y_val, X_test, y_test,
     train_indices, val_indices, cache_path) = extract_or_load_features(
        args, device, args.output_dir
    )

    # --------------------------------------------------------
    # Select C using validation data only.
    # --------------------------------------------------------
    print("\n[4/7] Selecting logistic-regression C using validation data...")
    validation_rows = []
    best_model = None
    best_C = None
    best_key = None

    for C in args.c_values:
        print(f"\nFitting logistic regression with C={C}...")
        candidate = make_classifier(C, args.max_iter, args.seed)
        fit_start = time.time()
        convergence_messages = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            candidate.fit(X_train, y_train)
        fit_seconds = time.time() - fit_start
        convergence_messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]

        val_predictions = candidate.predict(X_val)
        val_metrics = evaluate_predictions(y_val, val_predictions)
        row = {
            "C": float(C),
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_micro_f1": val_metrics["micro_f1"],
            "fit_seconds": round(fit_seconds, 3),
            "convergence_warning": bool(convergence_messages),
        }
        validation_rows.append(row)
        print(
            f"Validation accuracy={val_metrics['accuracy']:.4f} | "
            f"macro-F1={val_metrics['macro_f1']:.4f} | "
            f"fit={fit_seconds:.1f}s"
        )
        if convergence_messages:
            print("WARNING: Logistic regression did not fully converge for this C.")

        # Primary selection: validation accuracy. Tie-break: macro F1.
        key = (val_metrics["accuracy"], val_metrics["macro_f1"], -float(C))
        if best_key is None or key > best_key:
            best_key = key
            best_C = float(C)
            best_model = candidate

    if best_C is None:
        raise RuntimeError("Could not select a logistic-regression model")

    validation_csv = args.output_dir / "baseline_validation_c_search.csv"
    save_csv(
        validation_csv,
        validation_rows,
        ["C", "validation_accuracy", "validation_macro_f1",
         "validation_micro_f1", "fit_seconds", "convergence_warning"],
    )
    print(f"\nSelected C={best_C} using validation accuracy (macro-F1 tie-break).")
    print(f"Validation search saved: {validation_csv}")

    # --------------------------------------------------------
    # Refit chosen configuration on all 50,000 development images.
    # The official test split remains untouched until final evaluation.
    # --------------------------------------------------------
    print("\n[5/7] Refitting selected classifier on all 50,000 development images...")
    X_development = np.concatenate([X_train, X_val], axis=0)
    y_development = np.concatenate([y_train, y_val], axis=0)

    final_model = make_classifier(best_C, args.max_iter, args.seed)
    final_fit_start = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        final_model.fit(X_development, y_development)
    final_fit_seconds = time.time() - final_fit_start
    final_convergence_warning = any(
        issubclass(w.category, ConvergenceWarning) for w in caught
    )
    if final_convergence_warning:
        print("WARNING: The final logistic-regression fit reached max_iter before convergence.")
        print("Consider increasing --max-iter (for example, 3000) and rerunning.")

    # --------------------------------------------------------
    # Final test evaluation - do not use test results to tune settings.
    # --------------------------------------------------------
    print("\n[6/7] Evaluating exactly once on the official 10,000-image test split...")
    test_predictions = final_model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, test_predictions)

    class_report_text = classification_report(
        y_test,
        test_predictions,
        labels=np.arange(10),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    class_report_dict = classification_report(
        y_test,
        test_predictions,
        labels=np.arange(10),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    print("\n" + "=" * 72)
    print("FINAL CIFAR-10 BASELINE RESULTS")
    print("=" * 72)
    print(f"Training used for final fit: {len(y_development):,}")
    print(f"Validation used for C selection: {len(y_val):,}")
    print(f"Official test images: {len(y_test):,}")
    print(f"Selected C: {best_C}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test micro-F1: {test_metrics['micro_f1']:.4f}")
    print("\nPer-class classification report:")
    print(class_report_text)

    # --------------------------------------------------------
    # Save all outputs for future ProtoWords comparisons.
    # --------------------------------------------------------
    print("[7/7] Saving model, metrics, predictions, and plots...")
    model_path = args.output_dir / "baseline_dinov2_vits14_logreg_full.joblib"
    joblib.dump(final_model, model_path)

    metrics = {
        "experiment": "DINOv2 ViT-S/14 frozen global CLS features + logistic regression",
        "dataset": "CIFAR-10",
        "protocol": {
            "development_train_for_C_selection": 45000,
            "validation_for_C_selection": 5000,
            "final_classifier_refit_train_plus_validation": 50000,
            "official_test": 10000,
            "split_seed": args.seed,
        },
        "model": MODEL_NAME,
        "feature_dimension": EXPECTED_FEATURE_DIM,
        "input_size": args.image_size,
        "selected_C": best_C,
        "classifier": "sklearn LogisticRegression(solver='lbfgs')",
        "feature_normalization": "L2 per image, applied by sklearn Normalizer in saved Pipeline",
        "validation_search": validation_rows,
        "test_metrics": test_metrics,
        "classification_report": class_report_dict,
        "final_classifier_fit_seconds": final_fit_seconds,
        "final_fit_convergence_warning": final_convergence_warning,
        "feature_cache": str(cache_path),
        "saved_model": str(model_path),
    }

    metrics_path = args.output_dir / "baseline_dinov2_cifar10_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_path = args.output_dir / "baseline_dinov2_cifar10_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("ProtoWords: DINOv2 CIFAR-10 Baseline\n")
        f.write("=====================================\n\n")
        f.write("Method: frozen DINOv2 ViT-S/14 global CLS features + logistic regression\n")
        f.write("Split: 45,000 train / 5,000 validation / 10,000 official test\n")
        f.write(f"Split seed: {args.seed}\n")
        f.write(f"Image size: {args.image_size}\n")
        f.write(f"Selected C: {best_C}\n\n")
        f.write(f"Test accuracy: {test_metrics['accuracy']:.6f}\n")
        f.write(f"Test macro-F1: {test_metrics['macro_f1']:.6f}\n")
        f.write(f"Test micro-F1: {test_metrics['micro_f1']:.6f}\n\n")
        f.write("Validation C search:\n")
        for row in validation_rows:
            f.write(json.dumps(row) + "\n")
        f.write("\nPer-class classification report:\n")
        f.write(class_report_text)

    predictions_path = args.output_dir / "baseline_dinov2_cifar10_test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_index_in_official_test", "true_label", "true_class", "predicted_label", "predicted_class"])
        for i, (true_label, pred_label) in enumerate(zip(y_test, test_predictions)):
            writer.writerow([i, int(true_label), CLASS_NAMES[int(true_label)], int(pred_label), CLASS_NAMES[int(pred_label)]])

    cm = confusion_matrix(y_test, test_predictions, labels=np.arange(10))
    cm_csv = args.output_dir / "baseline_dinov2_cifar10_confusion_matrix_counts.csv"
    with cm_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + CLASS_NAMES)
        for class_name, row in zip(CLASS_NAMES, cm):
            writer.writerow([class_name] + row.tolist())

    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        labels=np.arange(10),
        display_labels=CLASS_NAMES,
        normalize="true",
        values_format=".2f",
        xticks_rotation=45,
        cmap="Blues",
        colorbar=True,
        ax=ax,
    )
    ax.set_title("CIFAR-10 Test Set — DINOv2 ViT-S/14 + Logistic Regression\nRow-normalized confusion matrix")
    fig.tight_layout()
    confusion_png = args.output_dir / "baseline_dinov2_cifar10_confusion_matrix.png"
    fig.savefig(confusion_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - total_start
    print("\nSaved outputs:")
    for path in [cache_path, model_path, metrics_path, report_path,
                 validation_csv, predictions_path, cm_csv, confusion_png]:
        print(f"  {path}")
    print(f"\nTotal elapsed time (including extraction if needed): {elapsed / 60:.1f} minutes")
    print("\nBaseline completed. Do not tune the method using the official test results.")
    print("Reuse the saved feature cache for the upcoming prototype experiments.")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        print("\nCUDA OUT OF MEMORY: rerun with a smaller batch size, for example --batch-size 8.")
        print("No complete feature cache is written until all feature splits finish.")
        raise
