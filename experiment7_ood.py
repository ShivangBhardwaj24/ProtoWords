#!/usr/bin/env python3
"""Experiment 7: OOD detection for ProtoWords.

Implements the OOD plan described in the ProtoWords project roadmap:
- ID: CIFAR-10 validation/test
- Near-OOD: CIFAR-100 train-derived calibration subset / official test
- Far-OOD: SVHN train-derived calibration subset / official test
- Scores: maximum softmax probability (converted to OOD score), nearest
  class-prototype distance, shared-vocabulary distance, and patch-histogram
  distance.

All fitted classifiers and prototype vocabularies use CIFAR-10 TRAIN only.
CIFAR-10 validation and OOD calibration subsets are reserved for threshold
calibration. Final CIFAR-10 / OOD test sets are not used to select thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import Normalizer, StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


FEATURE_CACHE_NAME = "cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def parse_args():
    p = argparse.ArgumentParser(description="Run ProtoWords Experiment 7: OOD detection")
    p.add_argument("--project-root", type=str, default=None,
                   help="ProtoWords root; auto-detected if omitted")
    p.add_argument("--batch-size", type=int, default=16,
                   help="DINOv2 inference batch size (16 is a conservative RTX 4050 setting)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--ood-val-size", type=int, default=5000,
                   help="Number of OOD train images reserved for threshold calibration per source")
    p.add_argument("--k-global", type=int, default=128,
                   help="Shared global ProtoWords vocabulary size")
    p.add_argument("--k-per-class", type=int, default=8,
                   help="Number of class-specific prototypes per CIFAR-10 class")
    p.add_argument("--patch-k", type=int, default=256,
                   help="Patch vocabulary size from Experiment 5")
    p.add_argument("--ood-datasets", nargs="+", choices=["cifar100", "svhn"],
                   default=["cifar100", "svhn"])
    p.add_argument("--max-ood-test", type=int, default=0,
                   help="Optional debug limit per OOD test source; 0 means use full official test split")
    p.add_argument("--force-extract", action="store_true",
                   help="Recompute OOD feature caches even when cached files exist")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def find_project_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "outputs" / FEATURE_CACHE_NAME).is_file():
            raise FileNotFoundError(f"Feature cache not found under --project-root: {root / 'outputs' / FEATURE_CACHE_NAME}")
        return root

    here = Path(__file__).resolve().parent
    candidates = [here, *here.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "outputs" / FEATURE_CACHE_NAME).is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate the ProtoWords project root. Run this script inside ~/ProtoWords "
        "or provide --project-root /path/to/ProtoWords."
    )


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def l2_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def min_euclidean_distance(x: np.ndarray, centers: np.ndarray, chunk_size: int = 8192) -> np.ndarray:
    """Memory-safe distance to closest center; returns Euclidean (not squared) distance."""
    x = np.asarray(x, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    c2 = np.sum(centers * centers, axis=1)[None, :]
    out = np.empty(x.shape[0], dtype=np.float32)
    for start in range(0, x.shape[0], chunk_size):
        xb = x[start:start + chunk_size]
        d2 = np.sum(xb * xb, axis=1, keepdims=True) + c2 - 2.0 * xb @ centers.T
        np.maximum(d2, 0.0, out=d2)
        out[start:start + len(xb)] = np.sqrt(d2.min(axis=1))
    return out


def load_cifar10_cache(root: Path):
    path = root / "outputs" / FEATURE_CACHE_NAME
    z = np.load(path)
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"Feature cache is missing keys: {missing}")
    data = {k: z[k] for k in required}
    expected = {"X_train": (45000, 384), "X_val": (5000, 384), "X_test": (10000, 384)}
    for key, shape in expected.items():
        if data[key].shape != shape:
            raise ValueError(f"Unexpected {key} shape {data[key].shape}; expected {shape}")
    for key in ["X_train", "X_val", "X_test"]:
        data[key] = l2_rows(data[key])
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Non-finite feature values found in {key}")
    return data


def make_transform():
    # Same 224px bicubic resize and ImageNet normalization used in previous experiments.
    return transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def extract_features_and_patch_hist(
    model: torch.nn.Module,
    dataset,
    indices: np.ndarray,
    cache_path: Path,
    patch_centers: np.ndarray,
    device: str,
    batch_size: int,
    num_workers: int,
    force: bool = False,
) -> Dict[str, np.ndarray]:
    if cache_path.is_file() and not force:
        cached = np.load(cache_path)
        if "features" in cached.files and "patch_hist" in cached.files:
            print(f"Using cached OOD features: {cache_path}")
            return {"features": cached["features"], "patch_hist": cached["patch_hist"]}

    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    centers_t = torch.as_tensor(patch_centers, dtype=torch.float32, device=device)
    center_norm = (centers_t ** 2).sum(dim=1)[None, :]

    feature_parts, histogram_parts = [], []
    t0 = time.time()
    model.eval()
    with torch.inference_mode():
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=(device == "cuda"))
            outputs = model.forward_features(images)
            global_features = F.normalize(outputs["x_norm_clstoken"].float(), dim=-1)
            patch_tokens = F.normalize(outputs["x_norm_patchtokens"].float(), dim=-1)

            b, n_patches, dim = patch_tokens.shape
            flat = patch_tokens.reshape(b * n_patches, dim)
            # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x.c. Since x is L2-normalized,
            # ||x||^2 is constant across centers and can be omitted for argmin.
            d2_without_xnorm = center_norm - 2.0 * (flat @ centers_t.T)
            assignments = d2_without_xnorm.argmin(dim=1).reshape(b, n_patches)
            hist = torch.zeros((b, patch_centers.shape[0]), dtype=torch.float32, device=device)
            hist.scatter_add_(1, assignments, torch.ones_like(assignments, dtype=torch.float32))

            feature_parts.append(global_features.cpu().numpy().astype(np.float32))
            histogram_parts.append(hist.cpu().numpy().astype(np.int16))
            if batch_idx % 100 == 0:
                print(f"  {cache_path.stem}: batch {batch_idx + 1}/{len(loader)} "
                      f"({time.time() - t0:.0f}s)")

    result = {
        "features": np.concatenate(feature_parts, axis=0),
        "patch_hist": np.concatenate(histogram_parts, axis=0),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result)
    print(f"Saved {cache_path} | features={result['features'].shape}, "
          f"patch_hist={result['patch_hist'].shape} | {time.time() - t0:.1f}s")
    return result


def build_ood_datasets(data_root: Path, transform, selected_sources):
    result = {}
    if "cifar100" in selected_sources:
        print("Preparing CIFAR-100 datasets (may download on first run)...")
        c100_train = datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=transform)
        c100_test = datasets.CIFAR100(root=str(data_root), train=False, download=True, transform=transform)
        result["cifar100"] = (c100_train, c100_test)
    if "svhn" in selected_sources:
        print("Preparing SVHN datasets (may download on first run)...")
        svhn_root = data_root / "svhn"
        svhn_train = datasets.SVHN(root=str(svhn_root), split="train", download=True, transform=transform)
        svhn_test = datasets.SVHN(root=str(svhn_root), split="test", download=True, transform=transform)
        result["svhn"] = (svhn_train, svhn_test)
    return result


def make_indices(n: int, count: int, seed: int) -> np.ndarray:
    if count <= 0 or count > n:
        raise ValueError(f"Requested calibration subset of {count} from dataset of size {n}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=count, replace=False)).astype(np.int64)


def fit_models(root: Path, data: Dict[str, np.ndarray], args, outdir: Path):
    """Fit on CIFAR-10 train only; keep validation available for calibration."""
    x_train, y_train = data["X_train"], data["y_train"].astype(np.int64)

    print("\nFitting continuous DINOv2 Logistic Regression (C=1.0)...")
    continuous_lr = LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    continuous_lr.fit(x_train, y_train)

    print(f"\nFitting shared global ProtoWords codebook (K={args.k_global}) on CIFAR-10 TRAIN only...")
    global_kmeans = KMeans(n_clusters=args.k_global, n_init=10, max_iter=300, random_state=args.seed)
    global_kmeans.fit(x_train)
    global_centers = global_kmeans.cluster_centers_.astype(np.float32)
    d_train = global_kmeans.transform(x_train).astype(np.float32)
    global_scaler = StandardScaler().fit(d_train)
    global_lr = LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    global_lr.fit(global_scaler.transform(d_train), y_train)

    print(f"\nFitting class-specific prototypes ({args.k_per_class} per class) on CIFAR-10 TRAIN only...")
    class_centers = []
    for cls in range(10):
        class_x = x_train[y_train == cls]
        if len(class_x) < args.k_per_class:
            raise ValueError(f"Class {cls} has only {len(class_x)} examples; cannot fit {args.k_per_class} prototypes")
        km = KMeans(n_clusters=args.k_per_class, n_init=10, max_iter=300, random_state=args.seed + cls)
        km.fit(class_x)
        class_centers.append(km.cluster_centers_.astype(np.float32))
    class_centers = np.concatenate(class_centers, axis=0)

    print(f"\nLoading Experiment 5 patch histograms (K={args.patch_k})...")
    e5_dir = root / "outputs" / "e5"
    h_train_path = e5_dir / f"hist_K{args.patch_k}_train.npy"
    labels_path = e5_dir / "labels_train.npy"
    centers_path = e5_dir / f"centroids_K{args.patch_k}.npy"
    for path in [h_train_path, labels_path, centers_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required Experiment 5 artifact not found: {path}")
    h_train = np.load(h_train_path)
    labels_e5 = np.load(labels_path).astype(np.int64)
    if not np.array_equal(labels_e5, y_train):
        raise ValueError("Experiment 5 train labels do not match the shared CIFAR-10 feature cache")
    if h_train.shape != (len(y_train), args.patch_k):
        raise ValueError(f"Unexpected patch histogram shape {h_train.shape}")
    patch_centers = np.load(centers_path).astype(np.float32)
    h_train_l2 = l2_rows(h_train)
    # E5 validation selected C=10 for K=256; preserve that selected setting.
    patch_lr_c = 10.0
    patch_lr = LogisticRegression(C=patch_lr_c, max_iter=2000, random_state=args.seed)
    patch_lr.fit(h_train_l2, y_train)

    # Class-conditioned mean histogram references, based on training images only.
    patch_class_centers = []
    for cls in range(10):
        mean_hist = h_train_l2[y_train == cls].mean(axis=0, dtype=np.float64).astype(np.float32)
        mean_hist /= max(float(np.linalg.norm(mean_hist)), 1e-12)
        patch_class_centers.append(mean_hist)
    patch_class_centers = np.stack(patch_class_centers)

    models = {
        "continuous_lr": continuous_lr,
        "global_kmeans": global_kmeans,
        "global_lr": global_lr,
        "global_scaler": global_scaler,
        "global_centers": global_centers,
        "class_centers": class_centers,
        "patch_lr": patch_lr,
        "patch_class_centers": patch_class_centers,
        "patch_centers": patch_centers,
    }
    model_dir = outdir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(continuous_lr, model_dir / "continuous_dinov2_lr.joblib")
    joblib.dump(global_kmeans, model_dir / f"global_codebook_k{args.k_global}.joblib")
    joblib.dump(global_scaler, model_dir / "global_distance_scaler.joblib")
    joblib.dump(global_lr, model_dir / "global_protowords_lr.joblib")
    np.save(model_dir / f"class_prototypes_k{args.k_per_class}_per_class.npy", class_centers)
    joblib.dump(patch_lr, model_dir / f"patch_hist_lr_k{args.patch_k}.joblib")
    np.save(model_dir / f"patch_class_hist_prototypes_k{args.patch_k}.npy", patch_class_centers)
    np.save(model_dir / f"patch_codebook_k{args.patch_k}.npy", patch_centers)

    print("Training-time protocol: models/codebooks fit on 45,000 CIFAR-10 TRAIN images only.")
    print("CIFAR-10 validation and OOD calibration subsets remain held out for threshold calibration.")
    return models


def score_images(features: np.ndarray, patch_hist: np.ndarray, models: Dict[str, object]) -> Dict[str, np.ndarray]:
    x = l2_rows(features)
    h = l2_rows(patch_hist)

    p_cont = models["continuous_lr"].predict_proba(x)
    p_global = models["global_lr"].predict_proba(
        models["global_scaler"].transform(
            models["global_kmeans"].transform(x).astype(np.float32)
        )
    )
    p_patch = models["patch_lr"].predict_proba(h)

    return {
        # All returned scores point in the same direction: larger = more OOD-like.
        "msp_continuous_dinov2": (1.0 - p_cont.max(axis=1)).astype(np.float32),
        "msp_global_protowords": (1.0 - p_global.max(axis=1)).astype(np.float32),
        "msp_patch_protowords": (1.0 - p_patch.max(axis=1)).astype(np.float32),
        "nearest_class_prototype_distance": min_euclidean_distance(x, models["class_centers"]),
        "shared_vocabulary_distance": min_euclidean_distance(x, models["global_centers"]),
        "patch_hist_nearest_class_distance": min_euclidean_distance(h, models["patch_class_centers"]),
    }


def calibrate_and_evaluate(
    score_name: str,
    id_val: np.ndarray,
    ood_val: np.ndarray,
    id_test: np.ndarray,
    ood_test: np.ndarray,
) -> Dict[str, float]:
    # OOD is the positive class; larger score means more OOD-like.
    # q=0.05 on OOD calibration scores gives a validation operating threshold
    # targeting 95% OOD TPR. This threshold is frozen before test evaluation.
    threshold = float(np.quantile(ood_val, 0.05))
    id_val_fpr = float(np.mean(id_val >= threshold))
    ood_val_tpr = float(np.mean(ood_val >= threshold))
    id_test_fpr = float(np.mean(id_test >= threshold))
    ood_test_tpr = float(np.mean(ood_test >= threshold))

    y_true = np.concatenate([np.zeros(len(id_test), dtype=np.int8), np.ones(len(ood_test), dtype=np.int8)])
    y_score = np.concatenate([id_test, ood_test]).astype(np.float64)
    return {
        "score": score_name,
        "auroc": float(roc_auc_score(y_true, y_score)),
        "aupr_ood": float(average_precision_score(y_true, y_score)),
        "ood_positive_class": 1,
        "higher_score_means": "OOD",
        "threshold_from_ood_validation_5th_percentile": threshold,
        "validation_id_fpr_at_threshold": id_val_fpr,
        "validation_ood_tpr_at_threshold": ood_val_tpr,
        "test_id_fpr_at_validation_calibrated_threshold": id_test_fpr,
        "test_ood_tpr_at_validation_calibrated_threshold": ood_test_tpr,
    }


def save_roc_plot(source_name: str, id_scores: Dict[str, np.ndarray], ood_scores: Dict[str, np.ndarray], path: Path):
    plt.figure(figsize=(8, 6))
    for name in id_scores:
        y_true = np.concatenate([np.zeros(len(id_scores[name])), np.ones(len(ood_scores[name]))])
        y_score = np.concatenate([id_scores[name], ood_scores[name]])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate (ID incorrectly flagged as OOD)")
    plt.ylabel("True Positive Rate (OOD correctly flagged)")
    plt.title(f"OOD ROC — {source_name.upper()} vs CIFAR-10")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def feature_cache_is_valid(path: Path, expected_n: int, patch_k: int) -> bool:
    if not path.is_file():
        return False
    try:
        z = np.load(path)
        return (
            "features" in z.files
            and "patch_hist" in z.files
            and z["features"].shape == (expected_n, 384)
            and z["patch_hist"].shape == (expected_n, patch_k)
        )
    except Exception:
        return False


def main():
    args = parse_args()
    set_seeds(args.seed)
    root = find_project_root(args.project_root)
    outdir = root / "outputs" / "e7_ood"
    feature_dir = outdir / "features"
    outdir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("EXPERIMENT 7: PROTOWORDS OUT-OF-DISTRIBUTION DETECTION")
    print("=" * 76)
    print("project root:", root)
    print("output directory:", outdir)
    device = choose_device()
    print("device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("OOD sources:", ", ".join(args.ood_datasets))

    t_start = time.time()
    data = load_cifar10_cache(root)
    print("CIFAR-10 cache:", data["X_train"].shape, data["X_val"].shape, data["X_test"].shape)

    models = fit_models(root, data, args, outdir)

    # Existing CIFAR-10 patch histograms generated by Experiment 5.
    e5_dir = root / "outputs" / "e5"
    id_hist = {
        "val": np.load(e5_dir / f"hist_K{args.patch_k}_val.npy"),
        "test": np.load(e5_dir / f"hist_K{args.patch_k}_test.npy"),
    }
    for split in ["val", "test"]:
        expected_labels = data[f"y_{split}"].astype(np.int64)
        label_path = e5_dir / f"labels_{split}.npy"
        if label_path.is_file() and not np.array_equal(np.load(label_path), expected_labels):
            raise ValueError(f"Experiment 5 labels_{split} do not align with CIFAR-10 feature cache")

    id_scores = {
        "val": score_images(data["X_val"], id_hist["val"], models),
        "test": score_images(data["X_test"], id_hist["test"], models),
    }

    # If no OOD feature cache exists, download datasets and extract all required
    # OOD validation/test features in one DINOv2 pass per split.
    required_cache_paths = {}
    for source in args.ood_datasets:
        required_cache_paths[source] = {
            "val": feature_dir / f"{source}_ood_validation_patchK{args.patch_k}.npz",
            "test": feature_dir / f"{source}_official_test_patchK{args.patch_k}.npz",
        }
    official_test_sizes = {"cifar100": 10000, "svhn": 26032}
    cache_complete = not args.force_extract
    for source, split_paths in required_cache_paths.items():
        expected_test_n = official_test_sizes[source] if args.max_ood_test <= 0 else min(args.max_ood_test, official_test_sizes[source])
        if not feature_cache_is_valid(split_paths["val"], args.ood_val_size, args.patch_k):
            cache_complete = False
        if not feature_cache_is_valid(split_paths["test"], expected_test_n, args.patch_k):
            cache_complete = False

    if not cache_complete:
        data_root = root / "data" / "ood_datasets"
        data_root.mkdir(parents=True, exist_ok=True)
        datasets_by_name = build_ood_datasets(data_root, make_transform(), args.ood_datasets)
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        patch_centers = models["patch_centers"]
        for source in args.ood_datasets:
            train_ds, test_ds = datasets_by_name[source]
            val_indices = make_indices(len(train_ds), args.ood_val_size, args.seed + (1 if source == "cifar100" else 2))
            test_count = len(test_ds) if args.max_ood_test <= 0 else min(args.max_ood_test, len(test_ds))
            test_indices = np.arange(test_count, dtype=np.int64)
            np.save(feature_dir / f"{source}_ood_validation_indices.npy", val_indices)
            np.save(feature_dir / f"{source}_official_test_indices.npy", test_indices)
            extract_features_and_patch_hist(
                model, train_ds, val_indices, required_cache_paths[source]["val"], patch_centers,
                device, args.batch_size, args.num_workers, args.force_extract,
            )
            extract_features_and_patch_hist(
                model, test_ds, test_indices, required_cache_paths[source]["test"], patch_centers,
                device, args.batch_size, args.num_workers, args.force_extract,
            )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    metric_rows, calibration_rows = [], []
    per_source_scores = {}
    for source in args.ood_datasets:
        val_cache = np.load(required_cache_paths[source]["val"])
        test_cache = np.load(required_cache_paths[source]["test"])
        ood_val_scores = score_images(val_cache["features"], val_cache["patch_hist"], models)
        ood_test_scores = score_images(test_cache["features"], test_cache["patch_hist"], models)
        per_source_scores[source] = {"val": ood_val_scores, "test": ood_test_scores}

        for score_name in id_scores["val"]:
            row = calibrate_and_evaluate(
                score_name,
                id_scores["val"][score_name],
                ood_val_scores[score_name],
                id_scores["test"][score_name],
                ood_test_scores[score_name],
            )
            row["ood_dataset"] = source
            metric_rows.append(row)
            calibration_rows.append({
                "ood_dataset": source,
                "score": score_name,
                "threshold": row["threshold_from_ood_validation_5th_percentile"],
                "validation_id_fpr": row["validation_id_fpr_at_threshold"],
                "validation_ood_tpr": row["validation_ood_tpr_at_threshold"],
                "test_id_fpr": row["test_id_fpr_at_validation_calibrated_threshold"],
                "test_ood_tpr": row["test_ood_tpr_at_validation_calibrated_threshold"],
            })

        save_roc_plot(
            source,
            id_scores["test"],
            ood_test_scores,
            outdir / f"roc_{source}.png",
        )

        # Store per-image scores so disagreements and score distributions can be examined later.
        score_frame_id = pd.DataFrame({"split": "cifar10_id_test", **id_scores["test"]})
        score_frame_ood = pd.DataFrame({"split": f"{source}_ood_test", **ood_test_scores})
        pd.concat([score_frame_id, score_frame_ood], ignore_index=True).to_csv(
            outdir / f"scores_{source}.csv", index=False
        )

    metrics_df = pd.DataFrame(metric_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    metrics_df.to_csv(outdir / "ood_metrics.csv", index=False)
    calibration_df.to_csv(outdir / "threshold_calibration.csv", index=False)

    summary = {
        "experiment": "E7_OOD_detection",
        "id_dataset": "CIFAR-10",
        "ood_datasets": args.ood_datasets,
        "feature_extractor": "frozen DINOv2 ViT-S/14",
        "feature_dimension": 384,
        "input_resolution": 224,
        "training_protocol": "all fitted classifiers and prototype codebooks use CIFAR-10 train split only (45,000 images)",
        "id_validation": "CIFAR-10 validation split (5,000 images), reserved for ID threshold calibration checks",
        "ood_validation": f"random {args.ood_val_size}-image subsets from CIFAR-100 train / SVHN train, seed-controlled, used for threshold calibration",
        "ood_test": "official CIFAR-100 test and SVHN test splits; not used for model or threshold selection",
        "scores": {
            "msp_continuous_dinov2": "1 - max softmax probability from continuous DINOv2 Logistic Regression; higher means OOD",
            "msp_global_protowords": "1 - max softmax probability from Logistic Regression on standardized distances to shared global prototypes; higher means OOD",
            "msp_patch_protowords": "1 - max softmax probability from Logistic Regression on L2-normalized patch histograms; higher means OOD",
            "nearest_class_prototype_distance": f"minimum Euclidean distance to any of {args.k_per_class} class-specific prototypes per known CIFAR-10 class; higher means OOD",
            "shared_vocabulary_distance": f"minimum Euclidean distance to any of {args.k_global} shared global K-Means centroids; higher means OOD; not class-semantic by itself",
            "patch_hist_nearest_class_distance": f"minimum Euclidean distance to one of ten mean class histograms built from K={args.patch_k} patch ProtoWords; higher means OOD",
        },
        "metric_conventions": {
            "positive_class": "OOD=1; ID=0",
            "score_direction": "higher score means more OOD-like",
            "auroc": "computed from held-out ID test and OOD official test scores",
            "aupr_ood": "average precision with OOD as the positive class",
            "fpr_reporting": "test ID rejection rate at a threshold set to the 5th percentile of OOD validation scores (targeting 95% OOD TPR); report test OOD TPR at that same frozen threshold",
            "test_threshold_leakage": False,
        },
        "hyperparameters": {
            "seed": args.seed,
            "global_vocabulary_K": args.k_global,
            "class_prototypes_per_class": args.k_per_class,
            "patch_vocabulary_K": args.patch_k,
            "continuous_lr_C": 1.0,
            "global_protowords_lr_C": 1.0,
            "patch_hist_lr_C_selected_in_E5_validation": 10.0,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "results": metric_rows,
        "runtime_seconds": time.time() - t_start,
    }
    with open(outdir / "ood_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 76)
    print("EXPERIMENT 7 RESULTS")
    print("=" * 76)
    cols = ["ood_dataset", "score", "auroc", "aupr_ood",
            "test_id_fpr_at_validation_calibrated_threshold",
            "test_ood_tpr_at_validation_calibrated_threshold"]
    print(metrics_df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nOutputs saved to:", outdir)
    print(f"Total runtime: {time.time() - t_start:.1f}s")
    print("\nInterpretation reminder: higher AUROC/AUPR-OOD and lower ID FPR are better.")
    print("The validation-calibrated threshold is frozen before final test scoring.")


if __name__ == "__main__":
    main()
