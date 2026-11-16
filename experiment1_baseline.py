from __future__ import annotations

import argparse
import csv
import json
import random
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "exp1_baseline"
MODEL_HUB = "facebookresearch/dinov2"
MODEL_NAME = "dinov2_vits14"
FEATURE_DIM = 384
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
DEFAULT_C_VALUES = [0.01, 0.1, 1.0, 10.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--c-values", type=float, nargs="+", default=DEFAULT_C_VALUES)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1 or args.num_workers < 0 or args.max_iter < 1:
        parser.error("batch size and max iterations must be positive; workers cannot be negative")
    if args.image_size < 14 or args.image_size % 14 != 0:
        parser.error("image size must be a positive multiple of 14")
    if not args.c_values or any(value <= 0 for value in args.c_values):
        parser.error("all C values must be positive")

    args.data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    args.outputs_dir = ROOT / "outputs" / EXPERIMENT
    args.results_dir = ROOT / "results" / EXPERIMENT
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_metadata(args):
    return {
        "dataset": "CIFAR-10",
        "model_hub": MODEL_HUB,
        "model_name": MODEL_NAME,
        "feature_type": "DINOv2 normalized CLS token returned by Hub model",
        "image_size": args.image_size,
        "normalization_mean": list(IMAGE_MEAN),
        "normalization_std": list(IMAGE_STD),
        "split_seed": args.seed,
        "split_sizes": {"train": 45000, "validation": 5000, "test": 10000},
    }


def extract_features(model, dataset, device, batch_size, num_workers, split_name):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    feature_parts = []
    label_parts = []
    start = time.time()

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=device.type == "cuda")
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = model(images)
            else:
                features = model(images)

            if isinstance(features, dict):
                if "x_norm_clstoken" not in features:
                    raise RuntimeError("DINOv2 output has no x_norm_clstoken feature")
                features = features["x_norm_clstoken"]

            if not isinstance(features, torch.Tensor) or features.ndim != 2:
                raise RuntimeError(f"Expected a 2-D feature tensor, got {type(features)}")

            feature_parts.append(features.float().cpu().numpy())
            label_parts.append(labels.numpy().astype(np.int64, copy=False))

            if batch_index % 100 == 0:
                print(f"{split_name}: {batch_index + 1}/{len(loader)} batches")

    X = np.concatenate(feature_parts).astype(np.float32, copy=False)
    y = np.concatenate(label_parts).astype(np.int64, copy=False)
    if X.shape != (len(dataset), FEATURE_DIM):
        raise RuntimeError(f"Unexpected {split_name} feature shape: {X.shape}")
    if not np.isfinite(X).all():
        raise RuntimeError(f"Non-finite values found in {split_name} features")

    print(f"{split_name}: {X.shape}, {time.time() - start:.1f}s")
    return X, y


def load_or_extract_cache(args, device, cache_path):
    metadata = get_metadata(args)

    if cache_path.exists() and not args.force_extract:
        with np.load(cache_path, allow_pickle=False) as cache:
            stored_metadata = json.loads(str(cache["metadata"].item()))
            if stored_metadata != metadata:
                raise RuntimeError("Feature cache settings differ; use --force-extract to rebuild it")
            X_train = cache["X_train"].astype(np.float32, copy=False)
            y_train = cache["y_train"].astype(np.int64, copy=False)
            X_val = cache["X_val"].astype(np.float32, copy=False)
            y_val = cache["y_val"].astype(np.int64, copy=False)
            X_test = cache["X_test"].astype(np.float32, copy=False)
            y_test = cache["y_test"].astype(np.int64, copy=False)
            train_indices = cache["train_indices"].astype(np.int64, copy=False)
            val_indices = cache["val_indices"].astype(np.int64, copy=False)
        print(f"Loaded feature cache: {cache_path}")
    else:
        transform = transforms.Compose([
            transforms.Resize(
                (args.image_size, args.image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ])
        train_full = datasets.CIFAR10(
            root=str(args.data_dir), train=True, transform=transform, download=True
        )
        test_full = datasets.CIFAR10(
            root=str(args.data_dir), train=False, transform=transform, download=True
        )
        if len(train_full) != 50000 or len(test_full) != 10000:
            raise RuntimeError("Unexpected CIFAR-10 dataset size")

        labels = np.asarray(train_full.targets, dtype=np.int64)
        indices = np.arange(len(train_full))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=5000,
            random_state=args.seed,
            shuffle=True,
            stratify=labels,
        )
        train_counts = np.bincount(labels[train_indices], minlength=10)
        val_counts = np.bincount(labels[val_indices], minlength=10)
        if not np.all(train_counts == 4500) or not np.all(val_counts == 500):
            raise RuntimeError("Unexpected class counts in train/validation split")

        train_subset = Subset(train_full, train_indices.tolist())
        val_subset = Subset(train_full, val_indices.tolist())

        print("Loading frozen DINOv2 ViT-S/14")
        model = torch.hub.load(MODEL_HUB, MODEL_NAME)
        model.eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        X_train, y_train = extract_features(
            model, train_subset, device, args.batch_size, args.num_workers, "train"
        )
        X_val, y_val = extract_features(
            model, val_subset, device, args.batch_size, args.num_workers, "validation"
        )
        X_test, y_test = extract_features(
            model, test_full, device, args.batch_size, args.num_workers, "test"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        temp_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
        np.savez_compressed(
            temp_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            train_indices=train_indices,
            val_indices=val_indices,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        temp_path.replace(cache_path)
        print(f"Saved feature cache: {cache_path}")

    expected_shapes = ((45000, FEATURE_DIM), (5000, FEATURE_DIM), (10000, FEATURE_DIM))
    actual_shapes = (X_train.shape, X_val.shape, X_test.shape)
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"Unexpected feature cache shapes: {actual_shapes}")
    if (len(y_train), len(y_val), len(y_test)) != (45000, 5000, 10000):
        raise RuntimeError("Unexpected label counts in feature cache")

    return X_train, y_train, X_val, y_val, X_test, y_test


def make_classifier(C, max_iter, seed):
    return Pipeline([
        ("normalizer", Normalizer(norm="l2")),
        ("logistic_regression", LogisticRegression(
            C=float(C), solver="lbfgs", max_iter=max_iter, random_state=seed
        )),
    ])


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.cpu)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    cache_path = args.outputs_dir / (
        f"cifar10_dinov2_vits14_{args.image_size}px_train45000_val5000_test10000_seed{args.seed}.npz"
    )
    model_path = args.outputs_dir / "dinov2_vits14_logreg_full.joblib"
    X_train, y_train, X_val, y_val, X_test, y_test = load_or_extract_cache(
        args, device, cache_path
    )

    validation_rows = []
    best_key = None
    best_C = None
    for C in args.c_values:
        model = make_classifier(C, args.max_iter, args.seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(y_val, predictions, average="macro", zero_division=0)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        validation_rows.append({
            "C": float(C),
            "validation_accuracy": float(accuracy),
            "validation_macro_f1": float(macro_f1),
            "converged": converged,
        })
        key = (accuracy, macro_f1, -float(C))
        if best_key is None or key > best_key:
            best_key = key
            best_C = float(C)

    with (args.results_dir / "validation_c_search.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=validation_rows[0].keys())
        writer.writeheader()
        writer.writerows(validation_rows)

    X_development = np.concatenate((X_train, X_val), axis=0)
    y_development = np.concatenate((y_train, y_val), axis=0)
    final_model = make_classifier(best_C, args.max_iter, args.seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        final_model.fit(X_development, y_development)
    final_converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    test_predictions = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_macro_f1 = f1_score(y_test, test_predictions, average="macro", zero_division=0)
    test_micro_f1 = f1_score(y_test, test_predictions, average="micro", zero_division=0)
    report = classification_report(
        y_test,
        test_predictions,
        labels=np.arange(10),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, test_predictions, labels=np.arange(10))

    joblib.dump(final_model, model_path)
    with (args.results_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump({
            "experiment": "DINOv2 ViT-S/14 frozen CLS features with logistic regression",
            "dataset": "CIFAR-10",
            "split": {"train_for_selection": 45000, "validation": 5000, "final_fit": 50000, "test": 10000},
            "seed": args.seed,
            "image_size": args.image_size,
            "feature_dimension": FEATURE_DIM,
            "selected_C": best_C,
            "test_accuracy": float(test_accuracy),
            "test_macro_f1": float(test_macro_f1),
            "test_micro_f1": float(test_micro_f1),
            "final_classifier_converged": final_converged,
            "feature_cache": str(cache_path.relative_to(ROOT)),
            "saved_model": str(model_path.relative_to(ROOT)),
        }, file, indent=2)

    (args.results_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    with (args.results_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["test_index", "true_label", "true_class", "predicted_label", "predicted_class"])
        for index, (true_label, predicted_label) in enumerate(zip(y_test, test_predictions)):
            writer.writerow([
                index, int(true_label), CLASS_NAMES[int(true_label)],
                int(predicted_label), CLASS_NAMES[int(predicted_label)],
            ])

    with (args.results_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + CLASS_NAMES)
        for name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([name] + row.tolist())

    print(f"Device: {device}")
    print(f"Selected C: {best_C}")
    print(f"Test accuracy: {test_accuracy:.4%}")
    print(f"Test macro-F1: {test_macro_f1:.4%}")
    print(f"Test micro-F1: {test_micro_f1:.4%}")
    print(f"Results: {args.results_dir}")
    print(f"Outputs: {args.outputs_dir}")


if __name__ == "__main__":
    main()
