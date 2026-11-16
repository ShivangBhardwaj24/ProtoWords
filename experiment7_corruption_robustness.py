from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from threadpoolctl import threadpool_limits


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "outputs").is_dir() else SCRIPT_DIR.parent
EXPERIMENT = "exp7_corruption_robustness"
MODEL_HUB = "facebookresearch/dinov2"
MODEL_NAME = "dinov2_vits14"
FEATURE_DIM = 384
N_CLASSES = 10
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
DEFAULT_CIFAR_C = Path("data/CIFAR-10-C")
DEFAULT_MODELS = Path("outputs/exp6_hybrid_final_evaluation/final_models.joblib")
DEFAULT_CLEAN_METRICS = Path("results/exp6_hybrid_final_evaluation/final_test_metrics.csv")


class CorruptedImages(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.fromarray(np.asarray(self.images[index]))
        return self.transform(image)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed E6 models on the 15 standard CIFAR-10-C corruptions."
    )
    parser.add_argument("--cifar10c-dir", type=Path, default=DEFAULT_CIFAR_C)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--clean-metrics", type=Path, default=DEFAULT_CLEAN_METRICS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--corruptions", nargs="+", default=CORRUPTIONS)
    args = parser.parse_args()

    for name in ("cifar10c_dir", "models", "clean_metrics"):
        value = getattr(args, name).expanduser()
        if not value.is_absolute():
            value = ROOT / value
        setattr(args, name, value.resolve())

    if args.batch_size < 1 or args.num_workers < 0 or args.threads < 1:
        parser.error("Batch size and threads must be positive; workers cannot be negative")

    unknown = sorted(set(args.corruptions).difference(CORRUPTIONS))
    if unknown:
        parser.error(f"Unknown or unsupported corruption names: {unknown}")
    if len(set(args.corruptions)) != len(args.corruptions):
        parser.error("Corruption names must not contain duplicates")

    args.results_dir = ROOT / "results" / EXPERIMENT
    args.outputs_dir = ROOT / "outputs" / EXPERIMENT
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    return args


def get_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def load_model_bundle(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"E6 model bundle not found: {path}\n"
            "Run Experiment 6 first and check outputs/exp6_hybrid_final_evaluation/."
        )
    bundle = joblib.load(path)
    required = {"K", "m", "temperature", "soft_weight", "codebook_centers", "distance_scaler", "models"}
    missing = required.difference(bundle)
    if missing:
        raise KeyError(f"E6 model bundle is missing: {sorted(missing)}")
    required_models = {"continuous_dinov2_lr", "distance", "soft", "hybrid"}
    missing_models = required_models.difference(bundle["models"])
    if missing_models:
        raise KeyError(f"E6 model bundle is missing models: {sorted(missing_models)}")

    centers = np.asarray(bundle["codebook_centers"], dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != FEATURE_DIM:
        raise ValueError(f"Unexpected codebook shape: {centers.shape}")
    if centers.shape[0] != int(bundle["K"]):
        raise ValueError("Codebook size does not match the saved K value")
    if not 1 <= int(bundle["m"]) <= int(bundle["K"]):
        raise ValueError("Invalid saved m value")
    if not np.isfinite(centers).all() or not np.isfinite(float(bundle["temperature"])):
        raise ValueError("E6 model bundle contains non-finite prototype settings")
    return bundle


def load_clean_metrics(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"E6 clean-test metrics not found: {path}\n"
            "Keep results/exp6_hybrid_final_evaluation/final_test_metrics.csv in the project."
        )
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"method", "test_accuracy", "test_macro_f1"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("E6 final_test_metrics.csv has an unexpected format")
    metrics = {}
    for row in rows:
        metrics[row["method"]] = {
            "accuracy": float(row["test_accuracy"]),
            "macro_f1": float(row["test_macro_f1"]),
        }
    expected = {"continuous_dinov2_lr", "distance", "soft", "hybrid"}
    if expected.difference(metrics):
        raise ValueError("E6 clean metrics do not contain all four methods")
    return metrics


def load_cifar10c_labels(cifar10c_dir):
    labels_path = cifar10c_dir / "labels.npy"
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"CIFAR-10-C labels not found: {labels_path}\n"
            "Extract CIFAR-10-C.tar so this folder contains labels.npy and the corruption .npy files."
        )
    labels = np.asarray(np.load(labels_path, allow_pickle=False), dtype=np.int64)
    if labels.shape == (50000,):
        label_blocks = labels.reshape(5, 10000)
        if not np.all(label_blocks == label_blocks[0]):
            raise ValueError("The five CIFAR-10-C label blocks do not match")
        labels = label_blocks[0]
    elif labels.shape != (10000,):
        raise ValueError(f"Unexpected CIFAR-10-C labels shape: {labels.shape}")

    if not np.isin(labels, np.arange(N_CLASSES)).all():
        raise ValueError("CIFAR-10-C labels must be integers from 0 to 9")
    return labels


def make_transform():
    return transforms.Compose([
        transforms.Resize(
            (224, 224),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])


def extract_features(model, images, transform, device, batch_size, num_workers, corruption, severity):
    dataset = CorruptedImages(images, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    parts = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=device.type == "cuda")
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = model(batch)
            else:
                features = model(batch)

            if isinstance(features, dict):
                if "x_norm_clstoken" not in features:
                    raise RuntimeError("DINOv2 output does not contain x_norm_clstoken")
                features = features["x_norm_clstoken"]
            if not isinstance(features, torch.Tensor) or features.ndim != 2:
                raise RuntimeError("Expected DINOv2 to return a 2-D feature tensor")
            parts.append(features.float().cpu().numpy())

            if (batch_index + 1) % 100 == 0:
                print(
                    f"  {corruption}, severity {severity}: "
                    f"{batch_index + 1}/{len(loader)} batches"
                )

    X = np.concatenate(parts).astype(np.float32, copy=False)
    if X.shape != (len(images), FEATURE_DIM) or not np.isfinite(X).all():
        raise ValueError(f"Invalid extracted feature array: {X.shape}")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def build_protowords_features(X, bundle, threads):
    centers = np.asarray(bundle["codebook_centers"], dtype=np.float32)
    k = centers.shape[0]
    m = int(bundle["m"])
    temperature = max(float(bundle["temperature"]), 1e-12)
    batch_size = 2048
    distances = np.empty((len(X), k), dtype=np.float32)
    soft = np.zeros((len(X), k), dtype=np.float32)
    center_norms = np.sum(centers * centers, axis=1)[None, :]

    with threadpool_limits(limits=threads):
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch = X[start:end]
            squared = (
                np.sum(batch * batch, axis=1, keepdims=True)
                + center_norms
                - 2.0 * (batch @ centers.T)
            )
            np.maximum(squared, 0.0, out=squared)

            nearest_ids = np.argpartition(squared, kth=m - 1, axis=1)[:, :m]
            nearest_squared = np.take_along_axis(squared, nearest_ids, axis=1)
            order = np.argsort(nearest_squared, axis=1)
            nearest_ids = np.take_along_axis(nearest_ids, order, axis=1)
            nearest_squared = np.take_along_axis(nearest_squared, order, axis=1)

            shifted = nearest_squared - nearest_squared[:, :1]
            weights = np.exp(-shifted / temperature).astype(np.float32)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            rows = np.arange(end - start)[:, None]
            soft[rows + start, nearest_ids] = weights
            if not np.allclose(soft[start:end].sum(axis=1), 1.0, atol=1e-5):
                raise RuntimeError("Soft-assignment rows must sum to 1")

            np.sqrt(squared, out=squared)
            distances[start:end] = squared

    distance_scaled = bundle["distance_scaler"].transform(distances).astype(np.float32, copy=False)
    hybrid = np.hstack((distance_scaled, soft * float(bundle["soft_weight"])))
    return distance_scaled, soft, hybrid


def evaluate_models(X, y, bundle, threads):
    distance, soft, hybrid = build_protowords_features(X, bundle, threads)
    representations = {
        "continuous_dinov2_lr": X,
        "distance": distance,
        "soft": soft,
        "hybrid": hybrid,
    }
    results = {}
    for name, values in representations.items():
        predictions = bundle["models"][name].predict(values)
        results[name] = {
            "accuracy": float(accuracy_score(y, predictions)),
            "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
            "correct_predictions": int(np.sum(predictions == y)),
        }
    return results


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    start_time = time.perf_counter()

    bundle = load_model_bundle(args.models)
    clean_metrics = load_clean_metrics(args.clean_metrics)
    labels = load_cifar10c_labels(args.cifar10c_dir)
    device = get_device(args.device)

    print(f"Device: {device}")
    print(f"CIFAR-10-C directory: {args.cifar10c_dir}")
    print(f"Corruptions: {len(args.corruptions)}")
    print("Loading frozen DINOv2 ViT-S/14")
    model = torch.hub.load(MODEL_HUB, MODEL_NAME)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    transform = make_transform()
    rows = []
    severity_rows = []
    start_labels = np.tile(labels, 5)

    for corruption in args.corruptions:
        path = args.cifar10c_dir / f"{corruption}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Corruption file not found: {path}")
        images = np.load(path, mmap_mode="r", allow_pickle=False)
        if images.shape != (50000, 32, 32, 3) or images.dtype != np.uint8:
            raise ValueError(f"Unexpected data in {path}: shape={images.shape}, dtype={images.dtype}")

        for severity in range(1, 6):
            begin = (severity - 1) * 10000
            finish = severity * 10000
            severity_images = images[begin:finish]
            y = start_labels[begin:finish]
            print(f"Extracting {corruption}, severity {severity}")
            X = extract_features(
                model, severity_images, transform, device,
                args.batch_size, args.num_workers, corruption, severity,
            )
            metrics = evaluate_models(X, y, bundle, args.threads)
            for method, values in metrics.items():
                rows.append({
                    "corruption": corruption,
                    "severity": severity,
                    "method": method,
                    "accuracy": values["accuracy"],
                    "macro_f1": values["macro_f1"],
                    "correct_predictions": values["correct_predictions"],
                    "images": len(y),
                })
            print(
                f"  completed: continuous={metrics['continuous_dinov2_lr']['accuracy']:.4f}, "
                f"distance={metrics['distance']['accuracy']:.4f}, "
                f"soft={metrics['soft']['accuracy']:.4f}, "
                f"hybrid={metrics['hybrid']['accuracy']:.4f}"
            )
            del X
        del images

    for method in ("continuous_dinov2_lr", "distance", "soft", "hybrid"):
        for severity in range(1, 6):
            selected = [r for r in rows if r["method"] == method and r["severity"] == severity]
            severity_rows.append({
                "method": method,
                "severity": severity,
                "mean_accuracy": float(np.mean([r["accuracy"] for r in selected])),
                "mean_macro_f1": float(np.mean([r["macro_f1"] for r in selected])),
                "corruptions_evaluated": len(selected),
            })

    overall_rows = []
    for method in ("continuous_dinov2_lr", "distance", "soft", "hybrid"):
        selected = [r for r in rows if r["method"] == method]
        mean_accuracy = float(np.mean([r["accuracy"] for r in selected]))
        mean_macro_f1 = float(np.mean([r["macro_f1"] for r in selected]))
        clean_accuracy = clean_metrics[method]["accuracy"]
        overall_rows.append({
            "method": method,
            "clean_test_accuracy": clean_accuracy,
            "mean_corruption_accuracy": mean_accuracy,
            "mean_corruption_macro_f1": mean_macro_f1,
            "accuracy_drop_from_clean_pp": 100.0 * (clean_accuracy - mean_accuracy),
            "worst_corruption_accuracy": float(min(r["accuracy"] for r in selected)),
            "corruption_conditions": len(selected),
        })

    write_csv(args.results_dir / "corruption_results.csv", rows)
    write_csv(args.results_dir / "severity_results.csv", severity_rows)
    write_csv(args.results_dir / "overall_results.csv", overall_rows)

    summary = {
        "experiment": "CIFAR-10-C corruption robustness of fixed E6 models",
        "dataset": "CIFAR-10-C",
        "benchmark_protocol": "15 standard corruption types, 5 severity levels, 10,000 images per condition",
        "corruptions": list(args.corruptions),
        "severity_levels": [1, 2, 3, 4, 5],
        "images_per_condition": 10000,
        "feature_extractor": "frozen DINOv2 ViT-S/14 CLS features",
        "feature_preprocessing": "224x224 bicubic resize, ImageNet mean/std normalization, L2 feature normalization",
        "model_source": str(args.models),
        "model_parameters": {
            "K": int(bundle["K"]),
            "m": int(bundle["m"]),
            "temperature": float(bundle["temperature"]),
            "soft_weight": float(bundle["soft_weight"]),
        },
        "clean_test_reference": clean_metrics,
        "overall_results": overall_rows,
        "device": str(device),
        "total_runtime_seconds": round(time.perf_counter() - start_time, 2),
        "official_test_usage": "Only CIFAR-10-C test corruptions were evaluated; no model fitting or hyperparameter selection was performed.",
    }
    with (args.results_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    model_reference = {
        "source_experiment": "exp6_hybrid_final_evaluation",
        "model_file": str(args.models),
        "K": int(bundle["K"]),
        "m": int(bundle["m"]),
        "temperature": float(bundle["temperature"]),
        "soft_weight": float(bundle["soft_weight"]),
        "corruptions": list(args.corruptions),
    }
    with (args.outputs_dir / "model_reference.json").open("w", encoding="utf-8") as file:
        json.dump(model_reference, file, indent=2)

    print("\nOverall corruption results")
    print("method                   mean accuracy   mean Macro-F1   drop from clean")
    for row in overall_rows:
        print(
            f"{row['method']:<24} {row['mean_corruption_accuracy']:.4f}          "
            f"{row['mean_corruption_macro_f1']:.4f}          "
            f"{row['accuracy_drop_from_clean_pp']:+.2f} pp"
        )
    print(f"Results: {args.results_dir}")
    print(f"Outputs: {args.outputs_dir}")
    print("The E6 models were used without retraining or tuning.")


if __name__ == "__main__":
    main()
