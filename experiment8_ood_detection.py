from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from scipy.special import logsumexp
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "outputs").is_dir() else SCRIPT_DIR.parent
EXPERIMENT = "exp8_ood_detection"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
FEATURE_DIM = 384
MODEL_PATH = Path("outputs/exp6_hybrid_final_evaluation/final_models.joblib")
FEATURE_CACHE = Path(
    "outputs/exp1_baseline/"
    "cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz"
)
TINY_IMAGENET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
OOD_METHODS = [
    "continuous_msp",
    "continuous_energy",
    "distance_confidence",
    "soft_confidence",
    "hybrid_confidence",
    "nearest_codebook_distance",
]


def parse_args():
    parser = argparse.ArgumentParser(description="ProtoWords Experiment 8: OOD detection")
    parser.add_argument("--models", type=Path, default=MODEL_PATH)
    parser.add_argument("--feature-cache", type=Path, default=FEATURE_CACHE)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ood_datasets"))
    parser.add_argument("--tiny-imagenet-dir", type=Path, default=Path("data/tiny-imagenet-200"))
    parser.add_argument(
        "--ood-datasets", nargs="+", choices=["cifar100", "svhn", "tinyimagenet"],
        default=["cifar100", "svhn", "tinyimagenet"],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    for field in ("models", "feature_cache", "data_dir", "tiny_imagenet_dir"):
        value = getattr(args, field)
        if not value.is_absolute():
            setattr(args, field, (ROOT / value).resolve())
        else:
            setattr(args, field, value.resolve())

    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers cannot be negative")
    if not args.ood_datasets:
        parser.error("Select at least one OOD dataset")

    args.results_dir = ROOT / "results" / EXPERIMENT
    args.outputs_dir = ROOT / "outputs" / EXPERIMENT
    args.feature_dir = args.outputs_dir / "features"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    return args


def choose_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def l2_normalize(X):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def load_e6_bundle(path):
    if not path.is_file():
        raise FileNotFoundError(f"E6 model bundle not found: {path}")
    bundle = joblib.load(path)
    required = {
        "codebook_centers", "distance_scaler", "models", "K", "m",
        "temperature", "soft_weight", "selected_configs",
    }
    missing = required.difference(bundle)
    if missing:
        raise KeyError(f"E6 model bundle is missing: {sorted(missing)}")

    expected_models = {"continuous_dinov2_lr", "distance", "soft", "hybrid"}
    if expected_models.difference(bundle["models"]):
        raise KeyError("E6 bundle does not contain all four trained classifiers")
    centers = np.asarray(bundle["codebook_centers"], dtype=np.float32)
    if centers.shape != (int(bundle["K"]), FEATURE_DIM):
        raise ValueError(f"Unexpected codebook shape: {centers.shape}")
    if int(bundle["m"]) > int(bundle["K"]):
        raise ValueError("E6 bundle has m greater than K")
    return bundle


def load_id_features(path, seed):
    if not path.is_file():
        raise FileNotFoundError(f"E1 feature cache not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if not {"X_test", "y_test"}.issubset(data.files):
            raise KeyError("E1 feature cache must contain X_test and y_test")
        X = np.asarray(data["X_test"], dtype=np.float32)
        y = np.asarray(data["y_test"], dtype=np.int64)

    if X.shape != (10000, FEATURE_DIM) or y.shape != (10000,):
        raise ValueError(f"Unexpected CIFAR-10 test cache shapes: {X.shape}, {y.shape}")
    if not np.isfinite(X).all() or not np.isin(y, np.arange(10)).all():
        raise ValueError("Invalid CIFAR-10 test features or labels")

    indices = np.arange(len(y))
    calibration_idx, test_idx = train_test_split(
        indices, test_size=0.5, random_state=seed, stratify=y
    )
    X = l2_normalize(X)
    return X[calibration_idx], X[test_idx], y[calibration_idx], y[test_idx]


def make_transform():
    return transforms.Compose([
        transforms.Resize(
            (224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ),
    ])


class TinyImageNetImages(Dataset):
    def __init__(self, root, transform):
        images_dir = root / "val" / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Tiny ImageNet validation images not found: {images_dir}"
            )
        self.paths = sorted(
            path for path in images_dir.iterdir()
            if path.suffix.lower() in {".jpeg", ".jpg", ".png"}
        )
        if len(self.paths) != 10000:
            raise ValueError(
                f"Expected 10,000 Tiny ImageNet validation images, found {len(self.paths)}"
            )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return self.transform(image), -1


def download_tiny_imagenet(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "tiny-imagenet-200.zip"
    partial = archive.with_suffix(".zip.part")
    print("Downloading Tiny ImageNet from Stanford (approximately 250 MB)")
    try:
        request = urllib.request.Request(TINY_IMAGENET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, partial.open("wb") as file:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                if downloaded // (32 * 1024 * 1024) != (downloaded - len(chunk)) // (32 * 1024 * 1024):
                    if total:
                        print(f"  downloaded {downloaded / total:.0%}")
                    else:
                        print(f"  downloaded {downloaded / (1024 ** 2):.0f} MB")
        partial.replace(archive)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    print("Extracting Tiny ImageNet")
    with zipfile.ZipFile(archive) as file:
        file.extractall(data_dir)
    if not (data_dir / "tiny-imagenet-200" / "val" / "images").is_dir():
        raise RuntimeError("Tiny ImageNet archive extracted, but val/images was not found")


def build_ood_datasets(args):
    transform = make_transform()
    datasets_by_name = {}

    if "cifar100" in args.ood_datasets:
        datasets_by_name["cifar100"] = datasets.CIFAR100(
            root=str(args.data_dir), train=False, download=True, transform=transform
        )
    if "svhn" in args.ood_datasets:
        datasets_by_name["svhn"] = datasets.SVHN(
            root=str(args.data_dir), split="test", download=True, transform=transform
        )
    if "tinyimagenet" in args.ood_datasets:
        tiny_root = args.tiny_imagenet_dir
        if not (tiny_root / "val" / "images").is_dir():
            if tiny_root.name == "tiny-imagenet-200":
                download_root = tiny_root.parent
            else:
                download_root = tiny_root
            download_tiny_imagenet(download_root)
            if not (tiny_root / "val" / "images").is_dir() and (download_root / "tiny-imagenet-200" / "val" / "images").is_dir():
                tiny_root = download_root / "tiny-imagenet-200"
        datasets_by_name["tinyimagenet"] = TinyImageNetImages(tiny_root, transform)

    return datasets_by_name


def extract_features(model, dataset, device, batch_size, num_workers, name):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    parts = []
    model.eval()
    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=device.type == "cuda")
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model.forward_features(images)
            else:
                output = model.forward_features(images)
            if not isinstance(output, dict) or "x_norm_clstoken" not in output:
                raise RuntimeError("DINOv2 did not return x_norm_clstoken")
            part = output["x_norm_clstoken"].float().cpu().numpy()
            parts.append(part)
            if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(loader):
                print(f"  {name}: {batch_index + 1}/{len(loader)} batches")

    X = np.concatenate(parts).astype(np.float32, copy=False)
    if X.shape != (len(dataset), FEATURE_DIM) or not np.isfinite(X).all():
        raise ValueError(f"Invalid DINOv2 features for {name}: {X.shape}")
    return X


def load_or_extract_features(name, dataset, model, device, args):
    cache_path = args.feature_dir / f"{name}_features.npz"
    if cache_path.is_file() and not args.force_extract:
        with np.load(cache_path, allow_pickle=False) as cache:
            X = np.asarray(cache["features"], dtype=np.float32)
            metadata = json.loads(str(cache["metadata"].item()))
        if metadata == {"dataset": name, "count": len(dataset), "model": "dinov2_vits14", "image_size": 224}:
            if X.shape == (len(dataset), FEATURE_DIM) and np.isfinite(X).all():
                print(f"Using cached features: {cache_path}")
                return l2_normalize(X)
        print(f"Feature cache metadata did not match; extracting {name} again")

    print(f"Extracting DINOv2 features: {name} ({len(dataset)} images)")
    X = extract_features(model, dataset, device, args.batch_size, args.num_workers, name)
    metadata = {"dataset": name, "count": len(dataset), "model": "dinov2_vits14", "image_size": 224}
    temp_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez_compressed(temp_path, features=X, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    temp_path.replace(cache_path)
    print(f"Saved feature cache: {cache_path}")
    return l2_normalize(X)


def build_representation_scores(X, bundle, batch_size=2048):
    X = l2_normalize(X)
    centers = np.asarray(bundle["codebook_centers"], dtype=np.float32)
    center_norms = np.sum(centers * centers, axis=1)[None, :]
    K = int(bundle["K"])
    m = int(bundle["m"])
    temperature = float(bundle["temperature"])
    soft_weight = float(bundle["soft_weight"])
    models = bundle["models"]
    scaler = bundle["distance_scaler"]

    score_parts = {name: [] for name in OOD_METHODS}
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
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

        distances = np.sqrt(squared, dtype=np.float32)
        nearest_distance = np.sqrt(np.maximum(nearest_squared[:, 0], 0.0))
        batch_rows = np.repeat(np.arange(len(batch), dtype=np.int32), m)
        shifted = nearest_squared - nearest_squared[:, :1]
        weights = np.exp(-shifted / max(temperature, 1e-12)).astype(np.float32)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        soft = csr_matrix(
            (weights.reshape(-1), (batch_rows, nearest_ids.reshape(-1))),
            shape=(len(batch), K), dtype=np.float32,
        )
        if not np.allclose(np.asarray(soft.sum(axis=1)).ravel(), 1.0, atol=1e-5):
            raise RuntimeError("Soft-assignment weights do not sum to one per image")

        distance_scaled = scaler.transform(distances).astype(np.float32, copy=False)
        hybrid = np.hstack((distance_scaled, soft.toarray() * soft_weight))

        continuous_model = models["continuous_dinov2_lr"]
        continuous_probs = continuous_model.predict_proba(batch)
        continuous_logits = continuous_model.decision_function(batch)
        score_parts["continuous_msp"].append(1.0 - continuous_probs.max(axis=1))
        score_parts["continuous_energy"].append(-logsumexp(continuous_logits, axis=1))
        score_parts["distance_confidence"].append(1.0 - models["distance"].predict_proba(distance_scaled).max(axis=1))
        score_parts["soft_confidence"].append(1.0 - models["soft"].predict_proba(soft).max(axis=1))
        score_parts["hybrid_confidence"].append(1.0 - models["hybrid"].predict_proba(hybrid).max(axis=1))
        score_parts["nearest_codebook_distance"].append(nearest_distance)

    return {name: np.concatenate(parts).astype(np.float64) for name, parts in score_parts.items()}


def calibrate_threshold(calibration_scores, target_id_tpr=0.95):
    return float(np.quantile(calibration_scores, target_id_tpr, method="higher"))


def evaluate_detector(id_scores, ood_scores, threshold):
    y_true = np.concatenate((np.zeros(len(id_scores), dtype=np.int8), np.ones(len(ood_scores), dtype=np.int8)))
    scores = np.concatenate((id_scores, ood_scores))
    sample_weights = np.concatenate((
        np.full(len(id_scores), 0.5 / len(id_scores), dtype=np.float64),
        np.full(len(ood_scores), 0.5 / len(ood_scores), dtype=np.float64),
    ))
    id_retention = float(np.mean(id_scores <= threshold))
    ood_fpr = float(np.mean(ood_scores <= threshold))
    return {
        "auroc_ood_positive": float(roc_auc_score(y_true, scores)),
        "aupr_ood_positive_balanced": float(average_precision_score(y_true, scores, sample_weight=sample_weights)),
        "aupr_id_positive_balanced": float(average_precision_score(1 - y_true, -scores, sample_weight=sample_weights)),
        "id_test_retention_tpr": id_retention,
        "ood_false_positive_rate_at_95pct_id_calibration": ood_fpr,
        "ood_detection_rate_at_95pct_id_calibration": 1.0 - ood_fpr,
        "calibration_threshold": float(threshold),
    }


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    start_time = time.perf_counter()
    device = choose_device(args.cpu)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Device: {device}")
    print(f"OOD datasets: {args.ood_datasets}")
    print("Loading E6 model bundle")
    bundle = load_e6_bundle(args.models)
    print(
        f"E6 settings: K={bundle['K']}, m={bundle['m']}, "
        f"temperature={bundle['temperature']:.6g}, soft weight={bundle['soft_weight']}"
    )

    X_id_cal, X_id_test, _, _ = load_id_features(args.feature_cache, args.seed)
    print(f"ID calibration features: {X_id_cal.shape}")
    print(f"ID evaluation features:  {X_id_test.shape}")
    calibration_scores = build_representation_scores(X_id_cal, bundle)
    id_test_scores = build_representation_scores(X_id_test, bundle)

    thresholds = {}
    threshold_rows = []
    for method in OOD_METHODS:
        threshold = calibrate_threshold(calibration_scores[method])
        thresholds[method] = threshold
        threshold_rows.append({
            "method": method,
            "threshold": threshold,
            "target_id_calibration_tpr": 0.95,
            "observed_id_calibration_tpr": float(np.mean(calibration_scores[method] <= threshold)),
            "id_calibration_images": int(len(X_id_cal)),
        })

    datasets_by_name = build_ood_datasets(args)
    print("Loading frozen DINOv2 ViT-S/14")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    results_rows = []
    saved_scores = {}
    for dataset_name, dataset in datasets_by_name.items():
        X_ood = load_or_extract_features(dataset_name, dataset, model, device, args)
        scores = build_representation_scores(X_ood, bundle)
        saved_scores[f"{dataset_name}_labels_ood"] = np.ones(len(X_ood), dtype=np.int8)
        for method in OOD_METHODS:
            row_metrics = evaluate_detector(
                id_test_scores[method], scores[method], thresholds[method]
            )
            results_rows.append({
                "ood_dataset": dataset_name,
                "method": method,
                "id_calibration_images": int(len(X_id_cal)),
                "id_test_images": int(len(X_id_test)),
                "ood_test_images": int(len(X_ood)),
                **row_metrics,
            })
            saved_scores[f"{dataset_name}_{method}"] = scores[method]
        print(f"Completed {dataset_name}: {len(X_ood)} OOD images")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    overall_rows = []
    for method in OOD_METHODS:
        subset = [row for row in results_rows if row["method"] == method]
        overall_rows.append({
            "method": method,
            "ood_datasets": ";".join(row["ood_dataset"] for row in subset),
            "mean_auroc": float(np.mean([row["auroc_ood_positive"] for row in subset])),
            "mean_aupr_ood_balanced": float(np.mean([row["aupr_ood_positive_balanced"] for row in subset])),
            "mean_aupr_id_balanced": float(np.mean([row["aupr_id_positive_balanced"] for row in subset])),
            "mean_ood_fpr_at_95pct_id_calibration": float(np.mean([row["ood_false_positive_rate_at_95pct_id_calibration"] for row in subset])),
            "mean_id_test_retention_tpr": float(np.mean([row["id_test_retention_tpr"] for row in subset])),
        })

    save_csv(args.results_dir / "calibration_thresholds.csv", threshold_rows)
    save_csv(args.results_dir / "ood_results.csv", results_rows)
    save_csv(args.results_dir / "overall_results.csv", overall_rows)

    saved_scores["id_calibration_labels"] = np.zeros(len(X_id_cal), dtype=np.int8)
    saved_scores["id_test_labels"] = np.zeros(len(X_id_test), dtype=np.int8)
    for method in OOD_METHODS:
        saved_scores[f"id_calibration_{method}"] = calibration_scores[method]
        saved_scores[f"id_test_{method}"] = id_test_scores[method]
    np.savez_compressed(args.outputs_dir / "ood_scores.npz", **saved_scores)

    summary = {
        "experiment": "OOD detection with fixed E6 ProtoWords classifiers",
        "in_distribution": "CIFAR-10 official test split, stratified 50/50 calibration/evaluation split",
        "ood_datasets": {name: int(len(dataset)) for name, dataset in datasets_by_name.items()},
        "ood_methods": OOD_METHODS,
        "threshold_protocol": "Each threshold is calibrated using only the 5,000-image ID calibration half to retain 95% of ID examples. OOD examples are not used to choose thresholds.",
        "reported_fpr": "Fraction of OOD examples incorrectly accepted as ID at the ID-calibrated threshold.",
        "aupr_protocol": "AUPR is reported with equal total sample weight for ID and OOD classes.",
        "id_calibration_images": int(len(X_id_cal)),
        "id_evaluation_images": int(len(X_id_test)),
        "model_bundle": str(args.models),
        "feature_cache": str(args.feature_cache),
        "seed": int(args.seed),
        "device": str(device),
        "results": overall_rows,
        "runtime_seconds": round(time.perf_counter() - start_time, 2),
    }
    with (args.results_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nOverall OOD results (macro-average across datasets)")
    print("method                       AUROC    AUPR-OOD  OOD-FPR@95-ID")
    for row in overall_rows:
        print(
            f"{row['method']:<28} {row['mean_auroc']:.4f}   "
            f"{row['mean_aupr_ood_balanced']:.4f}     "
            f"{row['mean_ood_fpr_at_95pct_id_calibration']:.4f}"
        )
    print(f"Results: {args.results_dir}")
    print(f"Outputs: {args.outputs_dir}")
    print("Thresholds were calibrated on ID calibration data only.")


if __name__ == "__main__":
    main()
