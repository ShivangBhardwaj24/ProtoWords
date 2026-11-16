from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import Normalizer, StandardScaler
from threadpoolctl import threadpool_limits


EXPERIMENT = "exp13_quantized_final_test"
DEFAULT_CACHE = Path(
    "outputs/exp1_baseline/"
    "cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz"
)
K_VALUES = (128, 256)
C_BY_K = {128: 1.0, 256: 0.1}
FORMATS_BY_K = {128: ("uint8",), 256: ("uint8", "float16", "float32")}
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_args():
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "outputs" / "exp1_baseline").exists():
        root = script_dir
    elif (script_dir.parent / "outputs" / "exp1_baseline").exists():
        root = script_dir.parent
    else:
        root = Path.cwd().resolve()

    parser = argparse.ArgumentParser(
        description="Final test evaluation of the selected quantized ProtoWords models."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--clip-percentile", type=float, default=0.5)
    args = parser.parse_args()

    if args.n_init < 1 or args.max_iter < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("n-init, max-iter, batch-size and threads must be positive.")
    if not 0.0 <= args.clip_percentile < 50.0:
        parser.error("clip-percentile must be between 0 and 50.")

    args.root = root
    args.data = args.data.resolve() if args.data.is_absolute() else (root / args.data).resolve()
    args.results_dir = root / "results" / EXPERIMENT
    args.outputs_dir = root / "outputs" / EXPERIMENT
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    return args


def load_development_features(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {"X_train", "y_train", "X_val", "y_val"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Feature cache is missing: {sorted(missing)}")
        X_train = np.asarray(data["X_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=np.int64)
        X_val = np.asarray(data["X_val"], dtype=np.float32)
        y_val = np.asarray(data["y_val"], dtype=np.int64)

    if X_train.ndim != 2 or X_val.ndim != 2 or X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Training and validation features must be compatible 2D arrays.")
    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise ValueError("Feature and label counts do not match.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Development features contain NaN or infinite values.")
    if set(np.unique(y_train)) != set(range(10)) or set(np.unique(y_val)) != set(range(10)):
        raise ValueError("Training and validation labels must contain CIFAR-10 classes 0 through 9.")

    X_dev = np.concatenate([X_train, X_val], axis=0)
    y_dev = np.concatenate([y_train, y_val], axis=0)
    return X_dev, y_dev


def load_test_features(path: Path):
    with np.load(path, allow_pickle=False) as data:
        required = {"X_test", "y_test"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Feature cache is missing official test arrays: {sorted(missing)}")
        X_test = np.asarray(data["X_test"], dtype=np.float32)
        y_test = np.asarray(data["y_test"], dtype=np.int64)

    if X_test.ndim != 2 or len(X_test) != len(y_test):
        raise ValueError("Official test features and labels have incompatible shapes.")
    if not np.isfinite(X_test).all():
        raise ValueError("Official test features contain NaN or infinite values.")
    if X_test.shape[1] == 0 or set(np.unique(y_test)) != set(range(10)):
        raise ValueError("Official test features or labels are invalid.")
    return X_test, y_test


def distances_to_centers(X, centers, batch_size):
    center_norms = np.sum(centers * centers, axis=1)[None, :]
    parts = []
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        row_norms = np.sum(batch * batch, axis=1, keepdims=True)
        squared = row_norms + center_norms - 2.0 * (batch @ centers.T)
        np.maximum(squared, 0.0, out=squared)
        parts.append(np.sqrt(squared).astype(np.float32, copy=False))
    return np.vstack(parts).astype(np.float32, copy=False)


def make_uint8_parameters(X_train, clip_percentile):
    low = np.percentile(X_train, clip_percentile, axis=0).astype(np.float32)
    high = np.percentile(X_train, 100.0 - clip_percentile, axis=0).astype(np.float32)
    span = high - low
    span[span < 1e-8] = 1e-8
    scale = span / 255.0
    return {"low": low, "high": high, "scale": scale}


def encode_distances(standardized, fmt, quantization):
    if fmt == "float32":
        return standardized.astype(np.float32, copy=False), 0.0
    if fmt == "float16":
        return standardized.astype(np.float16).astype(np.float32), 0.0

    low = quantization["low"]
    high = quantization["high"]
    scale = quantization["scale"]
    clipped = np.minimum(np.maximum(standardized, low), high)
    quantized = np.rint((clipped - low) / scale).clip(0, 255).astype(np.uint8)
    reconstructed = low + quantized.astype(np.float32) * scale
    saturation = float(np.mean((standardized < low) | (standardized > high)))
    return reconstructed.astype(np.float32, copy=False), saturation


def parameter_bytes(model):
    total = 0
    for name in ("coef_", "intercept_", "classes_"):
        value = getattr(model, name, None)
        if isinstance(value, np.ndarray):
            total += value.nbytes
    return int(total)


def evaluate(y_true, y_pred):
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "test_micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }



def median_inference_time(model_bundle, X, batch_size):
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        if model_bundle["kind"] == "continuous":
            model_bundle["classifier"].predict(X)
        else:
            distances = distances_to_centers(X, model_bundle["centers"], batch_size)
            standardized = model_bundle["distance_scaler"].transform(distances).astype(
                np.float32, copy=False
            )
            encoded, _ = encode_distances(
                standardized, model_bundle["format"], model_bundle["quantization"]
            )
            model_bundle["classifier"].predict(encoded)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings) * 1000.0 / len(X))


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    start_time = time.perf_counter()

    X_dev_raw, y_dev = load_development_features(args.data)
    print(f"Development features: {X_dev_raw.shape}")
    print("Selected configurations: K=128 uint8; K=256 uint8, float16, float32")
    print("All codebooks, scalers and quantization parameters are fitted on 50,000 development images")

    normalizer = Normalizer(norm="l2")
    X_dev = normalizer.fit_transform(X_dev_raw).astype(np.float32, copy=False)
    del X_dev_raw

    models = {}
    model_info = {}
    baseline = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=args.seed)
    with threadpool_limits(limits=args.threads):
        baseline.fit(X_dev, y_dev)
    models["continuous_dinov2_lr"] = {
        "kind": "continuous",
        "classifier": baseline,
    }
    baseline_payload = X_dev.shape[1] * 4
    baseline_shared = parameter_bytes(baseline)
    baseline_total = baseline_payload + baseline_shared / len(X_dev)

    print(f"Fitted continuous DINOv2 baseline (C=1.0)")
    model_info["continuous_dinov2_lr"] = {
        "method": "continuous_dinov2_lr",
        "K": "",
        "format": "float32",
        "selected_C": 1.0,
        "representation_bytes_per_image": baseline_payload,
        "shared_parameter_bytes": baseline_shared,
        "amortized_total_bytes_per_image": baseline_total,
    }

    for k in K_VALUES:
        print(f"\nK={k}: fitting K-Means on development features")
        kmeans = KMeans(
            n_clusters=k,
            random_state=args.seed,
            n_init=args.n_init,
            max_iter=args.max_iter,
            algorithm="lloyd",
        )
        with threadpool_limits(limits=args.threads):
            kmeans.fit(X_dev)
        centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)

        raw_dev_distances = distances_to_centers(X_dev, centers, args.batch_size)
        scaler = StandardScaler()
        dev_standardized = scaler.fit_transform(raw_dev_distances).astype(np.float32, copy=False)
        del raw_dev_distances

        quantization = make_uint8_parameters(dev_standardized, args.clip_percentile)
        codebook_bytes = int(centers.nbytes)
        scaler_bytes = int(scaler.mean_.nbytes + scaler.scale_.nbytes)
        c_value = C_BY_K[k]

        for fmt in FORMATS_BY_K[k]:
            dev_encoded, _ = encode_distances(dev_standardized, fmt, quantization)
            classifier = LogisticRegression(
                C=c_value,
                solver="lbfgs",
                max_iter=2000,
                random_state=args.seed,
            )
            with threadpool_limits(limits=args.threads):
                classifier.fit(dev_encoded, y_dev)

            name = f"k{k}_{fmt}"
            quantization_bytes = 0 if fmt != "uint8" else int(
                quantization["low"].nbytes
                + quantization["high"].nbytes
                + quantization["scale"].nbytes
            )
            payload_bytes = k * {"float32": 4, "float16": 2, "uint8": 1}[fmt]
            classifier_bytes = parameter_bytes(classifier)
            shared_bytes = codebook_bytes + scaler_bytes + quantization_bytes + classifier_bytes
            total_per_image = payload_bytes + shared_bytes / len(X_dev)

            models[name] = {
                "kind": "distance",
                "K": k,
                "format": fmt,
                "centers": centers,
                "distance_scaler": scaler,
                "quantization": quantization if fmt == "uint8" else None,
                "classifier": classifier,
                "C": c_value,
            }
            model_info[name] = {
                "method": name,
                "K": k,
                "format": fmt,
                "selected_C": c_value,
                "representation_bytes_per_image": payload_bytes,
                "shared_parameter_bytes": shared_bytes,
                "amortized_total_bytes_per_image": total_per_image,
            }
            print(f"Fitted {name} (C={c_value:g})")

        del dev_standardized

    print("\nAll models fitted. Loading official test features now.")
    X_test_raw, y_test = load_test_features(args.data)
    if X_test_raw.shape[1] != X_dev.shape[1]:
        raise ValueError("Test and development feature dimensions do not match.")
    print(f"Official test features: {X_test_raw.shape}")
    X_test = normalizer.transform(X_test_raw).astype(np.float32, copy=False)
    del X_test_raw, X_dev

    rows = []
    prediction_rows = []
    reports = []
    matrices = {}
    storage_reduction = {}
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate(y_test, baseline_pred)

    ordered_names = ["continuous_dinov2_lr"]
    ordered_names.extend(
        f"k{k}_{fmt}" for k in K_VALUES for fmt in FORMATS_BY_K[k]
    )
    for name in ordered_names:
        model_bundle = models[name]
        if model_bundle["kind"] == "continuous":
            y_pred = baseline_pred
            saturation_pct = 0.0
            reconstruction_mae = 0.0
            reconstruction_rmse = 0.0
        else:
            centers = model_bundle["centers"]
            raw_test_distances = distances_to_centers(X_test, centers, args.batch_size)
            standardized = model_bundle["distance_scaler"].transform(raw_test_distances).astype(
                np.float32, copy=False
            )
            reconstructed, saturation = encode_distances(
                standardized, model_bundle["format"], model_bundle["quantization"]
            )
            difference = reconstructed - standardized
            reconstruction_mae = float(np.mean(np.abs(difference)))
            reconstruction_rmse = float(np.sqrt(np.mean(difference * difference)))
            saturation_pct = 100.0 * saturation
            y_pred = model_bundle["classifier"].predict(reconstructed)
            del raw_test_distances, standardized, reconstructed

        metrics = evaluate(y_test, y_pred)
        timing_count = min(1000, len(X_test))
        inference_ms = median_inference_time(model_bundle, X_test[:timing_count], args.batch_size)
        storage = model_info[name]
        reduction = 100.0 * (1.0 - storage["amortized_total_bytes_per_image"] / baseline_total)
        storage_reduction[name] = reduction
        row = {
            **storage,
            **metrics,
            "accuracy_difference_vs_continuous_pp": 100.0 * (
                metrics["test_accuracy"] - baseline_metrics["test_accuracy"]
            ),
            "storage_reduction_vs_continuous_pct": reduction,
            "test_saturation_pct": saturation_pct,
            "test_reconstruction_mae": reconstruction_mae,
            "test_reconstruction_rmse": reconstruction_rmse,
            "encode_predict_ms_per_image": inference_ms,
        }
        rows.append(row)
        matrices[name] = confusion_matrix(y_test, y_pred, labels=np.arange(10))
        reports.append(
            f"\n{'=' * 72}\n{name}\n{'=' * 72}\n"
            + classification_report(
                y_test, y_pred, labels=np.arange(10), target_names=CLASS_NAMES,
                digits=4, zero_division=0
            )
        )
        for index, (true_label, predicted_label) in enumerate(zip(y_test, y_pred)):
            prediction_rows.append({
                "image_index": index,
                "method": name,
                "true_label": int(true_label),
                "true_class": CLASS_NAMES[int(true_label)],
                "predicted_label": int(predicted_label),
                "predicted_class": CLASS_NAMES[int(predicted_label)],
                "correct": bool(true_label == predicted_label),
            })
        print(
            f"{name:<22} accuracy={metrics['test_accuracy']:.4f}, "
            f"Macro-F1={metrics['test_macro_f1']:.4f}, "
            f"storage={storage['amortized_total_bytes_per_image']:.2f} bytes/image, "
            f"encode+predict={inference_ms:.5f} ms/image"
        )

    write_csv(args.results_dir / "final_test_metrics.csv", rows)
    write_csv(args.results_dir / "final_test_predictions.csv", prediction_rows)
    with (args.results_dir / "final_test_classification_reports.txt").open("w", encoding="utf-8") as file:
        file.write("\n".join(reports))
    np.savez_compressed(args.results_dir / "final_test_confusion_matrices.npz", **matrices)

    summary = {
        "experiment": "Final test evaluation of quantized distance ProtoWords",
        "dataset": "CIFAR-10",
        "feature_extractor": "Frozen DINOv2 ViT-S/14 global CLS features",
        "development_images": int(len(y_dev)),
        "official_test_images": int(len(y_test)),
        "selection_protocol": "K and formats, C values, and quantization method were fixed from E12 validation results before test evaluation.",
        "test_used_for_selection": False,
        "timing_note": "Median of three CPU measurements on up to 1,000 test features; DINOv2 feature extraction is excluded.",
        "random_seed": args.seed,
        "kmeans_n_init": args.n_init,
        "kmeans_max_iter": args.max_iter,
        "uint8_quantization": {
            "method": "Per-dimension affine quantization of standardized distances",
            "clip_percentile_each_tail": args.clip_percentile,
            "parameters_fit_on": "All 50,000 development images only",
        },
        "baseline_test_accuracy": baseline_metrics["test_accuracy"],
        "baseline_test_macro_f1": baseline_metrics["test_macro_f1"],
        "results": [
            "final_test_metrics.csv",
            "final_test_predictions.csv",
            "final_test_classification_reports.txt",
            "final_test_confusion_matrices.npz",
        ],
        "runtime_seconds": round(time.perf_counter() - start_time, 3),
    }
    with (args.results_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    joblib.dump(
        {
            "normalizer": normalizer,
            "models": models,
            "storage_reference": model_info,
            "class_names": CLASS_NAMES,
            "selection_protocol": summary["selection_protocol"],
        },
        args.outputs_dir / "final_models.joblib",
        compress=3,
    )

    print("\nFinal test comparison")
    print("method                 accuracy   Macro-F1   bytes/image   storage reduction   ms/image")
    for row in rows:
        print(
            f"{row['method']:<22} {row['test_accuracy']:.4f}     "
            f"{row['test_macro_f1']:.4f}     "
            f"{row['amortized_total_bytes_per_image']:.2f}       "
            f"{row['storage_reduction_vs_continuous_pct']:.2f}%          "
            f"{row['encode_predict_ms_per_image']:.5f}"
        )
    print(f"Results: {args.results_dir}")
    print(f"Outputs: {args.outputs_dir}")
    print("E12 selections and quantization settings were fixed before test evaluation.")


if __name__ == "__main__":
    main()
