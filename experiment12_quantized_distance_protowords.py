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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import Normalizer, StandardScaler
from threadpoolctl import threadpool_limits


EXPERIMENT = "exp12_quantized_distance_protowords"
DEFAULT_CACHE = Path(
    "outputs/exp1_baseline/"
    "cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz"
)
CLASS_COUNT = 10
K_VALUES = [32, 64, 128, 256]
C_BY_K = {32: 100.0, 64: 100.0, 128: 1.0, 256: 0.1}
FORMATS = ("float32", "float16", "uint8")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "outputs" / "exp1_baseline").exists():
        root = script_dir
    elif (script_dir.parent / "outputs" / "exp1_baseline").exists():
        root = script_dir.parent
    else:
        root = Path.cwd().resolve()

    parser = argparse.ArgumentParser(description="Evaluate quantized distance ProtoWords.")
    parser.add_argument("--data", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--ks", type=int, nargs="+", default=K_VALUES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--clip-percentile", type=float, default=0.5)
    args = parser.parse_args()

    args.root = root
    args.data = args.data.resolve() if args.data.is_absolute() else (root / args.data).resolve()
    args.results_dir = root / "results" / EXPERIMENT
    args.outputs_dir = root / "outputs" / EXPERIMENT
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    args.ks = sorted(set(args.ks))
    if not args.ks or any(k not in K_VALUES for k in args.ks):
        parser.error("K values must be selected from 32, 64, 128, 256.")
    if args.n_init < 1 or args.max_iter < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("n-init, max-iter, batch-size and threads must be positive.")
    if not 0.0 <= args.clip_percentile < 50.0:
        parser.error("clip-percentile must be between 0 and 50.")
    return args


def load_features(path: Path):
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
        raise ValueError("Features contain NaN or infinite values.")
    if set(np.unique(y_train)) != set(range(CLASS_COUNT)):
        raise ValueError("Training labels must contain CIFAR-10 classes 0 through 9.")
    if set(np.unique(y_val)) != set(range(CLASS_COUNT)):
        raise ValueError("Validation labels must contain CIFAR-10 classes 0 through 9.")
    return X_train, y_train, X_val, y_val


def distances_to_centers(X, centers, batch_size):
    center_norms = np.sum(centers * centers, axis=1)[None, :]
    parts = []
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        row_norms = np.sum(batch * batch, axis=1, keepdims=True)
        squared = row_norms + center_norms - 2.0 * (batch @ centers.T)
        np.maximum(squared, 0.0, out=squared)
        parts.append(np.sqrt(squared, dtype=np.float32))
    return np.vstack(parts).astype(np.float32, copy=False)


def encode_distances(X, centers, scaler, quantization, fmt, batch_size):
    raw_distances = distances_to_centers(X, centers, batch_size)
    standardized = scaler.transform(raw_distances).astype(np.float32, copy=False)

    if fmt == "float32":
        reconstructed = standardized
        saturation_rate = 0.0
    elif fmt == "float16":
        reconstructed = standardized.astype(np.float16).astype(np.float32)
        saturation_rate = 0.0
    else:
        low = quantization["low"]
        high = quantization["high"]
        scale = quantization["scale"]
        clipped = np.minimum(np.maximum(standardized, low), high)
        quantized = np.rint((clipped - low) / scale).clip(0, 255).astype(np.uint8)
        reconstructed = low + quantized.astype(np.float32) * scale
        saturation_rate = float(np.mean((standardized < low) | (standardized > high)))

    return reconstructed.astype(np.float32, copy=False), standardized, saturation_rate


def make_uint8_parameters(X_train, clip_percentile):
    tail = float(clip_percentile)
    low = np.percentile(X_train, tail, axis=0).astype(np.float32)
    high = np.percentile(X_train, 100.0 - tail, axis=0).astype(np.float32)
    span = high - low
    span[span < 1e-8] = 1e-8
    scale = span / 255.0
    return {"low": low, "high": high, "scale": scale}


def reconstruction_metrics(reference, reconstructed):
    diff = reconstructed.astype(np.float32) - reference.astype(np.float32)
    return {
        "reconstruction_mae": float(np.mean(np.abs(diff))),
        "reconstruction_rmse": float(np.sqrt(np.mean(diff * diff))),
        "reconstruction_max_abs_error": float(np.max(np.abs(diff))),
    }


def score_model(y_true, y_pred):
    return {
        "validation_accuracy": float(accuracy_score(y_true, y_pred)),
        "validation_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "validation_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def median_pipeline_time(X, centers, scaler, quantization, fmt, classifier, batch_size):
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        encoded, _, _ = encode_distances(
            X, centers, scaler, quantization, fmt, batch_size
        )
        classifier.predict(encoded)
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
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def model_parameter_bytes(model):
    total = 0
    for name in ("coef_", "intercept_", "classes_"):
        value = getattr(model, name, None)
        if isinstance(value, np.ndarray):
            total += value.nbytes
    return int(total)


def main():
    args = parse_args()
    start_time = time.perf_counter()

    X_train_raw, y_train, X_val_raw, y_val = load_features(args.data)
    print(f"Training features: {X_train_raw.shape}")
    print(f"Validation features: {X_val_raw.shape}")
    print(f"Vocabulary sizes: {args.ks}")
    print("Formats: float32, float16, uint8")
    print("Codebooks and all quantization parameters are fitted on training data only")

    normalizer = Normalizer(norm="l2")
    X_train = normalizer.fit_transform(X_train_raw).astype(np.float32, copy=False)
    X_val = normalizer.transform(X_val_raw).astype(np.float32, copy=False)
    del X_train_raw, X_val_raw

    with threadpool_limits(limits=args.threads):
        baseline = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=args.seed)
        baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_val)
    baseline_metrics = score_model(y_val, baseline_pred)
    print(
        f"Continuous baseline: accuracy={baseline_metrics['validation_accuracy']:.4f}, "
        f"Macro-F1={baseline_metrics['validation_macro_f1']:.4f}"
    )

    result_rows = [{
        "K": "baseline",
        "format": "float32",
        "selected_C": 1.0,
        **baseline_metrics,
        "representation_bytes_per_image": X_train.shape[1] * 4,
        "shared_parameter_bytes": model_parameter_bytes(baseline),
        "amortized_total_bytes_per_image": float(
            X_train.shape[1] * 4 + model_parameter_bytes(baseline) / len(X_train)
        ),
        "storage_reduction_vs_continuous_pct": 0.0,
        "reconstruction_mae": 0.0,
        "reconstruction_rmse": 0.0,
        "reconstruction_max_abs_error": 0.0,
        "validation_saturation_pct": 0.0,
        "encode_predict_ms_per_image": "",
    }]
    quantization_rows = []
    codebooks = {}
    classifiers = {}

    baseline_total_bytes = float(result_rows[0]["amortized_total_bytes_per_image"])
    n_train = len(X_train)

    for k in args.ks:
        print(f"\nK={k}: fitting global K-Means codebook")
        kmeans = KMeans(
            n_clusters=k,
            random_state=args.seed,
            n_init=args.n_init,
            max_iter=args.max_iter,
            algorithm="lloyd",
        )
        with threadpool_limits(limits=args.threads):
            kmeans.fit(X_train)
        centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)

        train_distances = distances_to_centers(X_train, centers, args.batch_size)
        val_distances = distances_to_centers(X_val, centers, args.batch_size)
        scaler = StandardScaler()
        train_standardized = scaler.fit_transform(train_distances).astype(np.float32, copy=False)
        val_standardized = scaler.transform(val_distances).astype(np.float32, copy=False)
        del train_distances, val_distances

        quantization = make_uint8_parameters(train_standardized, args.clip_percentile)
        c_value = C_BY_K[k]
        codebooks[k] = {"centers": centers, "distance_scaler": scaler}

        codebook_bytes = int(centers.nbytes)
        scaler_bytes = int(scaler.mean_.nbytes + scaler.scale_.nbytes)

        for fmt in FORMATS:
            if fmt == "float32":
                train_encoded = train_standardized
                val_encoded = val_standardized
                qparams = None
                quantization_bytes = 0
                payload_bytes = k * 4
                saturation_rate = 0.0
            elif fmt == "float16":
                train_encoded = train_standardized.astype(np.float16).astype(np.float32)
                val_encoded = val_standardized.astype(np.float16).astype(np.float32)
                qparams = None
                quantization_bytes = 0
                payload_bytes = k * 2
                saturation_rate = 0.0
            else:
                qparams = quantization
                low, high, scale = qparams["low"], qparams["high"], qparams["scale"]
                train_clipped = np.minimum(np.maximum(train_standardized, low), high)
                val_clipped = np.minimum(np.maximum(val_standardized, low), high)
                train_q = np.rint((train_clipped - low) / scale).clip(0, 255).astype(np.uint8)
                val_q = np.rint((val_clipped - low) / scale).clip(0, 255).astype(np.uint8)
                train_encoded = (low + train_q.astype(np.float32) * scale).astype(np.float32)
                val_encoded = (low + val_q.astype(np.float32) * scale).astype(np.float32)
                quantization_bytes = int(low.nbytes + high.nbytes + scale.nbytes)
                payload_bytes = k
                saturation_rate = float(
                    np.mean((val_standardized < low) | (val_standardized > high))
                )

            errors = reconstruction_metrics(val_standardized, val_encoded)
            with threadpool_limits(limits=args.threads):
                classifier = LogisticRegression(
                    C=c_value,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=args.seed,
                )
                fit_start = time.perf_counter()
                classifier.fit(train_encoded, y_train)
                fit_seconds = time.perf_counter() - fit_start

            predictions = classifier.predict(val_encoded)
            metrics = score_model(y_val, predictions)
            classifier_bytes = model_parameter_bytes(classifier)
            shared_bytes = codebook_bytes + scaler_bytes + quantization_bytes + classifier_bytes
            amortized_total = payload_bytes + shared_bytes / n_train
            storage_reduction = 100.0 * (1.0 - amortized_total / baseline_total_bytes)
            inference_ms = median_pipeline_time(
                X_val[: min(len(X_val), 1000)], centers, scaler, qparams,
                fmt, classifier, args.batch_size
            )

            row = {
                "K": k,
                "format": fmt,
                "selected_C": c_value,
                **metrics,
                "representation_bytes_per_image": payload_bytes,
                "shared_parameter_bytes": shared_bytes,
                "amortized_total_bytes_per_image": round(amortized_total, 6),
                "storage_reduction_vs_continuous_pct": round(storage_reduction, 4),
                **errors,
                "validation_saturation_pct": round(100.0 * saturation_rate, 6),
                "encode_predict_ms_per_image": round(inference_ms, 6),
                "classifier_fit_seconds": round(fit_seconds, 4),
            }
            result_rows.append(row)
            quantization_rows.append({
                "K": k,
                "format": fmt,
                **errors,
                "validation_saturation_pct": round(100.0 * saturation_rate, 6),
                "representation_bytes_per_image": payload_bytes,
            })
            classifiers[f"k{k}_{fmt}"] = {
                "K": k,
                "format": fmt,
                "C": c_value,
                "classifier": classifier,
                "quantization": qparams,
                "validation_metrics": metrics,
            }
            print(
                f"  {fmt:<7} accuracy={metrics['validation_accuracy']:.4f}, "
                f"Macro-F1={metrics['validation_macro_f1']:.4f}, "
                f"storage={amortized_total:.2f} bytes/image"
            )

            del train_encoded, val_encoded

        del train_standardized, val_standardized

    proto_rows = [row for row in result_rows if row["K"] != "baseline"]
    best = max(proto_rows, key=lambda row: (row["validation_accuracy"], row["validation_macro_f1"]))
    best_small = min(
        (row for row in proto_rows if row["validation_accuracy"] >= baseline_metrics["validation_accuracy"] - 0.002),
        key=lambda row: row["amortized_total_bytes_per_image"],
        default=None,
    )

    write_csv(args.results_dir / "validation_results.csv", result_rows)
    write_csv(args.results_dir / "quantization_error.csv", quantization_rows)

    summary = {
        "experiment": "Quantized distance ProtoWords",
        "dataset": "CIFAR-10",
        "feature_extractor": "Frozen DINOv2 ViT-S/14 global CLS features",
        "training_images": int(len(y_train)),
        "validation_images": int(len(y_val)),
        "official_test_used": False,
        "vocabulary_sizes": args.ks,
        "formats": list(FORMATS),
        "quantization": {
            "float16": "Standardized distances rounded to IEEE float16 and restored to float32 for Logistic Regression.",
            "uint8": "Per-dimension affine quantization to uint8 using training-only lower and upper percentiles; validation values outside the range are clipped.",
            "clip_percentile_each_tail": args.clip_percentile,
        },
        "fixed_C_by_K": {str(k): C_BY_K[k] for k in args.ks},
        "baseline": baseline_metrics,
        "best_protowords_configuration": {
            "K": int(best["K"]),
            "format": best["format"],
            "validation_accuracy": best["validation_accuracy"],
            "validation_macro_f1": best["validation_macro_f1"],
            "amortized_total_bytes_per_image": best["amortized_total_bytes_per_image"],
        },
        "smallest_configuration_within_0.2pp_of_baseline": None if best_small is None else {
            "K": int(best_small["K"]),
            "format": best_small["format"],
            "validation_accuracy": best_small["validation_accuracy"],
            "amortized_total_bytes_per_image": best_small["amortized_total_bytes_per_image"],
        },
        "runtime_seconds": round(time.perf_counter() - start_time, 3),
        "results": ["validation_results.csv", "quantization_error.csv"],
    }
    with (args.results_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    joblib.dump(
        {
            "normalizer": normalizer,
            "baseline_classifier": baseline,
            "codebooks": codebooks,
            "distance_classifiers": classifiers,
            "format_description": "Classifier inputs are reconstructed standardized distances after the selected storage quantization.",
        },
        args.outputs_dir / "quantized_distance_models.joblib",
        compress=3,
    )

    print("\nValidation summary")
    print("K      format   accuracy   Macro-F1   bytes/image")
    for row in result_rows:
        print(
            f"{str(row['K']):<6} {row['format']:<8} "
            f"{row['validation_accuracy']:.4f}     {row['validation_macro_f1']:.4f}   "
            f"{row['amortized_total_bytes_per_image']:.2f}"
        )
    print(f"Best ProtoWords configuration: K={best['K']}, {best['format']}")
    print(f"Results: {args.results_dir}")
    print(f"Outputs: {args.outputs_dir}")
    print("The official CIFAR-10 test split was not loaded or evaluated.")


if __name__ == "__main__":
    main()
