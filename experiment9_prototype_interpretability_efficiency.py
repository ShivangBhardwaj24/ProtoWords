from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from threadpoolctl import threadpool_limits


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "outputs").is_dir() else SCRIPT_DIR.parent
EXPERIMENT = "exp9_prototype_interpretability_efficiency"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CACHE_PATH = Path(
    "outputs/exp1_baseline/"
    "cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz"
)
MODEL_PATH = Path("outputs/exp6_hybrid_final_evaluation/final_models.joblib")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the learned ProtoWords vocabulary and representation efficiency."
    )
    parser.add_argument("--data", type=Path, default=CACHE_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--cifar-root", type=Path, default=Path("data"))
    parser.add_argument("--representative-count", type=int, default=5)
    parser.add_argument("--timing-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    if args.representative_count < 1 or args.timing_samples < 1:
        parser.error("representative-count and timing-samples must be positive")
    if args.batch_size < 1 or args.threads < 1 or args.timing_repeats < 1:
        parser.error("batch-size, threads and timing-repeats must be positive")

    if not args.data.is_absolute():
        args.data = (ROOT / args.data).resolve()
    if not args.model.is_absolute():
        args.model = (ROOT / args.model).resolve()
    if not args.cifar_root.is_absolute():
        args.cifar_root = (ROOT / args.cifar_root).resolve()

    args.results_dir = ROOT / "results" / EXPERIMENT
    args.outputs_dir = ROOT / "outputs" / EXPERIMENT
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    return args


def normalize_features(X):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def load_inputs(data_path, model_path):
    if not data_path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {data_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"E6 model bundle not found: {model_path}")

    with np.load(data_path, allow_pickle=False) as cache:
        required = {"X_train", "y_train", "X_val", "y_val", "train_indices", "val_indices"}
        missing = required.difference(cache.files)
        if missing:
            raise KeyError(f"Feature cache is missing: {sorted(missing)}")
        X_train = np.asarray(cache["X_train"], dtype=np.float32)
        y_train = np.asarray(cache["y_train"], dtype=np.int64)
        X_val = np.asarray(cache["X_val"], dtype=np.float32)
        y_val = np.asarray(cache["y_val"], dtype=np.int64)
        train_indices = np.asarray(cache["train_indices"], dtype=np.int64)
        val_indices = np.asarray(cache["val_indices"], dtype=np.int64)

    X_dev = normalize_features(np.concatenate((X_train, X_val), axis=0))
    y_dev = np.concatenate((y_train, y_val), axis=0)
    original_indices = np.concatenate((train_indices, val_indices), axis=0)

    if X_dev.ndim != 2 or X_dev.shape[1] != 384:
        raise ValueError(f"Expected 384-dimensional DINOv2 features, got {X_dev.shape}")
    if len(X_dev) != len(y_dev) or len(y_dev) != len(original_indices):
        raise ValueError("Feature, label and original-index counts do not match")
    if len(np.unique(original_indices)) != len(original_indices):
        raise ValueError("Train and validation original indices overlap")
    if not np.isfinite(X_dev).all() or not np.isin(y_dev, np.arange(10)).all():
        raise ValueError("Feature cache contains invalid features or labels")

    bundle = joblib.load(model_path)
    required_bundle = {"K", "m", "temperature", "soft_weight", "codebook_centers", "distance_scaler", "models"}
    missing_bundle = required_bundle.difference(bundle)
    if missing_bundle:
        raise KeyError(f"E6 model bundle is missing: {sorted(missing_bundle)}")

    centers = np.asarray(bundle["codebook_centers"], dtype=np.float32)
    if centers.shape != (int(bundle["K"]), X_dev.shape[1]):
        raise ValueError(f"Unexpected codebook shape: {centers.shape}")
    if int(bundle["K"]) != 256 or int(bundle["m"]) != 8:
        print(f"Note: loaded E6 settings K={bundle['K']}, m={bundle['m']}")

    expected_models = {"continuous_dinov2_lr", "distance", "soft", "hybrid"}
    if not expected_models.issubset(bundle["models"]):
        raise KeyError("E6 model bundle does not contain all four expected classifiers")

    return X_dev, y_dev, original_indices, bundle


def assign_to_codebook(X, centers, batch_size, threads):
    assignments = np.empty(len(X), dtype=np.int32)
    nearest_distances = np.empty(len(X), dtype=np.float32)
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
            ids = np.argmin(squared, axis=1)
            assignments[start:end] = ids
            nearest_distances[start:end] = np.sqrt(squared[np.arange(len(batch)), ids])

    return assignments, nearest_distances


def prototype_statistics(assignments, distances, labels, k):
    class_counts = np.zeros((k, len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(class_counts, (assignments, labels), 1)
    cluster_sizes = class_counts.sum(axis=1)
    dominant_ids = np.argmax(class_counts, axis=1)
    purity = np.divide(
        class_counts.max(axis=1), cluster_sizes,
        out=np.zeros(k, dtype=np.float64), where=cluster_sizes > 0,
    )

    probabilities = np.divide(
        class_counts, cluster_sizes[:, None],
        out=np.zeros_like(class_counts, dtype=np.float64),
        where=cluster_sizes[:, None] > 0,
    )
    log_probabilities = np.zeros_like(probabilities)
    positive = probabilities > 0
    log_probabilities[positive] = np.log2(probabilities[positive])
    entropy = -(probabilities * log_probabilities).sum(axis=1)
    normalized_entropy = entropy / np.log2(len(CLASS_NAMES))

    rows = []
    for prototype_id in range(k):
        row = {
            "prototype_id": prototype_id,
            "cluster_size": int(cluster_sizes[prototype_id]),
            "dominant_class": CLASS_NAMES[int(dominant_ids[prototype_id])],
            "purity": float(purity[prototype_id]),
            "entropy_bits": float(entropy[prototype_id]),
            "normalized_entropy": float(normalized_entropy[prototype_id]),
            "mean_assignment_distance": float(distances[assignments == prototype_id].mean())
            if cluster_sizes[prototype_id] else "",
        }
        for class_id, class_name in enumerate(CLASS_NAMES):
            row[f"count_{class_name}"] = int(class_counts[prototype_id, class_id])
        rows.append(row)
    return rows, class_counts, cluster_sizes


def representative_examples(assignments, distances, labels, original_indices, count):
    rows = []
    k = int(assignments.max()) + 1 if len(assignments) else 0
    for prototype_id in range(k):
        members = np.flatnonzero(assignments == prototype_id)
        if not len(members):
            continue
        nearest = members[np.argsort(distances[members])[:count]]
        for rank, sample_pos in enumerate(nearest, start=1):
            rows.append({
                "prototype_id": prototype_id,
                "rank": rank,
                "original_train_index": int(original_indices[sample_pos]),
                "class_id": int(labels[sample_pos]),
                "class_name": CLASS_NAMES[int(labels[sample_pos])],
                "distance_to_prototype": float(distances[sample_pos]),
            })
    return rows


def save_contact_sheet(path, examples, cifar_root):
    try:
        from PIL import Image, ImageDraw, ImageFont
        from torchvision.datasets import CIFAR10
    except ImportError as exc:
        raise RuntimeError(
            "Image preview requires Pillow and torchvision. Install Pillow or run with --skip-images."
        ) from exc

    dataset = CIFAR10(root=str(cifar_root), train=True, download=True)
    raw_images = dataset.data
    raw_labels = np.asarray(dataset.targets, dtype=np.int64)

    if not examples:
        raise RuntimeError("No representative examples were generated")

    columns = 16
    rows = (len(examples) + columns - 1) // columns
    cell_w, cell_h = 112, 104
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for item in examples:
        original_index = item["original_train_index"]
        if original_index < 0 or original_index >= len(raw_images):
            raise IndexError(f"CIFAR-10 index out of range: {original_index}")
        if int(raw_labels[original_index]) != item["class_id"]:
            raise ValueError("Cached labels do not match the original CIFAR-10 training order")
        image = Image.fromarray(raw_images[original_index]).resize((64, 64))
        col = item["prototype_id"] % columns
        row = item["prototype_id"] // columns
        x = col * cell_w
        y = row * cell_h
        sheet.paste(image, (x + 4, y + 4))
        draw.text((x + 72, y + 7), f"P{item['prototype_id']}", fill="black", font=font)
        draw.text((x + 72, y + 23), item["class_name"][:8], fill="black", font=font)
        draw.text((x + 72, y + 39), f"rank {item['rank']}", fill="black", font=font)
        draw.text((x + 72, y + 55), f"d={item['distance_to_prototype']:.2f}", fill="black", font=font)
    sheet.save(path)


def squared_distances(X, centers, batch_size, threads):
    output = np.empty((len(X), len(centers)), dtype=np.float32)
    center_norms = np.sum(centers * centers, axis=1)[None, :]
    with threadpool_limits(limits=threads):
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch = X[start:end]
            values = (
                np.sum(batch * batch, axis=1, keepdims=True)
                + center_norms
                - 2.0 * (batch @ centers.T)
            )
            np.maximum(values, 0.0, out=values)
            output[start:end] = values
    return output


def make_soft_matrix(dist_sq, m, temperature, k):
    ids = np.argpartition(dist_sq, kth=m - 1, axis=1)[:, :m]
    selected = np.take_along_axis(dist_sq, ids, axis=1)
    order = np.argsort(selected, axis=1)
    ids = np.take_along_axis(ids, order, axis=1)
    selected = np.take_along_axis(selected, order, axis=1)
    shifted = selected - selected[:, :1]
    weights = np.exp(-shifted / max(float(temperature), 1e-12)).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    rows = np.repeat(np.arange(len(ids), dtype=np.int32), m)
    matrix = csr_matrix(
        (weights.reshape(-1), (rows, ids.reshape(-1))),
        shape=(len(ids), k), dtype=np.float32,
    )
    if not np.allclose(np.asarray(matrix.sum(axis=1)).ravel(), 1.0, atol=1e-5):
        raise RuntimeError("Soft-assignment weights do not sum to one")
    return matrix


def time_method(name, X, bundle, args):
    models = bundle["models"]
    centers = bundle["codebook_centers"]
    scaler = bundle["distance_scaler"]
    k = int(bundle["K"])
    m = int(bundle["m"])
    temperature = float(bundle["temperature"])
    soft_weight = float(bundle["soft_weight"])

    def run_once():
        if name == "continuous_dinov2_lr":
            representation = X
        else:
            dist_sq = squared_distances(X, centers, args.batch_size, args.threads)
            if name == "distance":
                distances = np.sqrt(dist_sq, out=dist_sq)
                representation = scaler.transform(distances).astype(np.float32, copy=False)
            else:
                soft = make_soft_matrix(dist_sq, m, temperature, k)
                if name == "soft":
                    representation = soft
                else:
                    distances = np.sqrt(dist_sq, out=dist_sq)
                    distance_features = scaler.transform(distances).astype(np.float32, copy=False)
                    representation = np.hstack((distance_features, soft.toarray() * soft_weight))
        return models[name].predict(representation)

    run_once()
    durations = []
    for _ in range(args.timing_repeats):
        start = time.perf_counter()
        run_once()
        durations.append(time.perf_counter() - start)
    median_seconds = float(np.median(durations))
    return median_seconds, median_seconds * 1000.0 / max(len(X), 1)


def efficiency_table(X_dev, bundle, args):
    sample_count = min(args.timing_samples, len(X_dev))
    X_time = X_dev[:sample_count]
    k = int(bundle["K"])
    m = int(bundle["m"])
    codebook_bytes = int(bundle["codebook_centers"].nbytes)
    scaler = bundle["distance_scaler"]
    scaler_bytes = int(scaler.mean_.nbytes + scaler.scale_.nbytes)
    n_images = len(X_dev)

    specs = [
        ("continuous_dinov2_lr", "dense float32 DINOv2 feature", 384 * 4, 0, 97.61),
        ("hard_code", "uint8 prototype ID", 1, codebook_bytes, ""),
        ("distance", "dense float32 distance vector", k * 4, codebook_bytes + scaler_bytes, 97.41),
        ("soft", "CSR soft weights (int32 index + float32 weight)", m * 8 + 4, codebook_bytes, 96.90),
        ("hybrid", "dense float32 distance + weighted soft vector", 2 * k * 4, codebook_bytes + scaler_bytes, 97.50),
    ]

    rows = []
    for name, description, payload_bytes, shared_bytes, test_accuracy in specs:
        if name == "hard_code":
            start = time.perf_counter()
            dist_sq = squared_distances(
                X_time, bundle["codebook_centers"], args.batch_size, args.threads
            )
            np.argmin(dist_sq, axis=1).astype(np.uint8)
            total_seconds = time.perf_counter() - start
            ms_per_image = total_seconds * 1000.0 / sample_count
        else:
            total_seconds, ms_per_image = time_method(name, X_time, bundle, args)
        classifier = bundle["models"].get(name)
        classifier_bytes = 0
        if classifier is not None:
            classifier_bytes = sum(
                getattr(classifier, attribute).nbytes
                for attribute in ("coef_", "intercept_", "classes_", "n_iter_")
                if hasattr(classifier, attribute)
            )
        total_shared_bytes = int(shared_bytes + classifier_bytes)
        rows.append({
            "representation": name,
            "storage_format": description,
            "payload_bytes_per_image": int(payload_bytes),
            "shared_codebook_and_scaler_bytes": int(shared_bytes),
            "classifier_parameter_bytes": int(classifier_bytes),
            "total_shared_artifact_bytes": total_shared_bytes,
            "amortized_bytes_per_image_over_50000": round(payload_bytes + total_shared_bytes / n_images, 3),
            "timing_sample_images": sample_count,
            "timing_scope": "encode_only_no_E6_classifier" if name == "hard_code" else "encode_plus_classifier_predict",
            "median_encode_plus_predict_ms_per_image": round(ms_per_image, 5),
            "median_encode_plus_predict_seconds": round(total_seconds, 5),
            "clean_test_accuracy_percent_from_E6": test_accuracy,
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    X_dev, y_dev, original_indices, bundle = load_inputs(args.data, args.model)
    centers = bundle["codebook_centers"]
    k = int(bundle["K"])

    print(f"Development features: {X_dev.shape}")
    print(f"Loaded E6 codebook: K={k}, m={bundle['m']}")
    print("Assigning development images to their nearest prototype")
    assignments, distances = assign_to_codebook(
        X_dev, centers, args.batch_size, args.threads
    )

    summary_rows, class_counts, cluster_sizes = prototype_statistics(
        assignments, distances, y_dev, k
    )
    example_rows = representative_examples(
        assignments, distances, y_dev, original_indices, args.representative_count
    )
    write_csv(args.results_dir / "prototype_summary.csv", summary_rows)
    write_csv(args.results_dir / "prototype_representative_examples.csv", example_rows)
    np.savez_compressed(
        args.outputs_dir / "prototype_class_counts.npz",
        class_counts=class_counts,
        cluster_sizes=cluster_sizes,
        codebook_centers=centers,
    )

    if not args.skip_images:
        top_examples = [row for row in example_rows if row["rank"] == 1]
        save_contact_sheet(
            args.results_dir / "prototype_contact_sheet.png", top_examples, args.cifar_root
        )

    print("Measuring representation storage and inference time")
    with threadpool_limits(limits=args.threads):
        efficiency_rows = efficiency_table(X_dev, bundle, args)
    write_csv(args.results_dir / "representation_efficiency.csv", efficiency_rows)

    sizes = [row["cluster_size"] for row in summary_rows]
    purities = [row["purity"] for row in summary_rows if row["cluster_size"]]
    entropies = [row["entropy_bits"] for row in summary_rows if row["cluster_size"]]
    summary = {
        "experiment": "Prototype interpretability and representation efficiency",
        "dataset": "CIFAR-10",
        "feature_extractor": "Frozen DINOv2 ViT-S/14 CLS features",
        "feature_dimension": int(X_dev.shape[1]),
        "development_images": int(len(X_dev)),
        "source_model": "Experiment 6 final model bundle",
        "codebook_size": k,
        "active_soft_prototypes": int(bundle["m"]),
        "temperature": float(bundle["temperature"]),
        "hybrid_soft_weight": float(bundle["soft_weight"]),
        "empty_prototypes": int(np.sum(cluster_sizes == 0)),
        "smallest_cluster": int(min(sizes)) if sizes else 0,
        "largest_cluster": int(max(sizes)) if sizes else 0,
        "mean_cluster_purity": float(np.mean(purities)) if purities else 0.0,
        "mean_cluster_entropy_bits": float(np.mean(entropies)) if entropies else 0.0,
        "images_per_prototype_contact_sheet": 1,
        "timing_note": "Median CPU feature-encoding plus classifier-prediction time on the selected feature subset; DINOv2 extraction is excluded.",
        "storage_note": "Per-image representation payload and amortized shared codebook/scaler/classifier parameter storage are estimates. They exclude Python/joblib serialization overhead and DINOv2 backbone storage.",
        "results_files": [
            "prototype_summary.csv",
            "prototype_representative_examples.csv",
            "representation_efficiency.csv",
        ],
    }
    with (args.results_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nPrototype summary")
    print(f"Prototypes: {k}")
    print(f"Empty prototypes: {summary['empty_prototypes']}")
    print(f"Mean purity: {summary['mean_cluster_purity']:.4f}")
    print(f"Mean entropy: {summary['mean_cluster_entropy_bits']:.4f} bits")
    print("\nRepresentation efficiency")
    for row in efficiency_rows:
        print(
            f"{row['representation']:<22} "
            f"{row['payload_bytes_per_image']:>5} bytes/image  "
            f"{row['median_encode_plus_predict_ms_per_image']:.5f} ms/image"
        )
    if args.skip_images:
        print("Representative image contact sheet skipped (--skip-images).")
    print(f"Results: {args.results_dir}")
    print(f"Outputs: {args.outputs_dir}")


if __name__ == "__main__":
    main()
