from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "outputs").exists() else SCRIPT_DIR.parent
EXPERIMENT = "exp4_local_soft_assignment"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
N_CLASSES = len(CLASS_NAMES)
FEATURE_DIM = 384
DEFAULT_CACHE = Path(
    "outputs/exp1_baseline/"
    "cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--ks", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--m-values", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--temperature-multipliers", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1, 10, 100])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter-kmeans", type=int, default=300)
    parser.add_argument("--max-iter-logreg", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.data.is_absolute():
        args.data = args.data.resolve()
    else:
        args.data = (ROOT / args.data).resolve()

    args.ks = sorted(set(args.ks))
    args.m_values = sorted(set(args.m_values))
    args.temperature_multipliers = sorted(set(args.temperature_multipliers))
    args.c_values = sorted(set(args.c_values))

    if not args.ks or any(k < 2 for k in args.ks):
        parser.error("K values must be at least 2")
    if not args.m_values or any(m < 1 for m in args.m_values):
        parser.error("m values must be positive integers")
    if not args.temperature_multipliers or any(t <= 0 for t in args.temperature_multipliers):
        parser.error("Temperature multipliers must be positive")
    if not args.c_values or any(c <= 0 for c in args.c_values):
        parser.error("C values must be positive")
    if min(args.n_init, args.max_iter_kmeans, args.max_iter_logreg,
           args.batch_size, args.threads) < 1:
        parser.error("Iteration limits, batch size, and threads must be positive")

    args.results_dir = ROOT / "results" / EXPERIMENT
    args.outputs_dir = ROOT / "outputs" / EXPERIMENT
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    return args


def l2_normalize(X):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def load_features(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature cache not found: {path}\n"
            "Run Experiment 1 first, or pass the cache path with --data."
        )

    with np.load(path, allow_pickle=False) as data:
        required = {"X_train", "y_train", "X_val", "y_val"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Feature cache is missing: {sorted(missing)}")
        X_train = np.asarray(data["X_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=np.int64)
        X_val = np.asarray(data["X_val"], dtype=np.float32)
        y_val = np.asarray(data["y_val"], dtype=np.int64)

    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("Feature arrays must be two-dimensional")
    if X_train.shape[1] != FEATURE_DIM or X_val.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM}-dimensional DINOv2 features")
    if len(X_train) != len(y_train) or len(X_val) != len(y_val):
        raise ValueError("Feature and label counts do not match")
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Training and validation data must not be empty")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Feature cache contains NaN or infinite values")
    if not np.isin(y_train, np.arange(N_CLASSES)).all():
        raise ValueError("Training labels must be integers from 0 to 9")
    if not np.isin(y_val, np.arange(N_CLASSES)).all():
        raise ValueError("Validation labels must be integers from 0 to 9")

    return l2_normalize(X_train), y_train, l2_normalize(X_val), y_val


def nearest_prototypes(X, centers, m, batch_size, threads):
    center_norms = np.sum(centers * centers, axis=1)[None, :]
    all_ids = []
    all_distances = []

    with threadpool_limits(limits=threads):
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size]
            distances = (
                np.sum(batch * batch, axis=1, keepdims=True)
                + center_norms
                - 2.0 * (batch @ centers.T)
            )
            np.maximum(distances, 0.0, out=distances)
            ids = np.argpartition(distances, kth=m - 1, axis=1)[:, :m]
            selected = np.take_along_axis(distances, ids, axis=1)
            order = np.argsort(selected, axis=1)
            all_ids.append(np.take_along_axis(ids, order, axis=1))
            all_distances.append(np.take_along_axis(selected, order, axis=1))

    return np.vstack(all_ids).astype(np.int32), np.vstack(all_distances).astype(np.float32)


def make_soft_features(ids, squared_distances, n_prototypes, temperature):
    n_samples, m = ids.shape
    if m == 1:
        weights = np.ones((n_samples, 1), dtype=np.float32)
    else:
        shifted = squared_distances - squared_distances[:, :1]
        weights = np.exp(-shifted / max(float(temperature), 1e-12)).astype(np.float32)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    rows = np.repeat(np.arange(n_samples, dtype=np.int32), m)
    return csr_matrix(
        (weights.reshape(-1), (rows, ids.reshape(-1))),
        shape=(n_samples, n_prototypes),
        dtype=np.float32,
    )


def metrics_for(y_true, y_pred):
    return {
        "validation_accuracy": float(accuracy_score(y_true, y_pred)),
        "validation_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "validation_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_confusion_matrix(path, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(N_CLASSES))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + CLASS_NAMES)
        for name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([name] + row.tolist())


def save_predictions(path, y_true, y_pred):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["validation_index", "true_label", "true_class", "predicted_label", "predicted_class"])
        for index, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            writer.writerow([
                index, int(true_label), CLASS_NAMES[int(true_label)],
                int(pred_label), CLASS_NAMES[int(pred_label)],
            ])


def main():
    args = parse_args()
    X_train, y_train, X_val, y_val = load_features(args.data)

    if max(args.ks) > len(X_train):
        raise ValueError("K cannot exceed the number of training images")
    if max(args.m_values) > min(args.ks):
        raise ValueError("Each m value must be no larger than the smallest K")

    result_rows = []
    search_rows = []
    best_overall = None
    run_start = time.perf_counter()

    print(f"Training features: {X_train.shape}")
    print(f"Validation features: {X_val.shape}")
    print(f"K values: {args.ks}")
    print(f"Active prototype counts: {args.m_values}")

    for k in args.ks:
        start = time.perf_counter()
        kmeans = KMeans(
            n_clusters=k,
            random_state=args.seed,
            n_init=args.n_init,
            max_iter=args.max_iter_kmeans,
            algorithm="lloyd",
        )
        with threadpool_limits(limits=args.threads):
            kmeans.fit(X_train)
        kmeans_seconds = time.perf_counter() - start
        centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)

        max_m = max(m for m in args.m_values if m <= k)
        train_ids, train_d2 = nearest_prototypes(
            X_train, centers, max_m, args.batch_size, args.threads
        )
        val_ids, val_d2 = nearest_prototypes(
            X_val, centers, max_m, args.batch_size, args.threads
        )

        if max_m > 1:
            temperature_reference = float(np.median(train_d2[:, 1] - train_d2[:, 0]))
            temperature_reference = max(temperature_reference, 1e-12)
        else:
            temperature_reference = 1.0

        print(f"K={k}: codebook fitted")

        for m in args.m_values:
            if m > k:
                continue
            multipliers = [None] if m == 1 else args.temperature_multipliers

            for multiplier in multipliers:
                temperature = None if multiplier is None else temperature_reference * multiplier
                X_train_repr = make_soft_features(
                    train_ids[:, :m], train_d2[:, :m], k,
                    1.0 if temperature is None else temperature,
                )
                X_val_repr = make_soft_features(
                    val_ids[:, :m], val_d2[:, :m], k,
                    1.0 if temperature is None else temperature,
                )

                best_config = None
                for c_value in args.c_values:
                    model = LogisticRegression(
                        C=float(c_value), solver="lbfgs",
                        max_iter=args.max_iter_logreg, random_state=args.seed,
                    )
                    fit_start = time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        with threadpool_limits(limits=args.threads):
                            model.fit(X_train_repr, y_train)
                    fit_seconds = time.perf_counter() - fit_start
                    convergence_warning = any(
                        "converg" in str(item.message).lower() for item in caught
                    )

                    predictions = model.predict(X_val_repr)
                    scores = metrics_for(y_val, predictions)
                    search_row = {
                        "K": k,
                        "m": m,
                        "temperature_multiplier": "none" if multiplier is None else multiplier,
                        "temperature": "none" if temperature is None else temperature,
                        "C": float(c_value),
                        **scores,
                        "fit_seconds": fit_seconds,
                        "convergence_warning": convergence_warning,
                    }
                    search_rows.append(search_row)

                    key = (scores["validation_accuracy"], scores["validation_macro_f1"], -float(c_value))
                    if best_config is None or key > best_config["key"]:
                        best_config = {
                            "key": key,
                            "model": model,
                            "C": float(c_value),
                            "scores": scores,
                            "predictions": predictions,
                        }

                row = {
                    "K": k,
                    "m": m,
                    "temperature_multiplier": "none" if multiplier is None else multiplier,
                    "temperature": "none" if temperature is None else temperature,
                    "selected_C": best_config["C"],
                    **best_config["scores"],
                    "kmeans_fit_seconds": kmeans_seconds,
                }
                result_rows.append(row)

                overall_key = (
                    row["validation_accuracy"],
                    row["validation_macro_f1"],
                    -m,
                    -k,
                    0.0 if multiplier is None else -float(multiplier),
                )
                if best_overall is None or overall_key > best_overall["key"]:
                    best_overall = {
                        "key": overall_key,
                        "row": row.copy(),
                        "centers": centers.copy(),
                        "classifier": best_config["model"],
                        "predictions": best_config["predictions"].copy(),
                        "temperature_reference": temperature_reference,
                    }

        print(f"K={k}: hard and soft assignments evaluated")

    result_rows.sort(key=lambda r: (
        -r["validation_accuracy"], -r["validation_macro_f1"], r["m"], r["K"]
    ))

    results_path = args.results_dir / "validation_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    search_path = args.results_dir / "hyperparameter_search.csv"
    with search_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(search_rows[0].keys()))
        writer.writeheader()
        writer.writerows(search_rows)

    best_row = best_overall["row"]
    best_predictions = best_overall["predictions"]
    report = classification_report(
        y_val, best_predictions, labels=np.arange(N_CLASSES),
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    )
    (args.results_dir / "best_validation_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(
        args.results_dir / "best_validation_confusion_matrix.csv", y_val, best_predictions
    )
    save_predictions(
        args.results_dir / "best_validation_predictions.csv", y_val, best_predictions
    )

    model_bundle = {
        "method": "localized_soft_assignment_protowords",
        "K": int(best_row["K"]),
        "m": int(best_row["m"]),
        "temperature_multiplier": best_row["temperature_multiplier"],
        "temperature": best_row["temperature"],
        "temperature_reference": best_overall["temperature_reference"],
        "selected_C": float(best_row["selected_C"]),
        "centers": best_overall["centers"],
        "classifier": best_overall["classifier"],
        "class_names": CLASS_NAMES,
    }
    joblib.dump(model_bundle, args.outputs_dir / "best_validation_model.joblib")

    summary = {
        "experiment": "Localized Soft Assignment ProtoWords",
        "dataset": "CIFAR-10",
        "feature_extractor": "Frozen DINOv2 ViT-S/14 CLS features",
        "feature_dimension": FEATURE_DIM,
        "feature_cache": str(args.data),
        "training_images": int(len(y_train)),
        "validation_images": int(len(y_val)),
        "official_test_used": False,
        "normalization": "L2 normalization per image feature",
        "codebook_learning": "One global K-Means codebook fitted on training features for each K",
        "assignment": "Softmax over negative squared distances to the m nearest prototypes",
        "temperature_reference": "Median training-set gap between nearest and second-nearest squared distances",
        "k_values": args.ks,
        "m_values": args.m_values,
        "temperature_multipliers": args.temperature_multipliers,
        "c_values": args.c_values,
        "seed": args.seed,
        "kmeans_n_init": args.n_init,
        "best_validation_configuration": best_row,
        "total_runtime_seconds": time.perf_counter() - run_start,
        "results_file": "validation_results.csv",
        "hyperparameter_search_file": "hyperparameter_search.csv",
    }
    with (args.results_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nBest validation configuration")
    print(
        f"K={best_row['K']}, m={best_row['m']}, "
        f"temperature multiplier={best_row['temperature_multiplier']}, "
        f"C={best_row['selected_C']}"
    )
    print(
        f"Accuracy={best_row['validation_accuracy']:.4f}, "
        f"Macro-F1={best_row['validation_macro_f1']:.4f}"
    )
    print(f"Results: {args.results_dir}")
    print(f"Outputs: {args.outputs_dir}")
    print("Official CIFAR-10 test data was not loaded or evaluated.")


if __name__ == "__main__":
    main()
