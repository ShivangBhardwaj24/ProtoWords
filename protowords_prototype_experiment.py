#!/usr/bin/env python3
"""
ProtoWords experiment 1:
Class-wise K-Means prototypes on frozen DINOv2 CIFAR-10 features.

The official test set is deliberately NOT used here. Use validation results
to compare K=1,2,4,8, choose the configuration, then evaluate it once on test.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def l2_normalize(features):
    """Normalize every feature vector to have Euclidean length 1."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def learn_classwise_prototypes(X_train, y_train, k, random_state=42):
    """
    Learn k prototypes per class.
    K=1 uses the class mean; K>1 uses K-Means within each class.
    Returned prototypes are L2-normalized for cosine-similarity prediction.
    """
    all_prototypes = []
    prototype_labels = []

    for class_id in np.unique(y_train):
        class_features = X_train[y_train == class_id]

        if k == 1:
            centers = class_features.mean(axis=0, keepdims=True)
        else:
            model = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=10,
                max_iter=300,
                algorithm="lloyd"
            )
            model.fit(class_features)
            centers = model.cluster_centers_

        all_prototypes.append(centers)
        prototype_labels.extend([class_id] * len(centers))

    prototypes = np.vstack(all_prototypes).astype(np.float32)
    prototypes = l2_normalize(prototypes)
    return prototypes, np.asarray(prototype_labels, dtype=np.int64)


def predict_nearest_prototype(X, prototypes, prototype_labels, batch_size=2048):
    """Predict the label of the most cosine-similar prototype."""
    predictions = []
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        similarities = batch @ prototypes.T
        nearest = np.argmax(similarities, axis=1)
        predictions.append(prototype_labels[nearest])
    return np.concatenate(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz"
    )
    parser.add_argument("--output-dir", default="prototypes_experiment")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Cannot find {data_path}. Pass the correct path using --data."
        )

    data = np.load(data_path, allow_pickle=True)
    X_train = l2_normalize(data["X_train"].astype(np.float32))
    y_train = data["y_train"].astype(np.int64)
    X_val = l2_normalize(data["X_val"].astype(np.float32))
    y_val = data["y_val"].astype(np.int64)

    print("Loaded feature cache.")
    print(f"Training features: {X_train.shape}")
    print(f"Validation features: {X_val.shape}")
    print(f"Classes: {np.unique(y_train).tolist()}")
    print(f"Normalized training-vector norm (first 5): "
          f"{np.linalg.norm(X_train[:5], axis=1)}")

    rows = []
    saved_prototypes = {}

    for k in args.ks:
        print(f"\n{'=' * 60}\nLearning K={k} prototype(s) per class")
        prototypes, prototype_labels = learn_classwise_prototypes(
            X_train, y_train, k
        )
        val_predictions = predict_nearest_prototype(
            X_val, prototypes, prototype_labels
        )

        accuracy = accuracy_score(y_val, val_predictions)
        macro_f1 = f1_score(y_val, val_predictions, average="macro")

        rows.append({
            "K_prototypes_per_class": k,
            "total_prototypes": len(prototypes),
            "validation_accuracy": accuracy,
            "validation_macro_f1": macro_f1
        })
        saved_prototypes[f"prototypes_k{k}"] = prototypes
        saved_prototypes[f"labels_k{k}"] = prototype_labels

        print(f"Total prototypes: {len(prototypes)}")
        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Validation macro F1: {macro_f1:.4f}")

        report = classification_report(
            y_val,
            val_predictions,
            labels=list(range(10)),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )
        (output_dir / f"classification_report_k{k}.txt").write_text(report)

        matrix = confusion_matrix(y_val, val_predictions, labels=list(range(10)))
        np.savetxt(
            output_dir / f"confusion_matrix_k{k}.csv",
            matrix,
            delimiter=",",
            fmt="%d"
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "validation_results.csv", index=False)
    np.savez_compressed(output_dir / "learned_prototypes.npz", **saved_prototypes)

    print(f"\n{'=' * 60}\nVALIDATION COMPARISON")
    print(results.to_string(index=False))
    print(f"\nSaved results and prototypes in: {output_dir.resolve()}")
    print("\nDo not select a K based on the official test set. "
          "Choose using validation results first.")


if __name__ == "__main__":
    main()
