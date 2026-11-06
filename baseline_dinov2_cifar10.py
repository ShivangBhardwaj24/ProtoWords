"""
ProtoWords Baseline Experiment
------------------------------

Method:
    CIFAR-10
        -> Frozen pretrained DINOv2 ViT-S/14
        -> Continuous image feature vectors
        -> L2 normalization
        -> Logistic regression
        -> Test-set evaluation

Important:
    DINOv2 is NOT fine-tuned.
    Only the logistic-regression classifier is trained.

The script automatically downloads CIFAR-10 and DINOv2
the first time they are required.

Features are cached to disk so they can be reused.
"""

import argparse
import random
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import normalize

from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from tqdm import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

MODEL_NAME = "dinov2_vits14"


# ============================================================
# 2. RANDOM SEED
# ============================================================

def set_seed(seed):
    """
    Fix random seeds for reproducibility where possible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. SELECT A BALANCED SUBSET
# ============================================================

def make_subset(dataset, limit, seed):
    """
    Select a random, approximately class-balanced subset.

    If limit is None or is at least the dataset size,
    the complete dataset is returned.

    Examples:
        limit=1000 -> approximately 100 images per class
        limit=5000 -> approximately 500 images per class
    """

    dataset_size = len(dataset)

    if limit is None or limit >= dataset_size:
        return dataset

    if limit < len(CLASS_NAMES):
        raise ValueError(
            "The subset must contain at least one image per class."
        )

    targets = np.asarray(dataset.targets)
    rng = np.random.default_rng(seed)

    num_classes = len(CLASS_NAMES)

    base_count, remainder = divmod(limit, num_classes)

    selected_indices = []

    for class_index in range(num_classes):

        class_indices = np.flatnonzero(
            targets == class_index
        )

        rng.shuffle(class_indices)

        # Distribute any remainder across the first classes.
        number_to_take = base_count

        if class_index < remainder:
            number_to_take += 1

        selected_indices.extend(
            class_indices[:number_to_take].tolist()
        )

    # Shuffle the final subset.
    rng.shuffle(selected_indices)

    return Subset(dataset, selected_indices)


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def get_image_transform():
    """
    Convert CIFAR-10 images to the input format expected
    by the pretrained DINOv2 model.

    CIFAR-10 images are 32x32 pixels.
    We resize them to 224x224 and normalize them.

    No random augmentation is applied because this is
    a feature-extraction baseline.
    """

    return transforms.Compose([

        transforms.Resize(
            (224, 224),
            interpolation=InterpolationMode.BICUBIC,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


# ============================================================
# 5. DOWNLOAD AND LOAD CIFAR-10
# ============================================================

def load_datasets(data_dir, train_limit, test_limit, seed):
    """
    Download CIFAR-10 if required and create train/test datasets.
    """

    transform = get_image_transform()

    print("\nLoading CIFAR-10...")

    train_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        transform=transform,
        download=True,
    )

    train_dataset = make_subset(
        train_dataset,
        train_limit,
        seed,
    )

    test_dataset = make_subset(
        test_dataset,
        test_limit,
        seed + 1,
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Testing images:  {len(test_dataset)}")

    return train_dataset, test_dataset


# ============================================================
# 6. LOAD THE PRETRAINED DINOv2 MODEL
# ============================================================

def load_dinov2(device):
    """
    Load the official pretrained DINOv2 ViT-S/14 backbone.

    On its first use, PyTorch Hub downloads the repository
    and pretrained weights.

    The model remains frozen throughout this experiment.
    """

    print("\nLoading pretrained DINOv2 ViT-S/14...")

    model = torch.hub.load(
        "facebookresearch/dinov2",
        MODEL_NAME,
    )

    model = model.to(device)
    model.eval()

    # We are using DINOv2 only for feature extraction.
    for parameter in model.parameters():
        parameter.requires_grad = False

    print("DINOv2 loaded successfully.")

    return model


# ============================================================
# 7. EXTRACT FEATURES
# ============================================================

def extract_features(
    model,
    dataset,
    device,
    batch_size,
    num_workers,
    split_name,
):
    """
    Pass images through frozen DINOv2 and save their features.

    GPU memory is controlled by processing a small batch
    of images at a time.

    Returns:
        X: feature matrix, shape approximately (N, 384)
        y: class labels, shape (N,)
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

    all_features = []
    all_labels = []

    print(f"\nExtracting {split_name} features...")

    start_time = time.time()

    with torch.inference_mode():

        for images, labels in tqdm(
            loader,
            desc=f"{split_name} feature extraction",
        ):

            images = images.to(
                device,
                non_blocking=use_cuda,
            )

            # Mixed precision reduces GPU memory usage.
            if use_cuda:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    features = model(images)

            else:

                features = model(images)

            # Store the features as float32 on the CPU.
            features = features.float().cpu().numpy()

            all_features.append(features)

            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0).astype(
        np.float32,
        copy=False,
    )

    y = np.concatenate(all_labels, axis=0).astype(
        np.int64,
        copy=False,
    )

    elapsed = time.time() - start_time

    print(f"{split_name} feature shape: {X.shape}")
    print(f"{split_name} extraction time: {elapsed:.2f} seconds")

    return X, y


# ============================================================
# 8. MAIN EXPERIMENT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="DINOv2 + Logistic Regression baseline"
    )

    parser.add_argument(
        "--train-limit",
        type=int,
        default=50000,
        help="Number of training images; default is all 50000.",
    )

    parser.add_argument(
        "--test-limit",
        type=int,
        default=10000,
        help="Number of test images; default is all 10000.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Feature extraction batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of CPU workers for data loading.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset selection.",
    )

    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Recompute features even if a cache exists.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate parameters
    # --------------------------------------------------------

    if args.train_limit <= 0 or args.test_limit <= 0:
        raise ValueError(
            "Training and test limits must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative.")

    set_seed(args.seed)

    # --------------------------------------------------------
    # Device and paths
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("ProtoWords Baseline: DINOv2 + Logistic Regression")
    print("=" * 60)

    print(f"Device: {device}")

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    else:

        print(
            "WARNING: CUDA is unavailable. "
            "The experiment will run on the CPU."
        )

    project_dir = Path.cwd()

    data_dir = project_dir / "data"
    output_dir = project_dir / "outputs"

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = (
        f"train{args.train_limit}"
        f"_test{args.test_limit}"
        f"_seed{args.seed}"
    )

    cache_path = (
        output_dir / f"dinov2_features_{tag}.npz"
    )

    # --------------------------------------------------------
    # Extract features or reuse the existing cache
    # --------------------------------------------------------

    total_start = time.time()

    if cache_path.exists() and not args.force_extract:

        print("\nFound a feature cache.")
        print("Loading previously extracted features...")

        with np.load(cache_path) as saved:

            X_train = saved["X_train"]
            y_train = saved["y_train"]

            X_test = saved["X_test"]
            y_test = saved["y_test"]

        print("Feature cache loaded.")

    else:

        train_dataset, test_dataset = load_datasets(
            data_dir=data_dir,
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            seed=args.seed,
        )

        model = load_dinov2(device)

        X_train, y_train = extract_features(
            model=model,
            dataset=train_dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_name="Training",
        )

        X_test, y_test = extract_features(
            model=model,
            dataset=test_dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_name="Testing",
        )

        print("\nSaving extracted features...")

        np.savez_compressed(
            cache_path,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        print(f"Features saved to: {cache_path}")

        # Release GPU model memory.
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Normalize the feature vectors
    # --------------------------------------------------------

    print("\nNormalizing feature vectors...")

    X_train = normalize(
        X_train,
        norm="l2",
        axis=1,
    ).astype(np.float32, copy=False)

    X_test = normalize(
        X_test,
        norm="l2",
        axis=1,
    ).astype(np.float32, copy=False)

    print(f"Training feature matrix: {X_train.shape}")
    print(f"Test feature matrix:     {X_test.shape}")

    # --------------------------------------------------------
    # Train logistic regression
    # --------------------------------------------------------

    print("\nTraining Logistic Regression...")

    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=args.seed,
    )

    classifier_start = time.time()

    classifier.fit(
        X_train,
        y_train,
    )

    classifier_time = time.time() - classifier_start

    print(
        f"Classifier training time: "
        f"{classifier_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Evaluate the classifier
    # --------------------------------------------------------

    print("\nEvaluating on the test set...")

    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        y_test,
        predictions,
        average="micro",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)

    print(f"Training images: {len(y_train)}")
    print(f"Test images:     {len(y_test)}")

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Micro F1:        {micro_f1:.4f}")

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # Save trained classifier
    # --------------------------------------------------------

    classifier_path = (
        output_dir / f"logreg_{tag}.joblib"
    )

    joblib.dump(
        classifier,
        classifier_path,
    )

    print(f"Classifier saved to: {classifier_path}")

    # --------------------------------------------------------
    # Save evaluation report
    # --------------------------------------------------------

    report_path = (
        output_dir / f"report_{tag}.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:

        file.write("ProtoWords: DINOv2 Baseline\n\n")

        file.write(f"Model: {MODEL_NAME}\n")
        file.write("Classifier: Logistic Regression\n")
        file.write("C: 1.0\n")
        file.write("Feature normalization: L2\n\n")

        file.write(f"Training images: {len(y_train)}\n")
        file.write(f"Test images: {len(y_test)}\n")

        file.write(f"Accuracy: {accuracy:.6f}\n")
        file.write(f"Macro F1: {macro_f1:.6f}\n")
        file.write(f"Micro F1: {micro_f1:.6f}\n\n")

        file.write("Classification Report:\n")
        file.write(report)

    print(f"Report saved to: {report_path}")

    # --------------------------------------------------------
    # Save the confusion matrix
    # --------------------------------------------------------

    figure, axis = plt.subplots(figsize=(10, 8))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=np.arange(len(CLASS_NAMES)),
        display_labels=CLASS_NAMES,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )

    axis.set_title(
        "CIFAR-10: DINOv2 + Logistic Regression"
    )

    figure.tight_layout()

    confusion_path = (
        output_dir / f"confusion_matrix_{tag}.png"
    )

    figure.savefig(
        confusion_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Confusion matrix saved to: {confusion_path}")

    # --------------------------------------------------------
    # Final timing
    # --------------------------------------------------------

    total_time = time.time() - total_start

    print(f"\nTotal experiment time: {total_time:.2f} seconds")

    print("\nBaseline experiment completed successfully.")


if __name__ == "__main__":
    main()
