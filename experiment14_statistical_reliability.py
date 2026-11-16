from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score


EXPERIMENT = "exp14_statistical_reliability"
E13_PREDICTIONS = Path(
    "results/exp13_quantized_final_test/final_test_predictions.csv"
)
BASELINE = "continuous_dinov2_lr"
COMPARISONS = ["k128_uint8", "k256_uint8", "k256_float16", "k256_float32"]
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
        description="Paired statistical comparison of Experiment 13 test predictions."
    )
    parser.add_argument("--predictions", type=Path, default=E13_PREDICTIONS)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if args.bootstrap_reps < 1000:
        parser.error("Use at least 1,000 bootstrap repetitions.")
    if not 0.0 < args.alpha < 1.0:
        parser.error("alpha must be between 0 and 1.")

    args.root = root
    args.predictions = (
        args.predictions.resolve()
        if args.predictions.is_absolute()
        else (root / args.predictions).resolve()
    )
    args.results_dir = root / "results" / EXPERIMENT
    args.outputs_dir = root / "outputs" / EXPERIMENT
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    return args


def load_predictions(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"E13 prediction file not found: {path}\n"
            "Run Experiment 13 first, or pass its final_test_predictions.csv with --predictions."
        )

    rows = {}
    required = {
        "image_index", "method", "true_label", "predicted_label"
    }
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")

        for row in reader:
            method = row["method"].strip()
            index = int(row["image_index"])
            true_label = int(row["true_label"])
            predicted_label = int(row["predicted_label"])
            if not 0 <= true_label < len(CLASS_NAMES):
                raise ValueError(f"Invalid true label {true_label} at image {index}.")
            if not 0 <= predicted_label < len(CLASS_NAMES):
                raise ValueError(f"Invalid predicted label {predicted_label} at image {index}.")
            if method not in rows:
                rows[method] = {}
            if index in rows[method]:
                raise ValueError(f"Duplicate prediction for {method}, image {index}.")
            rows[method][index] = (true_label, predicted_label)

    needed = [BASELINE] + COMPARISONS
    missing_methods = [name for name in needed if name not in rows]
    if missing_methods:
        raise ValueError(f"Prediction CSV is missing methods: {missing_methods}")

    indices = sorted(rows[BASELINE])
    if not indices:
        raise ValueError("No test predictions were found.")
    expected = set(indices)

    for method in needed:
        method_indices = set(rows[method])
        if method_indices != expected:
            raise ValueError(
                f"{method} does not contain exactly the same image indices as the baseline."
            )

    y_true = np.asarray([rows[BASELINE][i][0] for i in indices], dtype=np.int64)
    predictions = {}
    for method in needed:
        true_values = np.asarray([rows[method][i][0] for i in indices], dtype=np.int64)
        if not np.array_equal(true_values, y_true):
            raise ValueError(f"True labels do not match across methods for {method}.")
        predictions[method] = np.asarray(
            [rows[method][i][1] for i in indices], dtype=np.int64
        )

    if len(indices) != 10000:
        print(f"Warning: expected 10,000 E13 test images, found {len(indices)}.")
    return np.asarray(indices, dtype=np.int64), y_true, predictions


def paired_bootstrap_accuracy_difference(
    baseline_correct: np.ndarray,
    method_correct: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
):
    paired_difference = method_correct.astype(np.float32) - baseline_correct.astype(np.float32)
    n = len(paired_difference)
    bootstrap_means = np.empty(repetitions, dtype=np.float32)

    batch_size = 200
    for start in range(0, repetitions, batch_size):
        count = min(batch_size, repetitions - start)
        sample_indices = rng.integers(0, n, size=(count, n), dtype=np.int32)
        bootstrap_means[start:start + count] = paired_difference[sample_indices].mean(axis=1)

    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(lower * 100.0), float(upper * 100.0), bootstrap_means


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def write_csv(path: Path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    indices, y_true, predictions = load_predictions(args.predictions)
    print(f"Loaded E13 predictions: {len(indices)} images")
    print(f"Baseline: {BASELINE}")
    print(f"Comparisons: {COMPARISONS}")
    print(f"Paired bootstrap repetitions: {args.bootstrap_reps}")

    baseline_pred = predictions[BASELINE]
    baseline_correct = baseline_pred == y_true
    baseline_accuracy = float(baseline_correct.mean())
    baseline_macro_f1 = float(
        f1_score(y_true, baseline_pred, average="macro", zero_division=0)
    )

    rng = np.random.default_rng(args.seed)
    comparison_rows = []
    overlap_rows = []
    class_rows = []
    bootstrap_arrays = {}
    raw_p_values = []

    for method in COMPARISONS:
        method_pred = predictions[method]
        method_correct = method_pred == y_true
        method_accuracy = float(method_correct.mean())
        method_macro_f1 = float(
            f1_score(y_true, method_pred, average="macro", zero_division=0)
        )

        ci_low, ci_high, bootstrap_values = paired_bootstrap_accuracy_difference(
            baseline_correct,
            method_correct,
            args.bootstrap_reps,
            rng,
        )
        bootstrap_arrays[method] = bootstrap_values

        both_correct = int(np.sum(baseline_correct & method_correct))
        baseline_only_correct = int(np.sum(baseline_correct & ~method_correct))
        method_only_correct = int(np.sum(~baseline_correct & method_correct))
        both_wrong = int(np.sum(~baseline_correct & ~method_correct))
        discordant = baseline_only_correct + method_only_correct
        if discordant == 0:
            p_value = 1.0
        else:
            p_value = float(
                binomtest(
                    min(baseline_only_correct, method_only_correct),
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
        raw_p_values.append(p_value)

        comparison_rows.append({
            "baseline": BASELINE,
            "method": method,
            "n_images": len(y_true),
            "baseline_accuracy": baseline_accuracy,
            "method_accuracy": method_accuracy,
            "accuracy_difference_pp": (method_accuracy - baseline_accuracy) * 100.0,
            "paired_bootstrap_95ci_low_pp": ci_low,
            "paired_bootstrap_95ci_high_pp": ci_high,
            "baseline_macro_f1": baseline_macro_f1,
            "method_macro_f1": method_macro_f1,
            "macro_f1_difference_pp": (method_macro_f1 - baseline_macro_f1) * 100.0,
            "mcnemar_exact_p": p_value,
            "holm_adjusted_p": "",
            "significant_after_holm": "",
        })

        overlap_rows.append({
            "baseline": BASELINE,
            "method": method,
            "both_correct": both_correct,
            "baseline_correct_method_wrong": baseline_only_correct,
            "baseline_wrong_method_correct": method_only_correct,
            "both_wrong": both_wrong,
            "discordant_predictions": discordant,
        })

        for class_id, class_name in enumerate(CLASS_NAMES):
            mask = y_true == class_id
            baseline_class_acc = float(baseline_correct[mask].mean())
            method_class_acc = float(method_correct[mask].mean())
            class_rows.append({
                "method": method,
                "class_id": class_id,
                "class_name": class_name,
                "n_images": int(mask.sum()),
                "baseline_accuracy": baseline_class_acc,
                "method_accuracy": method_class_acc,
                "accuracy_difference_pp": (method_class_acc - baseline_class_acc) * 100.0,
            })

    adjusted = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(comparison_rows, adjusted):
        row["holm_adjusted_p"] = float(adjusted_p)
        row["significant_after_holm"] = bool(adjusted_p < args.alpha)

    results_dir = args.results_dir
    write_csv(results_dir / "paired_comparisons.csv", comparison_rows)
    write_csv(results_dir / "error_overlap.csv", overlap_rows)
    write_csv(results_dir / "classwise_comparisons.csv", class_rows)
    np.savez_compressed(
        args.outputs_dir / "paired_bootstrap_accuracy_differences.npz",
        **bootstrap_arrays,
    )

    summary = {
        "experiment": "Statistical reliability of E13 quantized ProtoWords test results",
        "dataset": "CIFAR-10 official test split",
        "prediction_source": str(args.predictions),
        "test_images": int(len(y_true)),
        "baseline": BASELINE,
        "baseline_accuracy": baseline_accuracy,
        "baseline_macro_f1": baseline_macro_f1,
        "compared_methods": COMPARISONS,
        "bootstrap": {
            "method": "Paired nonparametric bootstrap over test-image indices",
            "repetitions": args.bootstrap_reps,
            "confidence_level": 0.95,
            "random_seed": args.seed,
        },
        "significance_test": {
            "method": "Exact two-sided McNemar test on paired correctness indicators",
            "multiple_comparison_correction": "Holm step-down correction across four comparisons",
            "alpha": args.alpha,
        },
        "test_data_used_for_model_selection": False,
        "interpretation_note": (
            "Confidence intervals describe paired test-set accuracy differences under image resampling. "
            "McNemar tests compare discordant predictions. These analyses do not replace evaluation "
            "across multiple independently trained models or multiple data splits."
        ),
        "results": [
            "paired_comparisons.csv",
            "error_overlap.csv",
            "classwise_comparisons.csv",
        ],
        "bootstrap_output": "paired_bootstrap_accuracy_differences.npz",
    }
    with (results_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nPaired comparison against continuous DINOv2 baseline")
    print("method                 accuracy diff (pp)   paired 95% CI (pp)       McNemar p   Holm p")
    for row in comparison_rows:
        print(
            f"{row['method']:<22} "
            f"{row['accuracy_difference_pp']:+.3f}              "
            f"[{row['paired_bootstrap_95ci_low_pp']:+.3f}, "
            f"{row['paired_bootstrap_95ci_high_pp']:+.3f}]      "
            f"{row['mcnemar_exact_p']:.5g}   {row['holm_adjusted_p']:.5g}"
        )
    print(f"\nResults: {results_dir}")
    print(f"Outputs: {args.outputs_dir}")
    print("No model training or test-set-based model selection was performed.")


if __name__ == "__main__":
    main()
