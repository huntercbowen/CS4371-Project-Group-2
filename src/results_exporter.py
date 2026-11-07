import csv
import json
from pathlib import Path

from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,)

def export_results(
    output_dir,
    hidden_attack,
    class_config,
    max_rows_per_file,
    y_true,
    y_pred,
    labels,
):
    """Export an experiment's results."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    # Calculate per-class and overall metrics
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    # Save experiment configuration
    config = {
        "hidden_attack": hidden_attack,
        "class_config": class_config,
        "max_rows_per_file": max_rows_per_file,
        "test_samples": len(y_true),
        "sampling_method": (
            "first_n_rows_per_file"
            if max_rows_per_file is not None
            else "full_dataset"
        ),
    }

    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    # Save overall metrics
    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
    }

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    # Save metrics for each class
    with (output_dir / "class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])

        for label in labels:
            metrics = report[str(label)]
            writer.writerow([
                label,
                metrics["precision"],
                metrics["recall"],
                metrics["f1-score"],
                metrics["support"],
            ])

    # Save confusion matrix with class labels
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Actual / Predicted", *labels])

        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])

    print(f"\nResults saved to: {output_dir}")
    return output_dir