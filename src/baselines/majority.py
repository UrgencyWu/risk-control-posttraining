"""
B0: Majority Class Baseline
Always predict "low risk" (the majority class).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.evaluation.metrics import (
    load_ground_truth, compute_metrics, save_predictions, load_predictions,
    generate_metrics_table
)

NORMALIZED_DIR = "data/processed/german/normalized"
OUTPUT_DIR = "outputs/baselines"


def run_majority_baseline():
    """B0: Always predict low_risk (0)."""
    results = {}

    for split in ("valid", "test"):
        # Load ground truth
        gt_records = load_ground_truth(f"{NORMALIZED_DIR}/{split}.jsonl")
        ground_truth = [r["risk_label"] for r in gt_records]

        # Always predict 0 (low risk)
        predictions = [0] * len(ground_truth)
        risk_scores = [0.0] * len(ground_truth)

        # Build prediction records
        pred_records = []
        for gt_r, pred, score in zip(gt_records, predictions, risk_scores):
            error_type = None
            if gt_r["risk_label"] == 1 and pred == 0:
                error_type = "false_negative"
            elif gt_r["risk_label"] == 0 and pred == 1:
                error_type = "false_positive"

            pred_records.append({
                "sample_id": gt_r["sample_id"],
                "ground_truth": gt_r["risk_label"],
                "prediction": pred,
                "risk_score": score,
                "threshold": 0.5,
                "error_type": error_type,
                "cost": 5 if error_type == "false_negative" else (1 if error_type == "false_positive" else 0),
                "model": "Majority",
            })

        save_predictions(pred_records, f"{OUTPUT_DIR}/majority_{split}.jsonl")
        metrics = compute_metrics(ground_truth, predictions, risk_scores)
        results["Majority"] = metrics
        print(f"  {split}: acc={metrics['accuracy']:.4f}, cost={metrics['cost']}, high_risk_recall={metrics['high_risk_recall']:.4f}")

    return results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== B0: Majority Class Baseline ===")
    results = run_majority_baseline()
    print("\n" + generate_metrics_table(results))
