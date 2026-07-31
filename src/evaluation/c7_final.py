"""C7: leakage-safe final evaluation of frozen German Credit predictions.

The script is intentionally evaluation-only: it never loads a model and never
re-runs inference.  For every model, it derives a threshold from the committed
validation prediction artifact, freezes it, and applies it once to the matching
test prediction artifact.  This makes the published result reproducible on a
CPU-only environment and prevents test-set operating-point optimisation.

Run from the repository root with:

    python -m src.evaluation.c7_final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)

from src.evaluation.metrics import apply_threshold, select_cost_threshold


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FN_COST = 5
FP_COST = 1

# Each pair is a committed, model-output artifact.  Do not replace these paths
# with live inference: adapter weight binaries are intentionally excluded from
# the repository, while these predictions are the reproducible evaluation input.
PREDICTION_ARTIFACTS: Mapping[str, Mapping[str, str]] = {
    "Majority": {
        "valid": "outputs/baselines/majority_valid.jsonl",
        "test": "outputs/baselines/majority_test.jsonl",
    },
    "LogisticRegression": {
        "valid": "outputs/baselines/logistic_regression_valid.jsonl",
        "test": "outputs/baselines/logistic_regression_test.jsonl",
    },
    "Qwen-ZeroShot": {
        "valid": "outputs/baselines/qwen_zero_shot_valid.jsonl",
        "test": "outputs/baselines/qwen_zero_shot_test.jsonl",
    },
    "SFT-seed7": {
        "valid": "outputs/sft/german_sft_seed7/valid_predictions.jsonl",
        "test": "outputs/sft/german_sft_seed7/test_predictions.jsonl",
    },
    "SFT-multi": {
        "valid": "outputs/sft/german_multi/valid_predictions.jsonl",
        "test": "outputs/sft/german_multi/test_predictions.jsonl",
    },
}


def load_prediction_records(path: Path) -> list[dict[str, Any]]:
    """Load and validate a JSONL prediction artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"prediction artifact not found: {path}")

    records: list[dict[str, Any]] = []
    required_fields = {"sample_id", "ground_truth", "risk_score"}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = required_fields - record.keys()
        if missing:
            raise ValueError(f"{path}:{line_number} is missing required fields: {sorted(missing)}")
        if record["ground_truth"] not in (0, 1):
            raise ValueError(f"{path}:{line_number} has a non-binary ground_truth value")
        if record["risk_score"] is None or not np.isfinite(float(record["risk_score"])):
            raise ValueError(f"{path}:{line_number} has a missing or non-finite risk_score")
        records.append(record)

    if not records:
        raise ValueError(f"prediction artifact is empty: {path}")
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError(f"prediction artifact contains duplicate sample IDs: {path}")
    return records


def _labels_and_scores(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([record["ground_truth"] for record in records], dtype=int)
    scores = np.asarray([record["risk_score"] for record in records], dtype=float)
    return labels, scores


def validate_ground_truth_alignment(records: Sequence[Mapping[str, Any]], split: str,
                                    repository_root: Path = REPOSITORY_ROOT) -> None:
    """Ensure prediction labels and IDs exactly match the frozen German split."""
    source_path = repository_root / "data/processed/german/normalized" / f"{split}.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(f"frozen German split not found: {source_path}")
    source_records = [
        json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    expected_labels = {record["sample_id"]: record["risk_label"] for record in source_records}
    artifact_labels = {record["sample_id"]: record["ground_truth"] for record in records}
    if artifact_labels != expected_labels:
        raise ValueError(
            f"prediction artifact does not exactly match frozen German {split} labels; "
            "regenerate predictions from the declared split before evaluation"
        )


def compute_ece(labels: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    """Compute expected calibration error using fixed-width probability bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        # Include scores of exactly 1.0 in the final bin.
        if index == n_bins - 1:
            mask = (scores >= bins[index]) & (scores <= bins[index + 1])
        else:
            mask = (scores >= bins[index]) & (scores < bins[index + 1])
        if not mask.any():
            continue
        ece += (mask.sum() / len(scores)) * abs(labels[mask].mean() - scores[mask].mean())
    return float(ece)


def evaluate_at_frozen_threshold(test_records: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    """Evaluate test predictions at an already selected threshold.

    `threshold` is mandatory.  Keeping selection outside this function makes it
    impossible for this test evaluator to tune an operating point from test
    labels.
    """
    if threshold is None:
        raise ValueError("test evaluation requires a threshold frozen on validation data")

    labels, scores = _labels_and_scores(test_records)
    predictions = apply_threshold(scores, threshold)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    false_negatives = int(matrix[1, 0])
    false_positives = int(matrix[0, 1])
    epsilon = 1e-12

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "high_risk_recall": float(recall_score(labels, predictions, pos_label=1)),
        "low_risk_recall": float(recall_score(labels, predictions, pos_label=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "nll": float(log_loss(labels, np.clip(scores, epsilon, 1 - epsilon), labels=[0, 1])),
        "brier": float(brier_score_loss(labels, scores)),
        "ece": compute_ece(labels, scores),
        "threshold": float(threshold),
        "threshold_source": "validation_cost_minimization",
        "cost": FN_COST * false_negatives + FP_COST * false_positives,
        "hr_pred_rate": float(predictions.mean()),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_model(valid_records: Sequence[Mapping[str, Any]],
                   test_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze a validation-selected threshold, then evaluate the test artifact."""
    valid_labels, valid_scores = _labels_and_scores(valid_records)
    threshold, validation_cost = select_cost_threshold(
        valid_scores, valid_labels, fn_cost=FN_COST, fp_cost=FP_COST
    )
    metrics = evaluate_at_frozen_threshold(test_records, threshold)
    metrics["validation_cost"] = validation_cost
    return metrics


def run_evaluation(repository_root: Path = REPOSITORY_ROOT) -> dict[str, dict[str, Any]]:
    """Evaluate every committed prediction pair and return JSON-serialisable metrics."""
    results: dict[str, dict[str, Any]] = {}
    for model_name, artifact_paths in PREDICTION_ARTIFACTS.items():
        valid_path = repository_root / artifact_paths["valid"]
        test_path = repository_root / artifact_paths["test"]
        valid_records = load_prediction_records(valid_path)
        test_records = load_prediction_records(test_path)
        validate_ground_truth_alignment(valid_records, "valid", repository_root)
        validate_ground_truth_alignment(test_records, "test", repository_root)
        results[model_name] = evaluate_model(valid_records, test_records)
    return results


def print_report(results: Mapping[str, Mapping[str, Any]]) -> None:
    """Print a compact report suitable for local runs and CI logs."""
    headers = ("Model", "ROC-AUC", "PR-AUC", "NLL", "Brier", "ECE", "Threshold", "Cost")
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for model_name, metrics in results.items():
        print(
            f"{model_name} | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | "
            f"{metrics['nll']:.4f} | {metrics['brier']:.4f} | {metrics['ece']:.4f} | "
            f"{metrics['threshold']:.2f} | {metrics['cost']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/c7_final_metrics.json",
        help="path for the regenerated metrics JSON (default: outputs/c7_final_metrics.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_evaluation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print_report(results)
    print(f"\nSaved leakage-safe C7 metrics: {args.output}")


if __name__ == "__main__":
    main()
