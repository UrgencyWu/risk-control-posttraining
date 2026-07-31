"""
Shared evaluation metrics for German Credit baselines.
All metrics computed from prediction records with format:
  {sample_id, ground_truth, prediction, risk_score, threshold, error_type, cost, model}
"""
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    recall_score, precision_score, roc_auc_score, average_precision_score,
    confusion_matrix
)


def load_predictions(path):
    """Load prediction JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_ground_truth(normalized_path):
    """Load ground truth from normalized JSONL."""
    records = []
    with open(normalized_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_cost(ground_truth, prediction, fn_cost=5, fp_cost=1):
    """
    Compute asymmetric cost.
    FN: actual high risk (1), predicted low risk (0)
    FP: actual low risk (0), predicted high risk (1)
    """
    total_cost = 0
    fn_count = 0
    fp_count = 0
    for gt, pred in zip(ground_truth, prediction):
        if gt == 1 and pred == 0:
            total_cost += fn_cost
            fn_count += 1
        elif gt == 0 and pred == 1:
            total_cost += fp_cost
            fp_count += 1
    return total_cost, fn_count, fp_count


def compute_metrics(ground_truth, predictions, risk_scores=None):
    """
    Compute all baseline metrics.
    Returns dict with: accuracy, balanced_acc, macro_f1, high_risk_recall,
                       roc_auc, pr_auc, cost, valid_rate
    """
    gt = np.array(ground_truth)
    pred = np.array(predictions)

    # Basic metrics
    acc = accuracy_score(gt, pred)
    balanced_acc = balanced_accuracy_score(gt, pred)
    macro_f1 = f1_score(gt, pred, average="macro")
    high_risk_recall = recall_score(gt, pred, pos_label=1)

    # Cost
    total_cost, fn_count, fp_count = compute_cost(gt, pred)

    # Valid rate: proportion of parseable predictions
    valid_rate = 1.0  # default; for LLM, may be <1 if output is unparseable

    result = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "high_risk_recall": round(high_risk_recall, 4),
        "cost": total_cost,
        "fn_count": fn_count,
        "fp_count": fp_count,
        "valid_rate": valid_rate,
    }

    # ROC-AUC and PR-AUC require risk scores
    if risk_scores is not None:
        scores = np.array(risk_scores)
        # Filter out None/NaN scores
        valid_mask = ~np.isnan(scores)
        if valid_mask.sum() > 1 and len(np.unique(gt[valid_mask])) > 1:
            result["roc_auc"] = round(roc_auc_score(gt[valid_mask], scores[valid_mask]), 4)
            result["pr_auc"] = round(average_precision_score(gt[valid_mask], scores[valid_mask]), 4)
        else:
            result["roc_auc"] = None
            result["pr_auc"] = None
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None

    return result


def generate_metrics_table(results_dict):
    """
    Generate the unified metrics table.
    results_dict: {model_name: metrics_dict}
    """
    headers = ["Model", "Accuracy", "Balanced Acc.", "Macro-F1", "High-risk Recall",
               "ROC-AUC", "PR-AUC", "Cost", "Valid Rate"]
    rows = [headers]

    for model_name, m in results_dict.items():
        row = [
            model_name,
            f"{m['accuracy']:.4f}",
            f"{m['balanced_accuracy']:.4f}",
            f"{m['macro_f1']:.4f}",
            f"{m['high_risk_recall']:.4f}",
            f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A",
            f"{m['pr_auc']:.4f}" if m['pr_auc'] is not None else "N/A",
            str(m['cost']),
            f"{m['valid_rate']:.4f}",
        ]
        rows.append(row)

    # Format as markdown table
    col_widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    md_lines = []
    for i, row in enumerate(rows):
        line = "| " + " | ".join(r.ljust(col_widths[j]) for j, r in enumerate(row)) + " |"
        md_lines.append(line)
        if i == 0:
            line = "|" + "|".join("-" * (col_widths[j] + 2) for j in range(len(headers))) + "|"
            md_lines.append(line)
    return "\n".join(md_lines)


def save_predictions(records, path):
    """Save prediction records as JSONL."""
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
