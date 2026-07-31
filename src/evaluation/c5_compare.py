"""C5: SFT vs DPO vs SimPO unified comparison on valid + test."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, confusion_matrix
)

def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def compute_all(gt, scores, preds):
    eps = 1e-12
    return {
        "roc_auc": roc_auc_score(gt, scores),
        "pr_auc": average_precision_score(gt, scores),
        "brier": brier_score_loss(gt, scores),
        "nll": log_loss(gt, np.clip(scores, eps, 1-eps), labels=[0,1]),
    }

def find_best_threshold(scores, gts, fn=5, fp=1):
    best_t, best_c = 0.5, float("inf")
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (np.array(scores) >= t).astype(int)
        c = sum(fn if g==1 and p==0 else (fp if g==0 and p==1 else 0) for g,p in zip(gts,preds))
        if c < best_c: best_c = c; best_t = t
    return best_t, best_c

models = {
    "SFT_seed7": "outputs/sft/german_sft_seed7",
    "DPO_seed42": "outputs/dpo/german_dpo_seed42",
    "SimPO_seed42": "outputs/dpo/german_simpo_seed42",
}

print("=" * 75)
print("C5: SFT vs DPO vs SimPO — Valid-set Evaluation")
print("=" * 75)

valid_results = {}
for name, path in models.items():
    vp = load(f"{path}/valid_predictions.jsonl")
    gt = [r["ground_truth"] for r in vp]
    sc = [r["risk_score"] for r in vp]
    m = compute_all(gt, sc, None)
    best_t, best_c = find_best_threshold(sc, gt)
    preds_opt = (np.array(sc) >= best_t).astype(int)
    cm = confusion_matrix(gt, preds_opt, labels=[0,1])
    hr_rate = preds_opt.mean()
    valid_results[name] = {**m, "threshold": best_t, "cost": best_c,
                            "hr_pred_rate": hr_rate, "cm": cm.tolist()}
    print(f"  {name:14s} PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} "
          f"NLL={m['nll']:.4f} Brier={m['brier']:.4f} "
          f"thresh={best_t:.2f} cost={best_c} HR%={hr_rate:.1%}")

print(f"\n{'='*75}")
print("C5: SFT vs DPO vs SimPO — Test-set Evaluation (threshold from valid)")
print(f"{'='*75}")

test_results = {}
for name, path in models.items():
    tp = load(f"{path}/test_predictions_raw.jsonl")
    gt = [r["ground_truth"] for r in tp]
    sc = [r["risk_score"] for r in tp]
    t = valid_results[name]["threshold"]
    preds_opt = (np.array(sc) >= t).astype(int)
    cm = confusion_matrix(gt, preds_opt, labels=[0,1])
    fn_c = cm[1,0]; fp_c = cm[0,1]
    cost = 5*fn_c + fp_c
    hr_rate = preds_opt.mean()
    m = compute_all(gt, sc, preds_opt)
    test_results[name] = {**m, "threshold": t, "cost": cost,
                           "hr_pred_rate": hr_rate, "cm": cm.tolist()}
    print(f"  {name:14s} PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} "
          f"NLL={m['nll']:.4f} Brier={m['brier']:.4f} "
          f"thresh={t:.2f} cost={cost} HR%={hr_rate:.1%} "
          f"cm=[TN={cm[0,0]} FP={fp_c} | FN={fn_c} TP={cm[1,1]}]")

# Unified table
print(f"\n{'='*75}")
print("C5: Unified Comparison Table (test set)")
print(f"{'='*75}")
print(f"{'Model':14s} {'ROC-AUC':>9} {'PR-AUC':>9} {'NLL':>8} {'Brier':>8} {'Thresh':>7} {'Cost':>6} {'HR%':>7}")
print("-" * 68)
for name in ["SFT_seed7", "DPO_seed42", "SimPO_seed42"]:
    m = test_results[name]
    print(f"{name:14s} {m['roc_auc']:>9.4f} {m['pr_auc']:>9.4f} {m['nll']:>8.4f} {m['brier']:>8.4f} "
          f"{m['threshold']:>7.2f} {m['cost']:>6.0f} {m['hr_pred_rate']:>6.1%}")

# Delta vs SFT
print(f"\n  Delta vs SFT:")
for name in ["DPO_seed42", "SimPO_seed42"]:
    d_roc = test_results[name]["roc_auc"] - test_results["SFT_seed7"]["roc_auc"]
    d_pr  = test_results[name]["pr_auc"] - test_results["SFT_seed7"]["pr_auc"]
    d_cost = test_results[name]["cost"] - test_results["SFT_seed7"]["cost"]
    d_nll  = test_results[name]["nll"] - test_results["SFT_seed7"]["nll"]
    print(f"    {name:14s} ΔROC-AUC={d_roc:+.4f}  ΔPR-AUC={d_pr:+.4f}  ΔNLL={d_nll:+.4f}  ΔCost={d_cost:+.0f}")

print(f"\n  Note: thresholds from valid only, applied to test without adjustment.")
