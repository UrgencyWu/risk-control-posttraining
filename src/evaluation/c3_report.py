"""C3 close-out: per-seed evaluation with valid-selected thresholds."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, numpy as np
from sklearn.metrics import confusion_matrix, recall_score

SEEDS = [10086, 42, 7]
OUT_DIR = "outputs/sft"

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def compute_cost(gts, preds, fn=5, fp=1):
    c = 0
    for g, p in zip(gts, preds):
        if g == 1 and p == 0: c += fn
        elif g == 0 and p == 1: c += fp
    return c

def find_best_threshold(scores, gts):
    best_t, best_c = 0.5, float("inf")
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (np.array(scores) >= t).astype(int)
        c = compute_cost(gts, preds)
        if c < best_c:
            best_c = c
            best_t = t
    return best_t, best_c

print("=" * 70)
print("C3 Close-out: Per-seed Evaluation (threshold from VALID only)")
print("=" * 70)

all_test = {}

for seed in SEEDS:
    run_dir = f"{OUT_DIR}/german_sft_seed{seed}"
    print(f"\n--- Seed {seed} ---")

    # Load valid predictions (always have raw scores)
    vp = load_jsonl(f"{run_dir}/valid_predictions.jsonl")
    v_gt = [r["ground_truth"] for r in vp]
    v_scores = [r["risk_score"] for r in vp]

    # Load test predictions (may be raw or adjusted; always have risk_score)
    # Prefer _raw if exists, else use regular
    tp_path = f"{run_dir}/test_predictions_raw.jsonl"
    if not os.path.exists(tp_path):
        tp_path = f"{run_dir}/test_predictions.jsonl"
    tp = load_jsonl(tp_path)
    t_gt = [r["ground_truth"] for r in tp]
    t_scores = [r["risk_score"] for r in tp]

    # ---- Default threshold 0.5 ----
    v_pred_05 = [1 if s >= 0.5 else 0 for s in v_scores]
    v_cost_05 = compute_cost(v_gt, v_pred_05)
    t_pred_05 = [1 if s >= 0.5 else 0 for s in t_scores]
    t_cost_05 = compute_cost(t_gt, t_pred_05)
    cm_05 = confusion_matrix(t_gt, t_pred_05, labels=[0, 1])

    # ---- Cost-optimal threshold (VALID only, cost=5*FN+1*FP) ----
    best_t, best_c = find_best_threshold(v_scores, v_gt)
    print(f"  Default (0.50): valid_cost={v_cost_05}  test_cost={t_cost_05}")
    print(f"  Optimal ({best_t:.2f}): valid_cost={best_c}")

    # Apply optimal threshold to test
    t_pred_opt = [1 if s >= best_t else 0 for s in t_scores]
    t_cost_opt = compute_cost(t_gt, t_pred_opt)
    cm_opt = confusion_matrix(t_gt, t_pred_opt, labels=[0, 1])
    tn, fp = cm_opt[0, 0], cm_opt[0, 1]
    fn_n, tp_n = cm_opt[1, 0], cm_opt[1, 1]

    hr_pred_rate = np.mean(t_pred_opt)
    low_risk_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    high_risk_recall = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0

    print(f"  Test (default 0.5):  TN={cm_05[0,0]} FP={cm_05[0,1]} | FN={cm_05[1,0]} TP={cm_05[1,1]}  cost={t_cost_05}")
    print(f"  Test (optimal {best_t:.2f}): TN={tn} FP={fp} | FN={fn_n} TP={tp_n}  cost={t_cost_opt}")
    print(f"  High-risk pred rate: {hr_pred_rate:.2%}  Low-risk recall: {low_risk_recall:.4f}  HR recall: {high_risk_recall:.4f}")

    # Save adjusted test predictions
    adjusted = []
    for r in tp:
        s = r["risk_score"]
        pred = 1 if s >= best_t else 0
        gt = r["ground_truth"]
        err = None
        if gt == 1 and pred == 0: err = "false_negative"
        elif gt == 0 and pred == 1: err = "false_positive"
        adjusted.append({
            "sample_id": r["sample_id"], "ground_truth": gt,
            "prediction": pred, "risk_score": r["risk_score"],
            "threshold": round(best_t, 2), "error_type": err,
            "cost": 5 if err == "false_negative" else (1 if err == "false_positive" else 0),
            "model": "Qwen3.5-4B-SFT",
        })
    with open(f"{run_dir}/test_predictions.jsonl", "w") as f:
        for r in adjusted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    all_test[seed] = {
        "threshold": best_t,
        "cost_05": t_cost_05, "cost_opt": t_cost_opt,
        "hr_pred_rate": hr_pred_rate,
        "low_risk_recall": low_risk_recall,
        "high_risk_recall": high_risk_recall,
        "cm_05": cm_05.tolist(), "cm_opt": cm_opt.tolist(),
    }

# ---- Aggregate ----
print(f"\n{'='*70}")
print("Aggregate (3 seeds, optimal threshold per seed)")
print(f"{'='*70}")

for metric in ["cost_05", "cost_opt", "hr_pred_rate", "low_risk_recall", "high_risk_recall"]:
    vals = [all_test[s][metric] for s in SEEDS]
    mean_v = np.mean(vals)
    std_v = np.std(vals)
    desc = {"cost_05":"Cost@0.5","cost_opt":"Cost@opt","hr_pred_rate":"HR pred rate",
            "low_risk_recall":"Low-risk recall","high_risk_recall":"High-risk recall"}[metric]
    if isinstance(vals[0], float):
        print(f"  {desc:20s}: {mean_v:.4f} ± {std_v:.4f}")
    else:
        print(f"  {desc:20s}: {mean_v:.1f} ± {std_v:.1f}")

# Confusion matrix sum
sum_cm = np.zeros((2, 2), dtype=int)
for s in SEEDS:
    sum_cm += np.array(all_test[s]["cm_opt"])
print(f"\n  Summed confusion (optimal):")
print(f"    TN={sum_cm[0,0]} FP={sum_cm[0,1]}")
print(f"    FN={sum_cm[1,0]} TP={sum_cm[1,1]}")
print(f"    Total cost: {sum_cm[1,0]*5 + sum_cm[0,1]*1}")

# Summary table
print(f"\n{'Seed':<12} {'Thresh':>6} {'Cost@0.5':>9} {'Cost@opt':>9} {'HR%':>7} {'LR.Recall':>10} {'HR.Recall':>10}")
print("-" * 65)
for s in SEEDS:
    d = all_test[s]
    print(f"{s:<12} {d['threshold']:>6.2f} {d['cost_05']:>9.0f} {d['cost_opt']:>9.0f} "
          f"{d['hr_pred_rate']:>7.2%} {d['low_risk_recall']:>10.4f} {d['high_risk_recall']:>10.4f}")

# Mean row
means = {k: np.mean([all_test[s][k] for s in SEEDS]) for k in all_test[SEEDS[0]]}
stds  = {k: np.std([all_test[s][k] for s in SEEDS]) for k in all_test[SEEDS[0]]}
print("-" * 65)
print(f"{'mean':<12} {means['threshold']:>6.2f} {means['cost_05']:>9.0f} {means['cost_opt']:>9.0f} "
      f"{means['hr_pred_rate']:>7.2%} {means['low_risk_recall']:>10.4f} {means['high_risk_recall']:>10.4f}")
print(f"{'±std':<12} {'':>6} {stds['cost_05']:>9.0f} {stds['cost_opt']:>9.0f} "
      f"{stds['hr_pred_rate']:>7.2%} {stds['low_risk_recall']:>10.4f} {stds['high_risk_recall']:>10.4f}")

print(f"\n  Conclusion: Cost-optimal threshold reduces cost from "
      f"{means['cost_05']:.0f}±{stds['cost_05']:.0f} to {means['cost_opt']:.0f}±{stds['cost_opt']:.0f} "
      f"({(1 - means['cost_opt']/means['cost_05'])*100:.0f}% reduction)")
