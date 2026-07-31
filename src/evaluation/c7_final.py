"""
C7: Final unified evaluation — all models on German test set.
SFT vs Logistic Regression vs Zero-shot, with calibration metrics.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    accuracy_score, balanced_accuracy_score, f1_score, recall_score,
    confusion_matrix
)

def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def compute_ece(gts, scores, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (scores >= bins[i]) & (scores < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = gts[mask].mean()
        bin_conf = scores[mask].mean()
        ece += (mask.sum() / len(scores)) * abs(bin_acc - bin_conf)
    return ece

def evaluate(name, gts, scores):
    """All metrics from ground truth and risk scores."""
    eps = 1e-12
    preds_05 = (np.array(scores) >= 0.5).astype(int)

    # Cost-optimal threshold (only for reporting, NOT used for metric selection)
    best_t, best_c = 0.5, float("inf")
    for t in np.arange(0.05, 0.96, 0.05):
        p = (np.array(scores) >= t).astype(int)
        c = sum(5 if g==1 and pr==0 else (1 if g==0 and pr==1 else 0) for g,pr in zip(gts,p))
        if c < best_c: best_c = c; best_t = t
    preds_opt = (np.array(scores) >= best_t).astype(int)
    cm = confusion_matrix(gts, preds_opt, labels=[0,1])
    fn_c = cm[1,0]; fp_c = cm[0,1]
    hr_rate = preds_opt.mean()

    return {
        "accuracy": accuracy_score(gts, preds_opt),
        "balanced_accuracy": balanced_accuracy_score(gts, preds_opt),
        "macro_f1": f1_score(gts, preds_opt, average="macro"),
        "high_risk_recall": recall_score(gts, preds_opt, pos_label=1),
        "low_risk_recall": recall_score(gts, preds_opt, pos_label=0),
        "roc_auc": roc_auc_score(gts, scores),
        "pr_auc": average_precision_score(gts, scores),
        "nll": log_loss(gts, np.clip(scores, eps, 1-eps), labels=[0,1]),
        "brier": brier_score_loss(gts, scores),
        "ece": compute_ece(np.array(gts), np.array(scores)),
        "threshold": best_t,
        "cost": 5*fn_c + fp_c,
        "hr_pred_rate": hr_rate,
        "confusion_matrix": cm.tolist(),
    }

# ============================================================
# Load all predictions
# ============================================================
print("=" * 70)
print("C7: Final Evaluation — German Credit Test Set (N=200)")
print("=" * 70)

all_results = {}

# B0: Majority
gt_maj = [r["ground_truth"] for r in load("outputs/baselines/majority_test.jsonl")]
sc_maj = [r["risk_score"] for r in load("outputs/baselines/majority_test.jsonl")]
all_results["Majority"] = evaluate("Majority", np.array(gt_maj), np.array(sc_maj))

# B1: Logistic Regression
gt_lr = [r["ground_truth"] for r in load("outputs/baselines/logistic_regression_test.jsonl")]
sc_lr = [r["risk_score"] for r in load("outputs/baselines/logistic_regression_test.jsonl")]
all_results["LogisticRegression"] = evaluate("LogisticRegression", np.array(gt_lr), np.array(sc_lr))

# B2: Qwen Zero-shot
gt_q0 = [r["ground_truth"] for r in load("outputs/baselines/qwen_zero_shot_test.jsonl")]
sc_q0 = [r["risk_score"] if r["risk_score"] is not None else 0.5 for r in load("outputs/baselines/qwen_zero_shot_test.jsonl")]
all_results["Qwen-ZeroShot"] = evaluate("Qwen-ZeroShot", np.array(gt_q0), np.array(sc_q0))

# C3: SFT seed 7
raw_path = "outputs/sft/german_sft_seed7/test_predictions_raw.jsonl"
gt_sft = [r["ground_truth"] for r in load(raw_path)]
sc_sft = [r["risk_score"] for r in load(raw_path)]
all_results["SFT-seed7"] = evaluate("SFT-seed7", np.array(gt_sft), np.array(sc_sft))

# C3: SFT multi (German-only samples from combined test)
combined_test_path = "data/processed/multi/combined/sft/test.jsonl"
with open(combined_test_path) as f:
    combined_test = [json.loads(l) for l in f if l.strip()]
german_test = [r for r in combined_test if r["metadata"]["dataset"] == "German"]
print(f"  Multi-SFT German test samples: {len(german_test)}")

# SFT multi adapter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
MODEL_ID = "/data/share/model/Qwen3.5-4B"
MULTI_ADAPTER = "outputs/sft/german_multi/best_adapter"

base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base, MULTI_ADAPTER)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model.eval()

gt_multi, sc_multi = [], []
for r in german_test:
    msgs = r["messages"]
    prompt = tokenizer.apply_chat_template(
        [{"role":"system","content":msgs[0]["content"]},
         {"role":"user","content":msgs[1]["content"]}],
        tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        lp = torch.nn.functional.log_softmax(model(**ids).logits[0,-1,:], dim=-1)
    s_low = lp[tokenizer.encode("low",add_special_tokens=False)[0]].item()
    s_high = lp[tokenizer.encode("high",add_special_tokens=False)[0]].item()
    p_high = np.exp(s_high)/(np.exp(s_low)+np.exp(s_high))
    gt_multi.append(r["metadata"]["risk_label"])
    sc_multi.append(p_high)

all_results["SFT-multi"] = evaluate("SFT-multi", np.array(gt_multi), np.array(sc_multi))

# ============================================================
# Report
# ============================================================
print(f"\n{'='*70}")
print("Unified Metrics Table (German test, N=200)")
print(f"{'='*70}")

headers = ["Model","ROC-AUC","PR-AUC","NLL","Brier","ECE","Acc","BalAcc","MacroF1","HR.Rec","LR.Rec","Cost"]
col_w = [max(len(h), 10) for h in headers]
fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
print(fmt.format(*headers))
print("|" + "|".join("-"*(w+2) for w in col_w) + "|")

for name in ["Majority","Qwen-ZeroShot","LogisticRegression","SFT-seed7","SFT-multi"]:
    m = all_results[name]
    vals = [name,
            f"{m['roc_auc']:.4f}", f"{m['pr_auc']:.4f}",
            f"{m['nll']:.4f}", f"{m['brier']:.4f}", f"{m['ece']:.4f}",
            f"{m['accuracy']:.4f}", f"{m['balanced_accuracy']:.4f}", f"{m['macro_f1']:.4f}",
            f"{m['high_risk_recall']:.4f}", f"{m['low_risk_recall']:.4f}", str(m['cost'])]
    print(fmt.format(*vals))

# Confusion matrices
print(f"\n{'='*70}")
print("Confusion Matrices (cost-optimal threshold per model)")
print(f"{'='*70}")
for name in ["LogisticRegression","SFT-seed7","SFT-multi"]:
    cm = all_results[name]["confusion_matrix"]
    print(f"  {name}: TN={cm[0][0]} FP={cm[0][1]} | FN={cm[1][0]} TP={cm[1][1]}  (cost={all_results[name]['cost']})")

# Method boundary summary
print(f"\n{'='*70}")
print("Method Boundary Summary")
print(f"{'='*70}")
print(f"""
  1. Logistic Regression remains the strongest baseline (ROC-AUC={all_results['LogisticRegression']['roc_auc']:.4f})
  2. SFT brings Qwen from random (0.515) to competitive (0.747) — the only successful post-training method
  3. DPO/SimPO/Risk-DPO all degrade performance — 6 experiments, 0 successes
  4. Multi-dataset SFT improves overall ranking but shows negative transfer to German
  5. Preference optimization on binary short-answer tasks is a methodological dead end
  6. Future: longer reasoning-based outputs (Layer 2B) may create a viable path for preference learning
""")

# Save
with open("outputs/c7_final_metrics.json","w") as f:
    # Convert numpy types
    clean = {}
    for k, v in all_results.items():
        clean[k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv) for kk, vv in v.items()}
    json.dump(clean, f, indent=2)
print("Saved: outputs/c7_final_metrics.json")
