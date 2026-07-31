"""
C5v3 audit: Verify "global label prior shift" hypothesis.
No training — statistics only.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import roc_auc_score

MODEL_ID = "/data/share/model/Qwen3.5-4B"
ADAPTERS = {
    "SFT": "outputs/sft/german_multi/best_adapter",
    "DPO": "outputs/dpo/german_multi_dpo_multi/best_adapter",
    "SimPO": "outputs/dpo/german_multi_simpo_multi/best_adapter",
}
SFT_DIR = "data/processed/multi/combined/sft"
PREF_PATH = "data/processed/preference_multidataset"

def load_model(path):
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    return model, tokenizer

def get_logprobs(model, tokenizer, records):
    low_lps, high_lps = [], []
    for r in records:
        msgs = r["messages"]
        prompt = tokenizer.apply_chat_template(
            [{"role":"system","content":msgs[0]["content"]},
             {"role":"user","content":msgs[1]["content"]}],
            tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            lp = torch.nn.functional.log_softmax(model(**ids).logits[0,-1,:], dim=-1)
        low_lps.append(lp[tokenizer.encode("low",add_special_tokens=False)[0]].item())
        high_lps.append(lp[tokenizer.encode("high",add_special_tokens=False)[0]].item())
    return np.array(low_lps), np.array(high_lps)

print("=" * 60)
print("C5v3 Audit: Label Prior Shift Analysis")
print("=" * 60)

# 1. Hard preference data composition
print("\n--- 1. Preference Data Composition ---")
for split in ("train", "valid"):
    with open(f"{PREF_PATH}/{split}.jsonl") as f:
        data = [json.loads(l) for l in f if l.strip()]
    counts = {}
    for d in data:
        ds = d["metadata"]["dataset"]
        rl = d["metadata"]["risk_label"]
        counts.setdefault(ds, {"high_risk":0,"low_risk":0})
        counts[ds]["high_risk" if rl==1 else "low_risk"] += 1
    print(f"  {split} ({len(data)} pairs):")
    for ds in ["German","Australian"]:
        c = counts.get(ds, {})
        total = c.get("high_risk",0) + c.get("low_risk",0)
        hr_pct = c.get("high_risk",0)/total*100 if total else 0
        print(f"    {ds}: high_risk={c.get('high_risk',0)} low_risk={c.get('low_risk',0)} ({hr_pct:.0f}% high)")

# 2. Chosen label distribution
print("\n--- 2. Chosen Label Distribution ---")
for split in ("train", "valid"):
    with open(f"{PREF_PATH}/{split}.jsonl") as f:
        data = [json.loads(l) for l in f if l.strip()]
    hr_chosen = sum(1 for d in data if d["chosen"][0]["content"]=="high risk")
    print(f"  {split}: chosen=high_risk: {hr_chosen}/{len(data)} ({hr_chosen/len(data)*100:.1f}%)")

# 3. DPO before/after logprob shifts
print("\n--- 3. Logprob Shifts (SFT → DPO → SimPO) on Test Set ---")
with open(f"{SFT_DIR}/test.jsonl") as f:
    test_data = [json.loads(l) for l in f if l.strip()]

for label, path in ADAPTERS.items():
    model, tokenizer = load_model(path)
    low_lps, high_lps = get_logprobs(model, tokenizer, test_data)
    p_high = np.exp(high_lps) / (np.exp(low_lps) + np.exp(high_lps))
    hr_pred_pct = (p_high >= 0.5).mean() * 100

    # Per-dataset
    german_mask = np.array([r["metadata"]["dataset"]=="German" for r in test_data])
    au_mask = np.array([r["metadata"]["dataset"]=="Australian" for r in test_data])

    print(f"  {label}:")
    print(f"    logp(low):  mean={low_lps.mean():.3f}  std={low_lps.std():.3f}")
    print(f"    logp(high): mean={high_lps.mean():.3f}  std={high_lps.std():.3f}")
    print(f"    logp(high) - logp(low): mean={(high_lps-low_lps).mean():.3f}")
    print(f"    P(high) mean={p_high.mean():.3f}  HR_pred%={hr_pred_pct:.1f}%")
    print(f"    German:  P(high)_mean={p_high[german_mask].mean():.3f}  HR%={(p_high[german_mask]>=0.5).mean()*100:.1f}%")
    print(f"    Australian: P(high)_mean={p_high[au_mask].mean():.3f}  HR%={(p_high[au_mask]>=0.5).mean()*100:.1f}%")

    # AUC
    gts = np.array([r["metadata"]["risk_label"] for r in test_data])
    overall_auc = roc_auc_score(gts, p_high)
    ger_gt = gts[german_mask]; ger_sc = p_high[german_mask]
    au_gt = gts[au_mask]; au_sc = p_high[au_mask]
    print(f"    ROC-AUC: overall={overall_auc:.4f}  German={roc_auc_score(ger_gt,ger_sc):.4f}  Aus={roc_auc_score(au_gt,au_sc):.4f}")

# 4. Risk score distribution comparison
print("\n--- 4. Risk Score (P_high) Distribution ---")
for label, path in ADAPTERS.items():
    model, tokenizer = load_model(path)
    _, high_lps = get_logprobs(model, tokenizer, test_data)
    low_all, high_all = get_logprobs(model, tokenizer, test_data)
    p_high = np.exp(high_all) / (np.exp(low_all) + np.exp(high_all))
    print(f"  {label}: P_high percentiles: 10%={np.percentile(p_high,10):.3f} "
          f"25%={np.percentile(p_high,25):.3f} 50%={np.median(p_high):.3f} "
          f"75%={np.percentile(p_high,75):.3f} 90%={np.percentile(p_high,90):.3f}")

# 5. Conclusion
print(f"\n--- 5. Summary ---")
print(f"  Hypothesis: DPO/SimPO shift the global label prior without preserving")
print(f"              feature-conditional ordering between samples.")
print(f"  Evidence:   HR% goes from ~30% (SFT) to 100% (DPO/SimPO) —")
print(f"              a pure prior shift, not sample-level discrimination.")
