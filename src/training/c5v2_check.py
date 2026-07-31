"""C5v2 implementation check + SFT train inference for hard preference construction."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "/data/share/model/Qwen3.5-4B"
SFT_ADAPTER = "outputs/sft/german_sft_seed7/best_adapter"
PREF_PATH = "data/processed/german/preference/preference_train.jsonl"
SFT_DIR = "data/processed/german/sft"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base, SFT_ADAPTER)
model.eval()
print("Model loaded.")

# ============ Step 1: Sanity check ============
print("\n=== Step 1: Chosen/Rejected direction check ===")
with open(PREF_PATH) as f:
    pref_data = [json.loads(l) for l in f if l.strip()]

def get_answer_logp(prompt_ids, full_ids):
    """Sum log-prob over answer tokens only."""
    with torch.no_grad():
        out = model(input_ids=full_ids.unsqueeze(0))
        shift_logits = out.logits[0, :-1, :]
        shift_labels = full_ids[1:]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(shift_logits, shift_labels)
        mask = torch.zeros(len(shift_labels), device=model.device)
        mask[len(prompt_ids):] = 1.0
        return -(token_loss * mask).sum().item()

for err_type in ["false_negative", "false_positive"]:
    samples = [d for d in pref_data[:200] if d["error_type"] == err_type][:3]
    print(f"\n  {err_type} (gt={samples[0]['metadata']['risk_label']}):")
    for item in samples:
        p_text = tokenizer.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)
        c_text = p_text + item["chosen"][0]["content"] + tokenizer.eos_token
        r_text = p_text + item["rejected"][0]["content"] + tokenizer.eos_token
        p_ids = tokenizer(p_text, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(c_text, truncation=True, max_length=2048)["input_ids"]
        r_ids = tokenizer(r_text, truncation=True, max_length=2048)["input_ids"]
        c_logp = get_answer_logp(torch.tensor(p_ids, device=model.device), torch.tensor(c_ids, device=model.device))
        r_logp = get_answer_logp(torch.tensor(p_ids, device=model.device), torch.tensor(r_ids, device=model.device))
        margin = c_logp - r_logp
        ok = "OK" if margin > 0 else "SWAPPED"
        print(f"    {item['metadata']['sample_id']}: c={c_logp:.2f} r={r_logp:.2f} margin={margin:+.2f} [{ok}]")

# ============ Step 2: SFT inference on all 700 train samples ============
print("\n=== Step 2: SFT seed7 inference on train 700 ===")
with open(f"{SFT_DIR}/train.jsonl") as f:
    train_data = [json.loads(l) for l in f if l.strip()]

results = []
margins = []
for i, r in enumerate(train_data):
    msgs = r["messages"]
    prompt = tokenizer.apply_chat_template(
        [{"role":"system","content":msgs[0]["content"]},
         {"role":"user","content":msgs[1]["content"]}],
        tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0,-1,:]
        lp = torch.nn.functional.log_softmax(logits, dim=-1)
    s_low  = lp[tokenizer.encode("low", add_special_tokens=False)[0]].item()
    s_high = lp[tokenizer.encode("high", add_special_tokens=False)[0]].item()

    gt = r["metadata"]["risk_label"]
    if gt == 1:
        margin = s_high - s_low
    else:
        margin = s_low - s_high

    results.append({
        "sample_id": r["metadata"]["sample_id"],
        "risk_label": gt,
        "logp_low_risk": round(s_low, 6),
        "logp_high_risk": round(s_high, 6),
        "margin": round(margin, 6),
    })
    margins.append(margin)
    if (i+1) % 200 == 0:
        print(f"  {i+1}/700...")

margins = np.array(margins)
ranking_errors = (margins <= 0).sum()
uncertain = ((margins > 0) & (margins < 0.5)).sum()
easy = (margins > 2.0).sum()
print(f"\n  Margin distribution (N=700):")
print(f"    mean={margins.mean():.2f} std={margins.std():.2f}")
print(f"    min={margins.min():.2f} max={margins.max():.2f}")
print(f"    ranking_error (≤0):   {ranking_errors} ({ranking_errors/7:.1f}%)")
print(f"    uncertain (0~0.5):    {uncertain} ({uncertain/7:.1f}%)")
print(f"    moderate (0.5~2.0):   {(margins>0.5).sum() - easy} ({(margins>0.5).mean()-easy/700:.1%})")
print(f"    easy (>2.0):          {easy} ({easy/7:.1f}%)")

# By class
fn_mask = np.array([r["risk_label"]==1 for r in results])
fp_mask = ~fn_mask
print(f"\n  By class:")
print(f"    high_risk (FN): N={fn_mask.sum()}, margin_mean={margins[fn_mask].mean():.2f}, errors={((margins[fn_mask]<=0).sum())}")
print(f"    low_risk (FP):  N={fp_mask.sum()}, margin_mean={margins[fp_mask].mean():.2f}, errors={(margins[fp_mask]<=0).sum()}")

# Save
os.makedirs("outputs/sft/german_sft_seed7", exist_ok=True)
with open("outputs/sft/german_sft_seed7/train_logprobs.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\n  Saved to outputs/sft/german_sft_seed7/train_logprobs.jsonl")

# ============ Step 3: Hard preference construction ============
print(f"\n=== Step 3: Hard Preference Construction ===")

# Keep all ranking errors + bottom 30% of correct samples per class
results_by_class = {"high_risk": [], "low_risk": []}
for r in results:
    cls = "high_risk" if r["risk_label"] == 1 else "low_risk"
    results_by_class[cls].append(r)

hard_samples = set()
# All ranking errors
for r in results:
    if r["margin"] <= 0:
        hard_samples.add(r["sample_id"])

# Bottom 30% by margin from correct samples per class
for cls in ["high_risk", "low_risk"]:
    correct = [r for r in results_by_class[cls] if r["margin"] > 0]
    correct.sort(key=lambda x: x["margin"])
    n_keep = max(int(len(correct) * 0.3), 1)
    for r in correct[:n_keep]:
        hard_samples.add(r["sample_id"])

print(f"  Hard samples: {len(hard_samples)}/{len(results)} ({len(hard_samples)/len(results):.1%})")
print(f"    ranking_errors: {(margins<=0).sum()}")
print(f"    low-margin correct: {len(hard_samples) - (margins<=0).sum()}")

# Build preference pairs
hard_pref = []
for r, item in zip(results, pref_data):
    if r["sample_id"] not in hard_samples:
        continue
    gt = r["risk_label"]
    if gt == 1:
        chosen_text = "high risk"; rejected_text = "low risk"
    else:
        chosen_text = "low risk"; rejected_text = "high risk"

    hard_pref.append({
        "prompt": item["prompt"],
        "chosen": [{"role": "assistant", "content": chosen_text}],
        "rejected": [{"role": "assistant", "content": rejected_text}],
        "metadata": item["metadata"],
        "error_type": "false_negative" if gt == 1 else "false_positive",
        "risk_weight": 5.0 if gt == 1 else 1.0,
        "sft_margin": r["margin"],
        "difficulty": "ranking_error" if r["margin"] <= 0 else "low_confidence",
        "source": "sft_seed7_error_oracle",
        "generation_stage": "sft_seed7",
        "model": "Qwen3.5-4B-SFT-seed7",
    })

hr_count = sum(1 for p in hard_pref if p["metadata"]["risk_label"] == 1)
lr_count = len(hard_pref) - hr_count
print(f"  Constructed {len(hard_pref)} hard preference pairs")
print(f"    high_risk: {hr_count}, low_risk: {lr_count}")

out_dir = "data/processed/german/preference"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/preference_train_hard.jsonl", "w") as f:
    for p in hard_pref:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

import hashlib
with open(f"{out_dir}/preference_train_hard.jsonl", "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
print(f"  SHA-256: {sha}")
print(f"  Saved to {out_dir}/preference_train_hard.jsonl")
