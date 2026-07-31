"""
C4: Construct preference data for DPO / SimPO / Risk-DPO.
Uses ground-truth labels to build oracle chosen/rejected pairs.
All DPO variants share exactly the same samples.
"""
import json, os

SFT_DIR = "data/processed/german/sft"
OUT_DIR = "data/processed/german/preference"
os.makedirs(OUT_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a financial risk assessment expert. Evaluate the creditworthiness "
    "based on the customer's financial profile. Classify the risk level as:\n"
    "- low risk: the customer is likely to repay\n"
    "- high risk: the customer is likely to default\n"
    "Respond with only 'low risk' or 'high risk'."
)

pref_data = []
label_dist = {"false_negative": 0, "false_positive": 0}

for split in ("train", "valid"):
    path = f"{SFT_DIR}/{split}.jsonl"
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    for r in records:
        msgs = r["messages"]
        risk_label = r["metadata"]["risk_label"]

        # Build prompt (system + user, no assistant)
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msgs[1]["content"]},
        ]

        if risk_label == 1:  # high risk
            chosen   = [{"role": "assistant", "content": "high risk"}]
            rejected = [{"role": "assistant", "content": "low risk"}]
            error_type = "false_negative"
            risk_weight = 5.0
        else:  # risk_label == 0, low risk
            chosen   = [{"role": "assistant", "content": "low risk"}]
            rejected = [{"role": "assistant", "content": "high risk"}]
            error_type = "false_positive"
            risk_weight = 1.0

        label_dist[error_type] += 1

        pref_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "sample_id": r["metadata"]["sample_id"],
                "dataset": r["metadata"]["dataset"],
                "split": r["metadata"]["split"],
                "task_type": r["metadata"]["task_type"],
                "risk_label": risk_label,
                "original_label": r["metadata"]["original_label"],
            },
            "error_type": error_type,
            "risk_weight": risk_weight,
            "source": "ground_truth_oracle",
            "generation_stage": "oracle",
            "model": "ground_truth",
        })

# Save
out_path = f"{OUT_DIR}/preference_train.jsonl"
with open(out_path, "w") as f:
    for item in pref_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

train_count = sum(1 for p in pref_data if p["metadata"]["split"] == "train")
valid_count = sum(1 for p in pref_data if p["metadata"]["split"] == "valid")
print(f"Preference data: {len(pref_data)} pairs (train={train_count}, valid={valid_count})")
print(f"  false_negative (weight=5): {label_dist['false_negative']}")
print(f"  false_positive (weight=1): {label_dist['false_positive']}")
print(f"  Saved to {out_path}")

# Sample
print("\nSample:")
print(json.dumps(pref_data[0], indent=2, ensure_ascii=False)[:500])
print("...")
print(json.dumps(pref_data[-1], indent=2, ensure_ascii=False)[:500])
