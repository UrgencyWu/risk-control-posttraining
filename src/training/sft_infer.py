"""C3: SFT inference — LoRA evaluation with first-token class scoring."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.evaluation.metrics import compute_metrics, generate_metrics_table, select_cost_threshold

MODEL_ID = os.environ.get("RISK_CONTROL_MODEL_ID", "/data/share/model/Qwen3.5-4B")
ADAPTER_DIR = "outputs/sft/german_sft_seed10086/best_adapter"
SFT_DIR = "data/processed/german/sft"
OUT_DIR = "outputs/sft/german_sft_seed10086"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print("Model loaded.")

best_t = 0.5
for split in ("valid", "test"):
    with open(f"{SFT_DIR}/{split}.jsonl") as f:
        data = [json.loads(l) for l in f if l.strip()]
    results = []
    for i, r in enumerate(data):
        msgs = r["messages"]
        prompt = tokenizer.apply_chat_template(
            [{"role":"system","content":msgs[0]["content"]},
             {"role":"user","content":msgs[1]["content"]}],
            tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0,-1,:]
            lp = torch.nn.functional.log_softmax(logits, dim=-1)
        # This compares the first token of the two labels, not full-sequence likelihood.
        s_low  = lp[tokenizer.encode("low", add_special_tokens=False)[0]].item()
        s_high = lp[tokenizer.encode("high", add_special_tokens=False)[0]].item()
        p_high = np.exp(s_high)/(np.exp(s_low)+np.exp(s_high))
        gt = r["metadata"]["risk_label"]
        pred = 1 if p_high >= 0.5 else 0
        err = None
        if gt==1 and pred==0: err="false_negative"
        elif gt==0 and pred==1: err="false_positive"
        results.append({"sample_id":r["metadata"]["sample_id"],"ground_truth":gt,"prediction":pred,
                        "risk_score":round(p_high,6),"threshold":0.5,"error_type":err,
                        "cost":5 if err=="false_negative" else (1 if err=="false_positive" else 0),
                        "model":"Qwen3.5-4B-SFT"})
        if (i+1) % 100 == 0:
            print(f"  {split}: {i+1}/{len(data)}")

    # Best threshold on valid
    if split == "valid":
        scores = np.array([r["risk_score"] for r in results])
        gts = np.array([r["ground_truth"] for r in results])
        best_t, best_c = select_cost_threshold(scores, gts)
        print(f"  Best threshold (valid): {best_t:.2f}, cost={best_c}")

    # Apply threshold for test
    if split == "test":
        for r in results:
            r["prediction"] = 1 if r["risk_score"] >= best_t else 0
            gt = r["ground_truth"]
            err = None
            if gt==1 and r["prediction"]==0: err="false_negative"
            elif gt==0 and r["prediction"]==1: err="false_positive"
            r["threshold"] = round(best_t,2)
            r["error_type"] = err
            r["cost"] = 5 if err=="false_negative" else (1 if err=="false_positive" else 0)

    with open(f"{OUT_DIR}/{split}_predictions.jsonl","w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

# Compute metrics
with open(f"{OUT_DIR}/test_predictions.jsonl") as f:
    test_preds = [json.loads(l) for l in f if l.strip()]
gt = [r["ground_truth"] for r in test_preds]
yp = [r["prediction"] for r in test_preds]
sc = [r["risk_score"] for r in test_preds]
m = compute_metrics(gt, yp, sc)
print(f"\nTest: acc={m['accuracy']:.4f} balanced={m['balanced_accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
print(f"      high_risk_recall={m['high_risk_recall']:.4f} roc_auc={m['roc_auc']} cost={m['cost']}")
print(f"\n{generate_metrics_table({'Qwen3.5-4B-SFT': m})}")
