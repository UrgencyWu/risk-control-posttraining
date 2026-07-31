"""C5 inference: evaluate DPO/SimPO adapters on valid+test sets."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, torch, numpy as np, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.evaluation.metrics import compute_metrics

MODEL_ID = "/data/share/model/Qwen3.5-4B"
SFT_ADAPTER = "outputs/sft/german_sft_seed7/best_adapter"
SFT_DIR = "data/processed/german/sft"

def load_model(adapter_dir):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, tokenizer

def run_inference(model, tokenizer, split, out_path):
    with open(f"{SFT_DIR}/{split}.jsonl") as f:
        data = [json.loads(l) for l in f if l.strip()]
    results = []
    for r in data:
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
        p_high = np.exp(s_high)/(np.exp(s_low)+np.exp(s_high))
        gt = r["metadata"]["risk_label"]
        pred = 1 if p_high >= 0.5 else 0
        err = None
        if gt==1 and pred==0: err="false_negative"
        elif gt==0 and pred==1: err="false_positive"
        results.append({"sample_id":r["metadata"]["sample_id"],"ground_truth":gt,"prediction":pred,
                        "risk_score":round(p_high,6),"threshold":0.5,"error_type":err,
                        "cost":5 if err=="false_negative" else (1 if err=="false_positive" else 0)})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    return results

def find_best_threshold(results):
    scores = np.array([r["risk_score"] for r in results])
    gts = np.array([r["ground_truth"] for r in results])
    best_t, best_c = 0.5, float("inf")
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (scores >= t).astype(int)
        c = sum(5 if g==1 and p==0 else (1 if g==0 and p==1 else 0) for g,p in zip(gts,preds))
        if c < best_c: best_c = c; best_t = t
    return best_t, best_c

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sft_adapter", default=SFT_ADAPTER)
    args = parser.parse_args()

    print(f"Loading adapter: {args.adapter}")
    model, tokenizer = load_model(args.adapter)

    # Valid inference + threshold
    vp = run_inference(model, tokenizer, "valid", f"{args.output}/valid_predictions.jsonl")
    best_t, best_c = find_best_threshold(vp)
    print(f"Valid: best_threshold={best_t:.2f}, cost={best_c}")

    # Test inference with optimal threshold
    tp = run_inference(model, tokenizer, "test", f"{args.output}/test_predictions_raw.jsonl")
    adjusted = []
    for r in tp:
        pred = 1 if r["risk_score"] >= best_t else 0
        gt = r["ground_truth"]
        err = None
        if gt==1 and pred==0: err="false_negative"
        elif gt==0 and pred==1: err="false_positive"
        adjusted.append({**r, "prediction":pred, "threshold":round(best_t,2), "error_type":err,
                        "cost":5 if err=="false_negative" else (1 if err=="false_positive" else 0)})
    with open(f"{args.output}/test_predictions.jsonl","w") as f:
        for r in adjusted: f.write(json.dumps(r, ensure_ascii=False)+"\n")

    gt = [r["ground_truth"] for r in adjusted]
    yp = [r["prediction"] for r in adjusted]
    sc = [r["risk_score"] for r in adjusted]
    m = compute_metrics(gt, yp, sc)
    print(f"Test: acc={m['accuracy']:.4f} balanced={m['balanced_accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
    print(f"      high_risk_recall={m['high_risk_recall']:.4f} roc_auc={m['roc_auc']} cost={m['cost']}")

if __name__ == "__main__":
    main()
