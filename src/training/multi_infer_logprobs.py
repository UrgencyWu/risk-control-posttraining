"""
C5v3-A.1: Multi-dataset SFT inference — compute logprobs + margins for train/valid.
Dry-run mode: --dry_run 10 for human inspection.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, torch, numpy as np, argparse, hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = os.environ.get("RISK_CONTROL_MODEL_ID", "/data/share/model/Qwen3.5-4B")
ADAPTER_DIR = "outputs/sft/german_multi/best_adapter"
SFT_DIR = "data/processed/multi/combined/sft"
OUT_DIR = "outputs/preference_inference"

def load_model():
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    return model, tokenizer

def run_inference(model, tokenizer, split, max_samples=None):
    path = f"{SFT_DIR}/{split}.jsonl"
    with open(path) as f:
        data = [json.loads(l) for l in f if l.strip()]

    if max_samples:
        data = data[:max_samples]
        print(f"  DRY-RUN: limiting to {len(data)} samples")

    results = []
    for i, r in enumerate(data):
        msgs = r["messages"]
        prompt = tokenizer.apply_chat_template(
            [{"role":"system","content":msgs[0]["content"]},
             {"role":"user","content":msgs[1]["content"]}],
            tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
            lp = torch.nn.functional.log_softmax(logits, dim=-1)

        low_id = tokenizer.encode("low", add_special_tokens=False)[0]
        high_id = tokenizer.encode("high", add_special_tokens=False)[0]

        s_low = lp[low_id].item()
        s_high = lp[high_id].item()

        gt = r["metadata"]["risk_label"]
        if gt == 1:  # high risk
            gt_margin = s_high - s_low
            correct_logp = s_high
            wrong_logp = s_low
        else:  # low risk
            gt_margin = s_low - s_high
            correct_logp = s_low
            wrong_logp = s_high

        # Prediction at threshold 0.5
        p_high = np.exp(s_high) / (np.exp(s_low) + np.exp(s_high))
        pred = 1 if p_high >= 0.5 else 0
        is_correct = (pred == gt)

        results.append({
            "sample_id": r["metadata"]["sample_id"],
            "dataset": r["metadata"]["dataset"],
            "split": split,
            "risk_label": gt,
            "logp_low_risk": round(s_low, 6),
            "logp_high_risk": round(s_high, 6),
            "p_high_risk": round(p_high, 6),
            "correct_logp": round(correct_logp, 6),
            "wrong_logp": round(wrong_logp, 6),
            "gt_margin": round(gt_margin, 6),
            "prediction": pred,
            "is_correct": is_correct,
        })

        if (i+1) % 200 == 0:
            print(f"    {split}: {i+1}/{len(data)}...")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", type=int, default=0)
    args = parser.parse_args()

    print(f"C5v3-A.1: Multi-dataset SFT Inference")
    print(f"  Model: {ADAPTER_DIR}")
    if args.dry_run:
        print(f"  MODE: DRY-RUN ({args.dry_run} samples)")

    model, tokenizer = load_model()
    print("  Model loaded.")

    for split in ["train", "valid"]:
        results = run_inference(model, tokenizer, split, max_samples=args.dry_run or None)

        # Distribution stats
        margins = np.array([r["gt_margin"] for r in results])
        correct = sum(1 for r in results if r["is_correct"])
        acc = correct / len(results) if results else 0

        print(f"\n  {split}: N={len(results)}  acc@0.5={acc:.3f}")
        print(f"    margin: mean={margins.mean():.3f} std={margins.std():.3f}  min={margins.min():.3f} max={margins.max():.3f}")
        print(f"    margin≤0: {(margins<=0).sum()} ({(margins<=0).mean()*100:.1f}%)")
        print(f"    margin 0-0.5: {((margins>0)&(margins<0.5)).sum()}")

        # Per-dataset
        for ds in sorted(set(r["dataset"] for r in results)):
            ds_results = [r for r in results if r["dataset"]==ds]
            ds_margins = np.array([r["gt_margin"] for r in ds_results])
            ds_correct = sum(1 for r in ds_results if r["is_correct"])
            print(f"    {ds}: N={len(ds_results)} acc={ds_correct/len(ds_results):.3f} margin_mean={ds_margins.mean():.3f} errors={(ds_margins<=0).sum()}")

        # Per-class
        for cls_name, cls_val in [("low_risk", 0), ("high_risk", 1)]:
            cls_results = [r for r in results if r["risk_label"]==cls_val]
            cls_margins = np.array([r["gt_margin"] for r in cls_results])
            print(f"    {cls_name}: N={len(cls_results)} margin_mean={cls_margins.mean():.3f} errors={(cls_margins<=0).sum()}")

        # Dry-run: show samples
        if args.dry_run and split == "train":
            print(f"\n  --- Dry-run samples ---")
            for r in results:
                direction = "✓" if r["gt_margin"] > 0 else "✗ SWAPPED"
                print(f"    {r['sample_id']} ({r['dataset']}) gt={r['risk_label']} "
                      f"correct_logp={r['correct_logp']:.3f} wrong_logp={r['wrong_logp']:.3f} "
                      f"margin={r['gt_margin']:+.3f} pred={'high' if r['prediction']==1 else 'low'} "
                      f"correct={r['is_correct']} [{direction}]")

        if not args.dry_run:
            os.makedirs(OUT_DIR, exist_ok=True)
            out_path = f"{OUT_DIR}/{split}_predictions.jsonl"
            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"    Saved: {out_path}")

    if not args.dry_run:
        # Manifest
        manifest = {
            "model": ADAPTER_DIR,
            "data": SFT_DIR,
            "train_samples": sum(1 for r in results if r["split"]=="train") if 'results' in dir() else "see files",
        }
        with open(f"{OUT_DIR}/train_predictions.jsonl", "rb") as f:
            manifest["train_sha256"] = hashlib.sha256(f.read()).hexdigest()
        with open(f"{OUT_DIR}/valid_predictions.jsonl", "rb") as f:
            manifest["valid_sha256"] = hashlib.sha256(f.read()).hexdigest()
        with open(f"{OUT_DIR}/manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n  Manifest: {OUT_DIR}/manifest.json")

if __name__ == "__main__":
    main()
