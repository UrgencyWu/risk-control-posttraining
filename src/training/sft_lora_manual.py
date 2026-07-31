"""
C3: LoRA SFT — Manual training loop
Avoids Trainer device management issues by controlling everything directly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, torch, numpy as np
from datetime import datetime, timezone
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader, Dataset as TorchDataset
from src.evaluation.metrics import compute_metrics as eval_compute_metrics, select_cost_threshold

MODEL_ID = os.environ.get("RISK_CONTROL_MODEL_ID", "/data/share/model/Qwen3.5-4B")
SFT_DIR = "data/processed/german/sft"
OUTPUT_DIR = "outputs/sft"

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
# 2× RTX PRO 6000 Blackwell (97GB each), 4B model ≈ 8GB VRAM
BATCH, GRAD_ACCUM = 4, 2     # effective batch = 8 per step, 16 total over 2 GPUs
LR, EPOCHS, MAX_LEN = 2e-4, 5, 2048
SEEDS = [42, 7]  # 10086 done

SYSTEM_PROMPT = (
    "You are a financial risk assessment expert. Evaluate the creditworthiness "
    "based on the customer's financial profile. Classify the risk level as:\n"
    "- low risk: the customer is likely to repay\n"
    "- high risk: the customer is likely to default\n"
    "Respond with only 'low risk' or 'high risk'."
)

class SFTHFDataset(TorchDataset):
    def __init__(self, split, tokenizer):
        self.tokenizer = tokenizer
        with open(f"{SFT_DIR}/{split}.jsonl") as f:
            self.data = [json.loads(l) for l in f if l.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        msgs = self.data[idx]["messages"]
        prompt = self.tokenizer.apply_chat_template(
            [{"role":"system","content":msgs[0]["content"]},
             {"role":"user","content":msgs[1]["content"]}],
            tokenize=False, add_generation_prompt=True)
        completion = msgs[2]["content"]
        full = prompt + completion + self.tokenizer.eos_token
        # Tokenize full and prompt separately to mask prompt tokens
        full_ids = self.tokenizer(full, truncation=True, max_length=MAX_LEN)["input_ids"]
        prompt_ids = self.tokenizer(prompt, truncation=True, max_length=MAX_LEN, add_special_tokens=False)["input_ids"]
        labels = [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]
        return {
            "input_ids": full_ids,
            "labels": labels,
        }

def collate(examples, tokenizer):
    max_len = max(len(e["input_ids"]) for e in examples)
    input_ids, labels, masks = [], [], []
    for e in examples:
        pad = max_len - len(e["input_ids"])
        input_ids.append(e["input_ids"] + [tokenizer.pad_token_id]*pad)
        labels.append(e["labels"] + [-100]*pad)
        masks.append([1]*(len(e["input_ids"])) + [0]*pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(masks),
    }

def train_seed(seed):
    run_dir = f"{OUTPUT_DIR}/german_sft_seed{seed}"
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    # Slurm sets CUDA_VISIBLE_DEVICES; device_map="auto" handles the rest
    device = torch.device("cuda")

    print(f"\n  Seed {seed} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train_ds = SFTHFDataset("train", tokenizer)
    valid_ds = SFTHFDataset("valid", tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate(b, tokenizer))
    valid_dl = DataLoader(valid_ds, batch_size=BATCH, collate_fn=lambda b: collate(b, tokenizer))

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    best_loss, best_step = float("inf"), 0
    train_log = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss, step = 0.0, 0
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item()

            if (i+1) % GRAD_ACCUM == 0:
                opt.step(); opt.zero_grad()
                step += 1

                if step % 10 == 0:
                    # Validation
                    model.eval()
                    v_loss = 0.0
                    with torch.no_grad():
                        for vb in valid_dl:
                            vb = {k: v.to(device) for k, v in vb.items()}
                            v_loss += model(**vb).loss.item()
                    v_loss /= len(valid_dl)
                    train_log.append({"epoch": epoch, "step": step, "train_loss": round(total_loss/step, 4), "valid_loss": round(v_loss, 4)})
                    print(f"    epoch={epoch} step={step} train_loss={total_loss/step:.4f} valid_loss={v_loss:.4f}")
                    if v_loss < best_loss:
                        best_loss = v_loss; best_step = step
                        model.save_pretrained(f"{run_dir}/best_adapter")
                        tokenizer.save_pretrained(f"{run_dir}/best_adapter")
                    model.train()

    print(f"  Best: step={best_step}, valid_loss={best_loss:.4f}")

    # Reload best adapter for inference
    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, f"{run_dir}/best_adapter")

    # Save log
    with open(f"{run_dir}/training_log.json", "w") as f:
        json.dump({"best_step": best_step, "best_valid_loss": round(best_loss, 4), "log": train_log}, f, indent=2)

    return model, tokenizer, run_dir

def run_inference(model, tokenizer, split, path):
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
        s_low = lp[tokenizer.encode("low",add_special_tokens=False)[0]].item()
        s_high = lp[tokenizer.encode("high",add_special_tokens=False)[0]].item()
        p_high = np.exp(s_high)/(np.exp(s_low)+np.exp(s_high))
        gt = r["metadata"]["risk_label"]
        pred = 1 if p_high >= 0.5 else 0
        err = None
        if gt==1 and pred==0: err = "false_negative"
        elif gt==0 and pred==1: err = "false_positive"
        results.append({"sample_id":r["metadata"]["sample_id"],"ground_truth":gt,"prediction":pred,
                        "risk_score":round(p_high,6),"threshold":0.5,"error_type":err,
                        "cost":5 if err=="false_negative" else (1 if err=="false_positive" else 0),
                        "model":"Qwen3.5-4B-SFT"})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w") as f:
        for r in results: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    return results

def find_best_threshold(results):
    scores = np.array([r["risk_score"] for r in results])
    gts = np.array([r["ground_truth"] for r in results])
    return select_cost_threshold(scores, gts)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}
    for seed in SEEDS:
        model, tok, run_dir = train_seed(seed)
        vp = run_inference(model, tok, "valid", f"{run_dir}/valid_predictions.jsonl")
        bt, bc = find_best_threshold(vp)
        print(f"  Best threshold (valid): {bt:.2f}, cost={bc}")
        tp = run_inference(model, tok, "test", f"{run_dir}/test_predictions_raw.jsonl")
        # Apply threshold
        adjusted = []
        for r in tp:
            pred = 1 if r["risk_score"] >= bt else 0
            gt = r["ground_truth"]
            err = None
            if gt==1 and pred==0: err="false_negative"
            elif gt==0 and pred==1: err="false_positive"
            adjusted.append({**r,"prediction":pred,"threshold":round(bt,2),"error_type":err,
                            "cost":5 if err=="false_negative" else (1 if err=="false_positive" else 0)})
        with open(f"{run_dir}/test_predictions.jsonl","w") as f:
            for r in adjusted: f.write(json.dumps(r,ensure_ascii=False)+"\n")
        gt = [r["ground_truth"] for r in adjusted]
        yp = [r["prediction"] for r in adjusted]
        sc = [r["risk_score"] for r in adjusted]
        m = eval_compute_metrics(gt, yp, sc)
        all_results[f"SFT_seed{seed}"] = m
        print(f"  Test: acc={m['accuracy']:.4f}, roc_auc={m['roc_auc']}, cost={m['cost']}")
        del model; torch.cuda.empty_cache()

    avg = {k: round(np.mean([all_results[f"SFT_seed{s}"][k] for s in SEEDS]),4)
           for k in ["accuracy","roc_auc","high_risk_recall","cost"]}
    print(f"\n  Avg: acc={avg['accuracy']}, roc_auc={avg['roc_auc']}, hr_recall={avg['high_risk_recall']}, cost={avg['cost']}")
    with open(f"{OUTPUT_DIR}/sft_results.json","w") as f:
        json.dump({"model":"Qwen3.5-4B-SFT","per_seed":all_results,"average":avg,
                   "timestamp":datetime.now(timezone.utc).isoformat()}, f, indent=2)

if __name__ == "__main__":
    main()
