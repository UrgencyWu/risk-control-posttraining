"""
C6v2: Cost-sensitive SFT (weighted cross-entropy baseline)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, torch, numpy as np, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

MODEL_ID = "/data/share/model/Qwen3.5-4B"
SFT_ADAPTER = "outputs/sft/german_sft_seed7/best_adapter"
SFT_DIR = "data/processed/german/sft"
OUT_BASE = "outputs/dpo"

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
BATCH, GRAD_ACCUM = 2, 4
LR, EPOCHS, MAX_LEN = 5e-6, 2, 2048
SEED = 42

class CSDataset(Dataset):
    def __init__(self, split, tokenizer, fn_weight=5.0, fp_weight=1.0):
        self.tokenizer = tokenizer
        self.fn_w = fn_weight; self.fp_w = fp_weight
        with open(f"{SFT_DIR}/{split}.jsonl") as f:
            self.data = [json.loads(l) for l in f if l.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        msgs = r["messages"]
        gt = r["metadata"]["risk_label"]
        prompt = self.tokenizer.apply_chat_template(
            [{"role":"system","content":msgs[0]["content"]},
             {"role":"user","content":msgs[1]["content"]}],
            tokenize=False, add_generation_prompt=True)
        full = prompt + msgs[2]["content"] + self.tokenizer.eos_token
        ids = self.tokenizer(full, truncation=True, max_length=MAX_LEN)["input_ids"]
        p_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        labels = [-100]*len(p_ids) + ids[len(p_ids):]
        w = self.fn_w if gt == 1 else self.fp_w
        return {"input_ids": ids, "labels": labels, "weight": w}

def collate(batch, tokenizer):
    def pad(seqs, val):
        m = max(len(s) for s in seqs)
        return [s + [val]*(m-len(s)) for s in seqs]
    return {
        "input_ids": torch.tensor(pad([b["input_ids"] for b in batch], tokenizer.pad_token_id)),
        "labels": torch.tensor(pad([b["labels"] for b in batch], -100)),
        "attention_mask": torch.tensor(pad([[1]*len(b["input_ids"]) for b in batch], 0)),
        "weights": torch.tensor([b["weight"] for b in batch]),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn_weight", type=float, required=True)
    parser.add_argument("--fp_weight", type=float, required=True)
    args = parser.parse_args()

    tag = f"fn{int(args.fn_weight)}_fp{int(args.fp_weight)}"
    run_dir = f"{OUT_BASE}/german_cost_sft_{tag}"
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"C6v2: Cost-sensitive SFT (FN={int(args.fn_weight)}, FP={int(args.fp_weight)})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, SFT_ADAPTER)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                              target_modules=LORA_TARGET, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train_ds = CSDataset("train", tokenizer, args.fn_weight, args.fp_weight)
    valid_ds = CSDataset("valid", tokenizer, args.fn_weight, args.fp_weight)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate(b, tokenizer))
    valid_dl = DataLoader(valid_ds, batch_size=BATCH, collate_fn=lambda b: collate(b, tokenizer))

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    best_loss, best_step = float("inf"), 0
    train_log = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss, step = 0.0, 0
        for i, batch in enumerate(train_dl):
            device = next(model.parameters()).device
            ids = batch["input_ids"].to(device); labels = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device); weights = batch["weights"].to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            # Per-sample weighted loss
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            tloss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            tloss = tloss.view(shift_labels.shape)
            label_mask = (shift_labels != -100).float()
            per_sample_loss = (tloss * label_mask).sum(dim=-1) / label_mask.sum(dim=-1).clamp(min=1)
            loss = (per_sample_loss * weights).mean() / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item()

            if (i+1) % GRAD_ACCUM == 0:
                opt.step(); opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    model.eval()
                    v_loss = 0.0
                    with torch.no_grad():
                        for vb in valid_dl:
                            v_ids = vb["input_ids"].to(device); v_labels = vb["labels"].to(device)
                            v_mask = vb["attention_mask"].to(device); v_weights = vb["weights"].to(device)
                            v_out = model(input_ids=v_ids, attention_mask=v_mask, labels=v_labels)
                            vs_logits = v_out.logits[..., :-1, :].contiguous()
                            vs_labels = v_labels[..., 1:].contiguous()
                            v_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                            v_tloss = v_loss_fct(vs_logits.view(-1, vs_logits.size(-1)), vs_labels.view(-1))
                            v_tloss = v_tloss.view(vs_labels.shape)
                            v_lmask = (vs_labels != -100).float()
                            v_per = (v_tloss * v_lmask).sum(dim=-1) / v_lmask.sum(dim=-1).clamp(min=1)
                            v_loss += (v_per * v_weights).mean().item()
                    v_loss /= max(len(valid_dl), 1)
                    train_log.append({"epoch":epoch,"step":step,"train_loss":round(total_loss/step,4),"valid_loss":round(v_loss,4)})
                    print(f"    e={epoch} s={step} train_loss={total_loss/step:.4f} valid_loss={v_loss:.4f}")
                    if v_loss < best_loss:
                        best_loss = v_loss; best_step = step
                        model.save_pretrained(f"{run_dir}/best_adapter")
                        tokenizer.save_pretrained(f"{run_dir}/best_adapter")
                    model.train()

    print(f"  Best: step={best_step}, loss={best_loss:.4f}")
    with open(f"{run_dir}/training_log.json","w") as f:
        json.dump({"fn":args.fn_weight,"fp":args.fp_weight,"best_step":best_step,"best_loss":round(best_loss,4),"log":train_log}, f, indent=2)

if __name__ == "__main__":
    main()
