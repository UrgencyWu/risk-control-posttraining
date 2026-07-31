"""
C6v2: Anchored Risk-DPO
L = mean(w̃_i × L_DPO,i) + λ × L_SFT
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, torch, numpy as np, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

MODEL_ID = "/data/share/model/Qwen3.5-4B"
SFT_ADAPTER = "outputs/sft/german_sft_seed7/best_adapter"
PREF_PATH = "data/processed/german/preference/preference_train_hard.jsonl"
OUT_BASE = "outputs/dpo"

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
BATCH, GRAD_ACCUM = 1, 8
LR, EPOCHS, MAX_LEN = 5e-6, 2, 2048
BETA = 0.1
LAMBDA = 1.0
SEED = 42

class PrefDataset(Dataset):
    def __init__(self, split, tokenizer, fn_weight=5.0, fp_weight=1.0):
        self.tokenizer = tokenizer
        with open(PREF_PATH) as f:
            all_data = [json.loads(l) for l in f if l.strip()]
        self.data = [d for d in all_data if d["metadata"]["split"] == split]
        # Compute normalized weights
        raw_weights = np.array([fn_weight if d["error_type"]=="false_negative" else fp_weight for d in self.data])
        mean_w = raw_weights.mean()
        self.norm_weights = (raw_weights / mean_w).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_text = self.tokenizer.apply_chat_template(
            item["prompt"], tokenize=False, add_generation_prompt=True)
        c_text = prompt_text + item["chosen"][0]["content"] + self.tokenizer.eos_token
        r_text = prompt_text + item["rejected"][0]["content"] + self.tokenizer.eos_token
        p_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        c_ids = self.tokenizer(c_text, truncation=True, max_length=MAX_LEN)["input_ids"]
        r_ids = self.tokenizer(r_text, truncation=True, max_length=MAX_LEN)["input_ids"]
        c_labels = [-100]*len(p_ids) + c_ids[len(p_ids):]
        r_labels = [-100]*len(p_ids) + r_ids[len(p_ids):]
        return {"c_ids": c_ids, "c_labels": c_labels, "r_ids": r_ids, "r_labels": r_labels,
                "weight": self.norm_weights[idx], "p_len": len(p_ids)}

def collate(batch, tokenizer):
    def pad(seqs, val):
        m = max(len(s) for s in seqs)
        return [s + [val]*(m-len(s)) for s in seqs]
    return {
        "c_ids": torch.tensor(pad([b["c_ids"] for b in batch], tokenizer.pad_token_id)),
        "c_labels": torch.tensor(pad([b["c_labels"] for b in batch], -100)),
        "c_mask": torch.tensor(pad([[1]*len(b["c_ids"]) for b in batch], 0)),
        "r_ids": torch.tensor(pad([b["r_ids"] for b in batch], tokenizer.pad_token_id)),
        "r_labels": torch.tensor(pad([b["r_labels"] for b in batch], -100)),
        "r_mask": torch.tensor(pad([[1]*len(b["r_ids"]) for b in batch], 0)),
        "weights": torch.tensor([b["weight"] for b in batch]),
    }

def seq_logprob(model, ids, labels, mask):
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    shift_logits = out.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    tloss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.shape)
    m = (shift_labels != -100).float()
    return -(tloss * m).sum(dim=-1)  # logprob per sample

def dpo_loss_per_sample(model, ref_model, batch, beta):
    device = next(model.parameters()).device
    c_ids = batch["c_ids"].to(device); c_labels = batch["c_labels"].to(device); c_mask = batch["c_mask"].to(device)
    r_ids = batch["r_ids"].to(device); r_labels = batch["r_labels"].to(device); r_mask = batch["r_mask"].to(device)
    with torch.no_grad():
        ref_c = seq_logprob(ref_model, c_ids, c_labels, c_mask)
        ref_r = seq_logprob(ref_model, r_ids, r_labels, r_mask)
    pi_c = seq_logprob(model, c_ids, c_labels, c_mask)
    pi_r = seq_logprob(model, r_ids, r_labels, r_mask)
    return -torch.nn.functional.logsigmoid(beta * (pi_c - ref_c - pi_r + ref_r))

def sft_loss(model, batch):
    device = next(model.parameters()).device
    c_ids = batch["c_ids"].to(device); c_labels = batch["c_labels"].to(device); c_mask = batch["c_mask"].to(device)
    out = model(input_ids=c_ids, attention_mask=c_mask, labels=c_labels)
    return out.loss  # average over non-masked tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn_weight", type=float, required=True)
    parser.add_argument("--fp_weight", type=float, required=True)
    parser.add_argument("--lambda_anchor", type=float, default=LAMBDA)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    tag = f"_{args.tag}" if args.tag else ""
    run_dir = f"{OUT_BASE}/german_anchored_riskdpo_fn{int(args.fn_weight)}_fp{int(args.fp_weight)}{tag}"
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    print(f"C6v2: Anchored Risk-DPO (FN={int(args.fn_weight)}, FP={int(args.fp_weight)}, λ={args.lambda_anchor})")
    print(f"  Output: {run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, SFT_ADAPTER)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                              target_modules=LORA_TARGET, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    ref_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    ref_model = PeftModel.from_pretrained(ref_base, SFT_ADAPTER)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    train_ds = PrefDataset("train", tokenizer, args.fn_weight, args.fp_weight)
    valid_ds = PrefDataset("valid", tokenizer, args.fn_weight, args.fp_weight)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate(b, tokenizer))
    valid_dl = DataLoader(valid_ds, batch_size=BATCH, collate_fn=lambda b: collate(b, tokenizer))

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    best_loss, best_step = float("inf"), 0
    train_log = []

    for epoch in range(EPOCHS):
        model.train()
        total_dpo, total_sft, total_combined, step = 0.0, 0.0, 0.0, 0
        for i, batch in enumerate(train_dl):
            weights = batch["weights"]
            dpo_per = dpo_loss_per_sample(model, ref_model, batch, BETA)
            dpo = (dpo_per * weights.to(dpo_per.device)).mean()
            sft = sft_loss(model, batch)
            loss = (dpo + args.lambda_anchor * sft) / GRAD_ACCUM
            loss.backward()
            total_dpo += dpo.item(); total_sft += sft.item(); total_combined += (dpo.item() + args.lambda_anchor * sft.item())

            if (i+1) % GRAD_ACCUM == 0:
                opt.step(); opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    model.eval()
                    v_dpo, v_sft, v_combined = 0.0, 0.0, 0.0
                    with torch.no_grad():
                        for vb in valid_dl:
                            w = vb["weights"]
                            d = dpo_loss_per_sample(model, ref_model, vb, BETA)
                            v_dpo += (d * w.to(d.device)).mean().item()
                            v_sft += sft_loss(model, vb).item()
                    v_dpo /= max(len(valid_dl), 1); v_sft /= max(len(valid_dl), 1)
                    v_combined = v_dpo + args.lambda_anchor * v_sft
                    train_log.append({"epoch":epoch,"step":step,"dpo":round(total_dpo/step,4),
                                      "sft":round(total_sft/step,4),"combined":round(total_combined/step,4),
                                      "v_dpo":round(v_dpo,4),"v_sft":round(v_sft,4),"v_combined":round(v_combined,4)})
                    print(f"    e={epoch} s={step} dpo={total_dpo/step:.4f} sft={total_sft/step:.4f} "
                          f"| v_dpo={v_dpo:.4f} v_sft={v_sft:.4f} v_comb={v_combined:.4f}")
                    if v_combined < best_loss:
                        best_loss = v_combined; best_step = step
                        model.save_pretrained(f"{run_dir}/best_adapter")
                        tokenizer.save_pretrained(f"{run_dir}/best_adapter")
                    model.train()

    print(f"  Best: step={best_step}, loss={best_loss:.4f}")
    with open(f"{run_dir}/training_log.json","w") as f:
        json.dump({"fn":args.fn_weight,"fp":args.fp_weight,"lambda":args.lambda_anchor,
                   "best_step":best_step,"best_loss":round(best_loss,4),"log":train_log}, f, indent=2)

if __name__ == "__main__":
    main()
