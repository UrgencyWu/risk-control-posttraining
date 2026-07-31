"""
C6: Risk-DPO — cost-sensitive DPO with sample-level weighting.
Tests weight ratios: 1:1, 2:1, 5:1, 10:1 (FN:FP).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, torch, numpy as np, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

MODEL_ID = os.environ.get("RISK_CONTROL_MODEL_ID", "/data/share/model/Qwen3.5-4B")
SFT_ADAPTER = "outputs/sft/german_sft_seed7/best_adapter"
PREF_PATH = "data/processed/german/preference/preference_train.jsonl"
OUT_BASE = "outputs/dpo"

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
BATCH, GRAD_ACCUM = 2, 4
LR, EPOCHS, MAX_LEN = 1e-5, 3, 2048
BETA = 0.1
SEED = 42

class PrefDataset(Dataset):
    def __init__(self, split, tokenizer, fn_weight=5.0, fp_weight=1.0):
        self.tokenizer = tokenizer
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight
        with open(PREF_PATH) as f:
            all_data = [json.loads(l) for l in f if l.strip()]
        self.data = [d for d in all_data if d["metadata"]["split"] == split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_text = self.tokenizer.apply_chat_template(
            item["prompt"], tokenize=False, add_generation_prompt=True)
        c_full = prompt_text + item["chosen"][0]["content"] + self.tokenizer.eos_token
        r_full = prompt_text + item["rejected"][0]["content"] + self.tokenizer.eos_token
        c_ids = self.tokenizer(c_full, truncation=True, max_length=MAX_LEN)["input_ids"]
        r_ids = self.tokenizer(r_full, truncation=True, max_length=MAX_LEN)["input_ids"]
        p_ids = self.tokenizer(prompt_text, truncation=True, max_length=MAX_LEN, add_special_tokens=False)["input_ids"]
        c_labels = [-100]*len(p_ids) + c_ids[len(p_ids):]
        r_labels = [-100]*len(p_ids) + r_ids[len(p_ids):]
        # Map error_type to weight
        w = self.fn_weight if item["error_type"] == "false_negative" else self.fp_weight
        return {"chosen_input_ids": c_ids, "chosen_labels": c_labels,
                "rejected_input_ids": r_ids, "rejected_labels": r_labels, "weight": w}

def collate(batch, tokenizer):
    def pad(seqs, val):
        m = max(len(s) for s in seqs)
        return [s + [val]*(m-len(s)) for s in seqs]
    return {
        "chosen_input_ids": torch.tensor(pad([b["chosen_input_ids"] for b in batch], tokenizer.pad_token_id)),
        "chosen_labels": torch.tensor(pad([b["chosen_labels"] for b in batch], -100)),
        "chosen_attention_mask": torch.tensor(pad([[1]*len(b["chosen_input_ids"]) for b in batch], 0)),
        "rejected_input_ids": torch.tensor(pad([b["rejected_input_ids"] for b in batch], tokenizer.pad_token_id)),
        "rejected_labels": torch.tensor(pad([b["rejected_labels"] for b in batch], -100)),
        "rejected_attention_mask": torch.tensor(pad([[1]*len(b["rejected_input_ids"]) for b in batch], 0)),
        "weights": torch.tensor([b["weight"] for b in batch]),
    }

def seq_logprob(model, ids, labels, mask):
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    shift_logits = out.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    tloss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    tloss = tloss.view(shift_labels.shape)
    m = (shift_labels != -100).float()
    return -(tloss * m).sum(dim=-1)

def risk_dpo_loss(model, ref_model, batch, beta):
    device = next(model.parameters()).device
    c_ids = batch["chosen_input_ids"].to(device)
    c_labels = batch["chosen_labels"].to(device)
    c_mask = batch["chosen_attention_mask"].to(device)
    r_ids = batch["rejected_input_ids"].to(device)
    r_labels = batch["rejected_labels"].to(device)
    r_mask = batch["rejected_attention_mask"].to(device)
    weights = batch["weights"].to(device)

    with torch.no_grad():
        ref_c = seq_logprob(ref_model, c_ids, c_labels, c_mask)
        ref_r = seq_logprob(ref_model, r_ids, r_labels, r_mask)
    pi_c = seq_logprob(model, c_ids, c_labels, c_mask)
    pi_r = seq_logprob(model, r_ids, r_labels, r_mask)

    log_ratio_c = pi_c - ref_c
    log_ratio_r = pi_r - ref_r
    per_sample_loss = -torch.nn.functional.logsigmoid(beta * (log_ratio_c - log_ratio_r))
    return (per_sample_loss * weights).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn_weight", type=float, required=True)
    parser.add_argument("--fp_weight", type=float, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    ratio_label = f"fn{int(args.fn_weight)}_fp{int(args.fp_weight)}"
    run_dir = f"{OUT_BASE}/german_riskdpo_{ratio_label}_seed{args.seed}"
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"C6: Risk-DPO ({ratio_label}) | seed={args.seed} | {run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    ref_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
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
        total_loss, step = 0.0, 0
        for i, batch in enumerate(train_dl):
            loss = risk_dpo_loss(model, ref_model, batch, BETA) / GRAD_ACCUM
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
                            v_loss += risk_dpo_loss(model, ref_model, vb, BETA).item()
                    v_loss /= max(len(valid_dl), 1)
                    train_log.append({"epoch":epoch,"step":step,"train_loss":round(total_loss/step,4),"valid_loss":round(v_loss,4)})
                    print(f"    epoch={epoch} step={step} train_loss={total_loss/step:.4f} valid_loss={v_loss:.4f}")
                    if v_loss < best_loss:
                        best_loss = v_loss; best_step = step
                        model.save_pretrained(f"{run_dir}/best_adapter")
                        tokenizer.save_pretrained(f"{run_dir}/best_adapter")
                    model.train()

    print(f"  Best: step={best_step}, valid_loss={best_loss:.4f}")
    with open(f"{run_dir}/training_log.json","w") as f:
        json.dump({"fn_weight":args.fn_weight,"fp_weight":args.fp_weight,"seed":args.seed,
                   "best_step":best_step,"best_valid_loss":round(best_loss,4),"log":train_log}, f, indent=2)

if __name__ == "__main__":
    main()
