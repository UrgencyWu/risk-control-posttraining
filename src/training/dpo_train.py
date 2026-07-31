"""
C5: DPO / SimPO training from SFT seed 7 adapter.
700 train pairs, 100 valid pairs, equal risk_weight.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, torch, numpy as np, argparse
from datetime import datetime, timezone
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

MODEL_ID = os.environ.get("RISK_CONTROL_MODEL_ID", "/data/share/model/Qwen3.5-4B")
SFT_ADAPTER = "outputs/sft/german_sft_seed7/best_adapter"
PREF_PATH_DEFAULT = "data/processed/german/preference/preference_train.jsonl"
OUT_BASE = "outputs/dpo"

# LoRA (same as SFT)
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# Training
BATCH, GRAD_ACCUM = 2, 4  # effective batch = 8
LR, EPOCHS, MAX_LEN = 1e-5, 3, 2048
BETA = 0.1       # DPO temperature
GAMMA = 0.5      # SimPO margin
SEED = 42

SYSTEM_PROMPT = (
    "You are a financial risk assessment expert. Evaluate the creditworthiness "
    "based on the customer's financial profile. Classify the risk level as:\n"
    "- low risk: the customer is likely to repay\n"
    "- high risk: the customer is likely to default\n"
    "Respond with only 'low risk' or 'high risk'."
)

# ---- Data ----
class PrefDataset(Dataset):
    def __init__(self, split, tokenizer, pref_path):
        self.tokenizer = tokenizer
        with open(pref_path) as f:
            all_data = [json.loads(l) for l in f if l.strip()]
        self.data = [d for d in all_data if d["metadata"]["split"] == split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format prompt + chosen, prompt + rejected as full texts
        def build_full(prompt_msgs, answer_msgs):
            msgs = prompt_msgs + answer_msgs
            return self.tokenizer.apply_chat_template(msgs, tokenize=False)

        prompt_text = self.tokenizer.apply_chat_template(
            item["prompt"], tokenize=False, add_generation_prompt=True)

        chosen_full = prompt_text + item["chosen"][0]["content"] + self.tokenizer.eos_token
        rejected_full = prompt_text + item["rejected"][0]["content"] + self.tokenizer.eos_token

        # Tokenize
        c_ids = self.tokenizer(chosen_full, truncation=True, max_length=MAX_LEN)["input_ids"]
        r_ids = self.tokenizer(rejected_full, truncation=True, max_length=MAX_LEN)["input_ids"]
        p_ids = self.tokenizer(prompt_text, truncation=True, max_length=MAX_LEN, add_special_tokens=False)["input_ids"]

        # Labels: mask prompt, keep answer
        c_labels = [-100]*len(p_ids) + c_ids[len(p_ids):]
        r_labels = [-100]*len(p_ids) + r_ids[len(p_ids):]

        return {
            "chosen_input_ids": c_ids,
            "chosen_labels": c_labels,
            "rejected_input_ids": r_ids,
            "rejected_labels": r_labels,
        }

def collate(batch, tokenizer):
    def pad_seq(seqs, pad_val):
        max_len = max(len(s) for s in seqs)
        return [s + [pad_val]*(max_len-len(s)) for s in seqs]
    return {
        "chosen_input_ids": torch.tensor(pad_seq([b["chosen_input_ids"] for b in batch], tokenizer.pad_token_id)),
        "chosen_labels": torch.tensor(pad_seq([b["chosen_labels"] for b in batch], -100)),
        "chosen_attention_mask": torch.tensor(pad_seq([[1]*len(b["chosen_input_ids"]) for b in batch], 0)),
        "rejected_input_ids": torch.tensor(pad_seq([b["rejected_input_ids"] for b in batch], tokenizer.pad_token_id)),
        "rejected_labels": torch.tensor(pad_seq([b["rejected_labels"] for b in batch], -100)),
        "rejected_attention_mask": torch.tensor(pad_seq([[1]*len(b["rejected_input_ids"]) for b in batch], 0)),
    }

# ---- Logprob helper ----
def compute_seq_logprob(model, input_ids, labels, attention_mask):
    """Sum of logprobs on label tokens (non-masked)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # Per-token CE loss: out.logits -> shift for next-token prediction
    # Use the built-in loss but per-sample
    shift_logits = out.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_loss = token_loss.view(shift_labels.shape)
    mask = (shift_labels != -100).float()
    seq_nll = (token_loss * mask).sum(dim=-1)
    seq_len = mask.sum(dim=-1).clamp(min=1)
    return -seq_nll, seq_len  # logprob, token count

# ---- DPO Loss ----
def dpo_loss(model, ref_model, batch, beta):
    """Standard DPO loss."""
    device = next(model.parameters()).device
    c_ids = batch["chosen_input_ids"].to(device)
    c_labels = batch["chosen_labels"].to(device)
    c_mask = batch["chosen_attention_mask"].to(device)
    r_ids = batch["rejected_input_ids"].to(device)
    r_labels = batch["rejected_labels"].to(device)
    r_mask = batch["rejected_attention_mask"].to(device)

    with torch.no_grad():
        ref_c_logp, _ = compute_seq_logprob(ref_model, c_ids, c_labels, c_mask)
        ref_r_logp, _ = compute_seq_logprob(ref_model, r_ids, r_labels, r_mask)

    pi_c_logp, _ = compute_seq_logprob(model, c_ids, c_labels, c_mask)
    pi_r_logp, _ = compute_seq_logprob(model, r_ids, r_labels, r_mask)

    log_ratio_c = pi_c_logp - ref_c_logp
    log_ratio_r = pi_r_logp - ref_r_logp
    loss = -torch.nn.functional.logsigmoid(beta * (log_ratio_c - log_ratio_r)).mean()
    return loss

# ---- SimPO Loss ----
def simpo_loss(model, batch, beta, gamma):
    """SimPO: length-normalized, reference-free, with margin."""
    device = next(model.parameters()).device
    c_ids = batch["chosen_input_ids"].to(device)
    c_labels = batch["chosen_labels"].to(device)
    c_mask = batch["chosen_attention_mask"].to(device)
    r_ids = batch["rejected_input_ids"].to(device)
    r_labels = batch["rejected_labels"].to(device)
    r_mask = batch["rejected_attention_mask"].to(device)

    pi_c_logp, c_len = compute_seq_logprob(model, c_ids, c_labels, c_mask)
    pi_r_logp, r_len = compute_seq_logprob(model, r_ids, r_labels, r_mask)

    # Length-normalized
    pi_c_norm = pi_c_logp / c_len.float()
    pi_r_norm = pi_r_logp / r_len.float()

    loss = -torch.nn.functional.logsigmoid(beta * (pi_c_norm - pi_r_norm - gamma)).mean()
    return loss

# ---- Main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dpo", "simpo"], required=True)
    parser.add_argument("--pref_data", default=PREF_PATH_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    mode = args.mode
    seed = args.seed
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = f"{OUT_BASE}/german_{mode}_seed{seed}{tag}"
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda")

    print(f"C5: {mode.upper()} training | seed={seed} | output={run_dir}")
    print(f"  SFT adapter: {SFT_ADAPTER}")

    # Load base model + SFT adapter
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER)
    model.config.use_cache = False

    # LoRA on top of SFT adapter
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Reference model for DPO
    ref_model = None
    if mode == "dpo":
        ref_base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        ref_model = PeftModel.from_pretrained(ref_base, SFT_ADAPTER)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        print("  Reference model loaded.")

    # Data
    train_ds = PrefDataset("train", tokenizer, args.pref_data)
    valid_ds = PrefDataset("valid", tokenizer, args.pref_data)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate(b, tokenizer))
    valid_dl = DataLoader(valid_ds, batch_size=BATCH, collate_fn=lambda b: collate(b, tokenizer))

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    best_loss, best_step = float("inf"), 0
    train_log = []
    total_steps = (len(train_ds) // (BATCH * GRAD_ACCUM)) * EPOCHS

    print(f"  Training: {len(train_ds)} pairs, {EPOCHS} epochs, ~{total_steps} steps")

    for epoch in range(EPOCHS):
        model.train()
        total_loss, step = 0.0, 0
        for i, batch in enumerate(train_dl):
            if mode == "dpo":
                loss = dpo_loss(model, ref_model, batch, BETA) / GRAD_ACCUM
            else:
                loss = simpo_loss(model, batch, BETA, GAMMA) / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item()

            if (i + 1) % GRAD_ACCUM == 0:
                opt.step(); opt.zero_grad()
                step += 1

                if step % 10 == 0:
                    model.eval()
                    v_loss = 0.0
                    with torch.no_grad():
                        for vb in valid_dl:
                            if mode == "dpo":
                                v_loss += dpo_loss(model, ref_model, vb, BETA).item()
                            else:
                                v_loss += simpo_loss(model, vb, BETA, GAMMA).item()
                    v_loss /= max(len(valid_dl), 1)
                    train_log.append({"epoch": epoch, "step": step, "train_loss": round(total_loss/step, 4), "valid_loss": round(v_loss, 4)})
                    print(f"    epoch={epoch} step={step} train_loss={total_loss/step:.4f} valid_loss={v_loss:.4f}")
                    if v_loss < best_loss:
                        best_loss = v_loss; best_step = step
                        model.save_pretrained(f"{run_dir}/best_adapter")
                        tokenizer.save_pretrained(f"{run_dir}/best_adapter")
                    model.train()

    print(f"  Best: step={best_step}, valid_loss={best_loss:.4f}")

    with open(f"{run_dir}/training_log.json", "w") as f:
        json.dump({"mode": mode, "seed": seed, "best_step": best_step, "best_valid_loss": round(best_loss, 4), "log": train_log}, f, indent=2)

    print(f"  Done. Adapter saved to {run_dir}/best_adapter")

if __name__ == "__main__":
    main()
