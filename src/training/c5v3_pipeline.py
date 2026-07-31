"""
C5v3-A.2 → A.3 → A.4 pipeline.
Step A.2: Full inference (already done by multi_infer_logprobs.py)
Step A.3: Build multi-dataset hard preference from margin files
Step A.4: DPO/SimPO training
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, torch, numpy as np, argparse, hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

# ============================================================
# Config
# ============================================================
MODEL_ID = os.environ.get("RISK_CONTROL_MODEL_ID", "/data/share/model/Qwen3.5-4B")
SFT_ADAPTER = "outputs/sft/german_multi/best_adapter"
SFT_DIR = "data/processed/multi/combined/sft"
INFER_DIR = "outputs/preference_inference"
PREF_OUT = "data/processed/preference_multidataset"
DPO_OUT = "outputs/dpo/german_multi"

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
BATCH, GRAD_ACCUM = 2, 4
LR_DPO, EPOCHS_DPO, MAX_LEN = 5e-6, 1, 2048  # pilot: 1 epoch, full: 2 epochs
BETA = 0.05  # Lower beta for stability
SEED = 42

SYSTEM_PROMPT = (
    "You are a financial risk assessment expert. Evaluate the creditworthiness "
    "based on the customer's financial profile. Classify the risk level as:\n"
    "- low risk: the customer is likely to repay\n"
    "- high risk: the customer is likely to default\n"
    "Respond with only 'low risk' or 'high risk'."
)

# ============================================================
# A.3: Hard Preference Construction
# ============================================================
def build_hard_preference(train_path, valid_path, keep_frac=0.3):
    """Build hard preference pairs from inference margin files.
    - margin<=0: keep ALL (ranking errors)
    - margin>0: keep bottom `keep_frac` per (dataset × risk_label)
    """
    with open(train_path) as f:
        train_data = [json.loads(l) for l in f if l.strip()]
    with open(valid_path) as f:
        valid_data = [json.loads(l) for l in f if l.strip()]

    def build_pairs(data, split_name):
        # Group by dataset × risk_label
        groups = {}
        for r in data:
            key = (r["dataset"], r["risk_label"])
            groups.setdefault(key, []).append(r)

        selected = set()
        for (ds, rl), items in groups.items():
            errors = [r for r in items if r["gt_margin"] <= 0]
            correct = sorted([r for r in items if r["gt_margin"] > 0], key=lambda x: x["gt_margin"])
            n_keep = max(int(len(correct) * keep_frac), 1)

            for r in errors:
                selected.add(r["sample_id"])
            for r in correct[:n_keep]:
                selected.add(r["sample_id"])

            print(f"    {ds}/{('low_risk' if rl==0 else 'high_risk')}: "
                  f"errors={len(errors)} correct_kept={min(n_keep, len(correct))}/{len(correct)} "
                  f"total_kept={len(errors)+min(n_keep, len(correct))}/{len(items)}")

        pairs = []
        for r in data:
            if r["sample_id"] not in selected:
                continue
            gt = r["risk_label"]
            if gt == 1:
                chosen_text, rejected_text = "high risk", "low risk"
                err_type = "false_negative"
                rw = 5.0
            else:
                chosen_text, rejected_text = "low risk", "high risk"
                err_type = "false_positive"
                rw = 1.0

            # Build prompt from original SFT data
            # (We don't have the original prompt handy, so reconstruct)
            pairs.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ""},  # placeholder, filled below
                ],
                "chosen": [{"role": "assistant", "content": chosen_text}],
                "rejected": [{"role": "assistant", "content": rejected_text}],
                "metadata": {
                    "sample_id": r["sample_id"],
                    "dataset": r["dataset"],
                    "split": split_name,
                    "task_type": "credit_scoring",
                    "risk_label": gt,
                },
                "error_type": err_type,
                "risk_weight": rw,
                "sft_margin": r["gt_margin"],
                "difficulty": "ranking_error" if r["gt_margin"] <= 0 else "low_confidence",
                "source": "multi_sft_error_oracle",
                "generation_stage": "multi_sft",
                "model": "Qwen3.5-4B-SFT-multi",
            })
        return pairs

    print("  Building hard preference pairs...")
    train_pairs = build_pairs(train_data, "train")
    valid_pairs = build_pairs(valid_data, "valid")

    # Fill user prompts from SFT data
    def fill_user_prompts(pairs, split):
        sft_path = f"{SFT_DIR}/{split}.jsonl"
        with open(sft_path) as f:
            sft_data = {r["metadata"]["sample_id"]: r for r in [json.loads(l) for l in f if l.strip()]}
        for p in pairs:
            sid = p["metadata"]["sample_id"]
            if sid in sft_data:
                p["prompt"][1]["content"] = sft_data[sid]["messages"][1]["content"]

    fill_user_prompts(train_pairs, "train")
    fill_user_prompts(valid_pairs, "valid")

    os.makedirs(PREF_OUT, exist_ok=True)
    with open(f"{PREF_OUT}/train.jsonl", "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(f"{PREF_OUT}/valid.jsonl", "w") as f:
        for p in valid_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # SHA + manifest
    with open(f"{PREF_OUT}/train.jsonl", "rb") as f:
        train_sha = hashlib.sha256(f.read()).hexdigest()
    with open(f"{PREF_OUT}/valid.jsonl", "rb") as f:
        valid_sha = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "source_inference": INFER_DIR,
        "train_pairs": len(train_pairs),
        "valid_pairs": len(valid_pairs),
        "keep_frac": keep_frac,
        "train_sha256": train_sha,
        "valid_sha256": valid_sha,
    }
    with open(f"{PREF_OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {len(train_pairs)} train + {len(valid_pairs)} valid pairs")
    print(f"  SHA-256 train={train_sha[:16]}... valid={valid_sha[:16]}...")
    return train_pairs, valid_pairs

# ============================================================
# A.4: DPO/SimPO Training
# ============================================================
class PrefDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.tokenizer = tokenizer
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        p_text = self.tokenizer.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)
        c_text = p_text + item["chosen"][0]["content"] + self.tokenizer.eos_token
        r_text = p_text + item["rejected"][0]["content"] + self.tokenizer.eos_token
        p_ids = self.tokenizer(p_text, add_special_tokens=False)["input_ids"]
        c_ids = self.tokenizer(c_text, truncation=True, max_length=MAX_LEN)["input_ids"]
        r_ids = self.tokenizer(r_text, truncation=True, max_length=MAX_LEN)["input_ids"]
        return {"c_ids": c_ids, "c_labels": [-100]*len(p_ids) + c_ids[len(p_ids):],
                "r_ids": r_ids, "r_labels": [-100]*len(p_ids) + r_ids[len(p_ids):]}

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
    }

def seq_logprob(model, ids, labels, mask):
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    shift_logits = out.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    tloss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.shape)
    m = (shift_labels != -100).float()
    return -(tloss * m).sum(dim=-1)

def train_dpo(mode, train_pairs, valid_pairs):
    """mode: 'dpo' or 'simpo'. Returns best adapter path."""
    tag = f"{mode}_multi"
    run_dir = f"{DPO_OUT}_{tag}"
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"\n  Training {mode.upper()} on {len(train_pairs)} pairs...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, SFT_ADAPTER)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                              target_modules=LORA_TARGET, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    # Reference model for DPO
    ref_model = None
    if mode == "dpo":
        ref_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        ref_model = PeftModel.from_pretrained(ref_base, SFT_ADAPTER)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    train_ds = PrefDataset(train_pairs, tokenizer)
    valid_ds = PrefDataset(valid_pairs, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate(b, tokenizer))
    valid_dl = DataLoader(valid_ds, batch_size=BATCH, collate_fn=lambda b: collate(b, tokenizer))

    opt = torch.optim.AdamW(model.parameters(), lr=LR_DPO)
    best_loss, best_step = float("inf"), 0
    eval_every = 5  # pilot: validate every 5 steps

    for epoch in range(EPOCHS_DPO):
        model.train()
        total_loss, step = 0.0, 0
        for i, batch in enumerate(train_dl):
            device = next(model.parameters()).device
            c_ids = batch["c_ids"].to(device); c_labels = batch["c_labels"].to(device); c_mask = batch["c_mask"].to(device)
            r_ids = batch["r_ids"].to(device); r_labels = batch["r_labels"].to(device); r_mask = batch["r_mask"].to(device)

            pi_c = seq_logprob(model, c_ids, c_labels, c_mask)
            pi_r = seq_logprob(model, r_ids, r_labels, r_mask)

            if mode == "dpo":
                with torch.no_grad():
                    ref_c = seq_logprob(ref_model, c_ids, c_labels, c_mask)
                    ref_r = seq_logprob(ref_model, r_ids, r_labels, r_mask)
                loss = -torch.nn.functional.logsigmoid(BETA * (pi_c - ref_c - pi_r + ref_r)).mean()
            else:  # simpo
                c_len = (c_labels != -100).float().sum(dim=-1).clamp(min=1)
                r_len = (r_labels != -100).float().sum(dim=-1).clamp(min=1)
                loss = -torch.nn.functional.logsigmoid(BETA * (pi_c/c_len - pi_r/r_len - 0.5)).mean()

            (loss / GRAD_ACCUM).backward()
            total_loss += loss.item()
            if (i+1) % GRAD_ACCUM == 0:
                opt.step(); opt.zero_grad()
                step += 1
                if step % eval_every == 0:
                    model.eval()
                    v_loss = 0.0
                    margin_sum = 0.0
                    n_v = 0
                    with torch.no_grad():
                        for vb in valid_dl:
                            vc = vb["c_ids"].to(device); vcl = vb["c_labels"].to(device); vcm = vb["c_mask"].to(device)
                            vr = vb["r_ids"].to(device); vrl = vb["r_labels"].to(device); vrm = vb["r_mask"].to(device)
                            pi_cv = seq_logprob(model, vc, vcl, vcm)
                            pi_rv = seq_logprob(model, vr, vrl, vrm)
                            if mode == "dpo":
                                with torch.no_grad():
                                    ref_cv = seq_logprob(ref_model, vc, vcl, vcm)
                                    ref_rv = seq_logprob(ref_model, vr, vrl, vrm)
                                v_loss += -torch.nn.functional.logsigmoid(BETA*(pi_cv-ref_cv-pi_rv+ref_rv)).mean().item()
                            else:
                                cv_len = (vcl != -100).float().sum(dim=-1).clamp(min=1)
                                rv_len = (vrl != -100).float().sum(dim=-1).clamp(min=1)
                                v_loss += -torch.nn.functional.logsigmoid(BETA*(pi_cv/cv_len-pi_rv/rv_len-0.5)).mean().item()
                            margin_sum += (pi_cv - pi_rv).mean().item()
                            n_v += 1
                    v_loss /= max(n_v, 1)
                    avg_margin = margin_sum / max(n_v, 1)

                    # Pilot stability checks
                    warnings = []
                    if v_loss < 0.01:
                        warnings.append(f"LOSS_COLLAPSE: valid_loss={v_loss:.6f}")
                    if avg_margin > 10:
                        warnings.append(f"MARGIN_EXPLOSION: margin={avg_margin:.1f}")

                    print(f"    e={epoch} s={step} t_loss={total_loss/step:.4f} v_loss={v_loss:.4f} margin={avg_margin:.2f}", end="")
                    if warnings:
                        print(f"  ⚠️  {' | '.join(warnings)}")
                    else:
                        print()

                    if v_loss < best_loss:
                        best_loss = v_loss; best_step = step
                        model.save_pretrained(f"{run_dir}/best_adapter")
                        tokenizer.save_pretrained(f"{run_dir}/best_adapter")
                    model.train()

    print(f"  Best: step={best_step}, loss={best_loss:.4f}")
    with open(f"{run_dir}/training_log.json", "w") as f:
        json.dump({"mode": mode, "seed": SEED, "best_step": best_step, "best_loss": round(best_loss, 4),
                   "train_pairs": len(train_pairs), "valid_pairs": len(valid_pairs),
                   "beta": BETA}, f, indent=2)
    return f"{run_dir}/best_adapter"

# ============================================================
# Per-dataset evaluation (reused from eval_multi)
# ============================================================
def evaluate_adapter(adapter_path, label):
    print(f"\n  Evaluating {label}...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    for split in ("valid", "test"):
        with open(f"{SFT_DIR}/{split}.jsonl") as f:
            data = [json.loads(l) for l in f if l.strip()]
        all_gts, all_scores = [], []
        per_ds = {}
        for r in data:
            ds = r["metadata"]["dataset"]
            msgs = r["messages"]
            prompt = tokenizer.apply_chat_template(
                [{"role":"system","content":msgs[0]["content"]},
                 {"role":"user","content":msgs[1]["content"]}],
                tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                lp = torch.nn.functional.log_softmax(model(**ids).logits[0,-1,:], dim=-1)
            s_low = lp[tokenizer.encode("low",add_special_tokens=False)[0]].item()
            s_high = lp[tokenizer.encode("high",add_special_tokens=False)[0]].item()
            p_high = np.exp(s_high)/(np.exp(s_low)+np.exp(s_high))
            gt = r["metadata"]["risk_label"]
            all_gts.append(gt); all_scores.append(p_high)
            per_ds.setdefault(ds, {"gt":[], "score":[]})
            per_ds[ds]["gt"].append(gt); per_ds[ds]["score"].append(p_high)

        from sklearn.metrics import roc_auc_score
        print(f"    {split}:")
        for ds in sorted(per_ds):
            g = per_ds[ds]["gt"]; s = per_ds[ds]["score"]
            auc = roc_auc_score(g, s) if len(set(g))>1 else 0.5
            hr_pct = sum(1 for x in s if x>=0.5)/len(s)*100
            print(f"      {ds:12s}: ROC-AUC={auc:.4f} HR%={hr_pct:.1f}%")
        overall_auc = roc_auc_score(all_gts, all_scores)
        print(f"      {'OVERALL':12s}: ROC-AUC={overall_auc:.4f}")

# ============================================================
# Main pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["a3", "a4", "all"], default="all")
    parser.add_argument("--mode", choices=["dpo", "simpo", "both"], default="both")
    args = parser.parse_args()

    if args.step in ("a3", "all"):
        print("=" * 60)
        print("A.3: Building Hard Preference Dataset")
        print("=" * 60)
        train_pairs, valid_pairs = build_hard_preference(
            f"{INFER_DIR}/train_predictions.jsonl",
            f"{INFER_DIR}/valid_predictions.jsonl")

    if args.step in ("a4", "all"):
        # Load pairs from A.3
        with open(f"{PREF_OUT}/train.jsonl") as f:
            train_pairs = [json.loads(l) for l in f if l.strip()]
        with open(f"{PREF_OUT}/valid.jsonl") as f:
            valid_pairs = [json.loads(l) for l in f if l.strip()]
        print(f"\n{'='*60}")
        print(f"A.4: DPO/SimPO Training ({len(train_pairs)} train, {len(valid_pairs)} valid)")
        print(f"{'='*60}")

        # SFT baseline evaluation
        evaluate_adapter(SFT_ADAPTER, "SFT baseline")

        if args.mode in ("dpo", "both"):
            dpo_path = train_dpo("dpo", train_pairs, valid_pairs)
            evaluate_adapter(dpo_path, "DPO multi")

        if args.mode in ("simpo", "both"):
            simpo_path = train_dpo("simpo", train_pairs, valid_pairs)
            evaluate_adapter(simpo_path, "SimPO multi")

    print("\nDone.")

if __name__ == "__main__":
    main()
