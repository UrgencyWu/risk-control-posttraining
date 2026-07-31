"""
C5v3-A: Multi-dataset SFT (German + Australian)
Same LoRA config as C3, combined data.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

MODEL_ID = "/data/share/model/Qwen3.5-4B"
SFT_DIR = "data/processed/multi/combined/sft"
OUT_DIR = "outputs/sft/german_multi"

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
BATCH, GRAD_ACCUM = 4, 2
LR, EPOCHS, MAX_LEN = 2e-4, 5, 2048
SEED = 42

class SFTDataset(Dataset):
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
        full = prompt + msgs[2]["content"] + self.tokenizer.eos_token
        ids = self.tokenizer(full, truncation=True, max_length=MAX_LEN)["input_ids"]
        p_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        labels = [-100]*len(p_ids) + ids[len(p_ids):]
        return {"input_ids": ids, "labels": labels}

def collate(batch, tokenizer):
    def pad(seqs, val):
        m = max(len(s) for s in seqs)
        return [s + [val]*(m-len(s)) for s in seqs]
    return {
        "input_ids": torch.tensor(pad([b["input_ids"] for b in batch], tokenizer.pad_token_id)),
        "labels": torch.tensor(pad([b["labels"] for b in batch], -100)),
        "attention_mask": torch.tensor(pad([[1]*len(b["input_ids"]) for b in batch], 0)),
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda")
    print(f"C5v3-A: Multi-dataset SFT (German + Australian)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                              target_modules=LORA_TARGET, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train_ds = SFTDataset("train", tokenizer)
    valid_ds = SFTDataset("valid", tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate(b, tokenizer))
    valid_dl = DataLoader(valid_ds, batch_size=BATCH, collate_fn=lambda b: collate(b, tokenizer))
    print(f"  Train: {len(train_ds)}  Valid: {len(valid_ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    best_loss, best_step = float("inf"), 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, step = 0.0, 0
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item()
            if (i+1) % GRAD_ACCUM == 0:
                opt.step(); opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    model.eval()
                    v_loss = sum(model(**{k:v.to(device) for k,v in vb.items()}).loss.item() for vb in valid_dl)/len(valid_dl)
                    print(f"    e={epoch} s={step} train_loss={total_loss/step:.4f} valid_loss={v_loss:.4f}")
                    if v_loss < best_loss:
                        best_loss = v_loss; best_step = step
                        model.save_pretrained(f"{OUT_DIR}/best_adapter")
                        tokenizer.save_pretrained(f"{OUT_DIR}/best_adapter")
                    model.train()

    print(f"  Best: step={best_step}, loss={best_loss:.4f}")
    with open(f"{OUT_DIR}/training_log.json","w") as f:
        json.dump({"seed":SEED,"train_size":len(train_ds),"valid_size":len(valid_ds),
                   "best_step":best_step,"best_loss":round(best_loss,4)}, f, indent=2)

if __name__ == "__main__":
    main()
