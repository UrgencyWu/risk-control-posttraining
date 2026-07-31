"""
C3: LoRA SFT — Qwen3.5-4B on German Credit
Trains LoRA adapters to classify "low risk" / "high risk".
Evaluates on valid set, selects best checkpoint, tests on test set.
Supports multiple random seeds.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import torch
import numpy as np
from datetime import datetime, timezone
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset

from src.evaluation.metrics import compute_metrics as eval_compute_metrics, generate_metrics_table

# ============================================================
# Config
# ============================================================
MODEL_ID = "/data/share/model/Qwen3.5-4B"
SFT_DIR = "data/processed/german/sft"
OUTPUT_BASE = "outputs/sft"
# Use GPU 3 via CUDA_VISIBLE_DEVICES=3 (becomes cuda:0 to PyTorch)
GPU_ID = "4"

# LoRA hyperparams
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training hyperparams
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16   # effective batch = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
MAX_SEQ_LENGTH = 512
WARMUP_RATIO = 0.1
LOGGING_STEPS = 10
SAVE_STEPS = 50
EVAL_STEPS = 50
FP16 = False  # use bf16

SEEDS = [10086]  # start with 1, add 42, 7 after verifying pipeline

# System prompt (same as B2)
SYSTEM_PROMPT = (
    "You are a financial risk assessment expert. Evaluate the creditworthiness "
    "based on the customer's financial profile. Classify the risk level as:\n"
    "- low risk: the customer is likely to repay\n"
    "- high risk: the customer is likely to default\n"
    "Respond with only 'low risk' or 'high risk'."
)


def load_dataset(split):
    """Load SFT JSONL and convert to HuggingFace Dataset with pre-tokenized format."""
    data = []
    path = os.path.join(SFT_DIR, f"{split}.jsonl")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_for_training(tokenizer, records):
    """Format records as {'input_text': ..., 'output_text': ...} for seq2seq training."""
    formatted = []
    for r in records:
        msgs = r["messages"]
        # Build prompt: system + user (using Qwen3.5 chat template)
        # We use apply_chat_template to get the full prompt
        prompt_msgs = [
            {"role": "system", "content": msgs[0]["content"]},
            {"role": "user", "content": msgs[1]["content"]},
        ]
        # Build full input using chat template without assistant response
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        # Assistant response
        assistant_text = msgs[2]["content"]

        formatted.append({
            "prompt": prompt_text,
            "completion": assistant_text,
            "risk_label": r["metadata"]["risk_label"],
            "sample_id": r["metadata"]["sample_id"],
        })
    return formatted


def tokenize_function(examples, tokenizer):
    """Tokenize prompt+completion pairs for training."""
    full_texts = [p + c + tokenizer.eos_token for p, c in zip(examples["prompt"], examples["completion"])]
    tokenized = tokenizer(
        full_texts, truncation=True, padding=False, max_length=MAX_SEQ_LENGTH
    )
    # Labels: input_ids, with prompt part masked to -100
    labels_list = []
    for i, (prompt, completion) in enumerate(zip(examples["prompt"], examples["completion"])):
        prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LENGTH, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + completion + tokenizer.eos_token, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"]
        # Mask prompt tokens
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        # Pad to max_length in collator, just return raw here
        labels_list.append(labels)
    return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"], "labels": labels_list}


class SFTDataCollator:
    """Pad input_ids and labels to max length in batch."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            padded_input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            padded_attention_mask.append(f["attention_mask"] + [0] * pad_len)
            padded_labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def train_sft(seed):
    """Train LoRA SFT for one seed. Returns path to best checkpoint."""
    run_name = f"german_sft_seed{seed}"
    output_dir = os.path.join(OUTPUT_BASE, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Seed: {seed}  |  Output: {output_dir}")
    print(f"{'='*60}")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load tokenizer and model
    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # disable KV cache during training
    # Skip gradient checkpointing — can cause issues on some GPUs

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print("  Loading data...")
    train_raw = load_dataset("train")
    valid_raw = load_dataset("valid")

    train_formatted = format_for_training(tokenizer, train_raw)
    valid_formatted = format_for_training(tokenizer, valid_raw)

    train_dataset = Dataset.from_list(train_formatted)
    valid_dataset = Dataset.from_list(valid_formatted)

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True,
        remove_columns=train_dataset.column_names
    )
    valid_dataset = valid_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True,
        remove_columns=valid_dataset.column_names
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        dataloader_drop_last=False,
        report_to="none",
        seed=seed,
        lr_scheduler_type="cosine",
        save_total_limit=3,
        logging_dir=os.path.join(output_dir, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=SFTDataCollator(tokenizer),
    )

    print("  Training...")
    trainer.train()

    # Save best model
    best_ckpt = trainer.state.best_model_checkpoint
    print(f"  Best checkpoint: {best_ckpt}")

    # Save final adapter separately
    adapter_dir = os.path.join(output_dir, "best_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Save training log
    log_history = trainer.state.log_history
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    return adapter_dir, model, tokenizer


def run_inference(model, tokenizer, split, output_path):
    """Run inference on valid or test set using LoRA model."""
    records = load_dataset(split)
    results = []

    for r in records:
        msgs = r["messages"]
        prompt_msgs = [
            {"role": "system", "content": msgs[0]["content"]},
            {"role": "user", "content": msgs[1]["content"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        # Get logprobs for "low risk" vs "high risk"
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        low_id = tokenizer.encode("low", add_special_tokens=False)[0]
        high_id = tokenizer.encode("high", add_special_tokens=False)[0]

        s_low = log_probs[low_id].item()
        s_high = log_probs[high_id].item()

        p_high = np.exp(s_high) / (np.exp(s_low) + np.exp(s_high))

        gt = r["metadata"]["risk_label"]
        pred = 1 if p_high >= 0.5 else 0

        error_type = None
        if gt == 1 and pred == 0:
            error_type = "false_negative"
        elif gt == 0 and pred == 1:
            error_type = "false_positive"

        results.append({
            "sample_id": r["metadata"]["sample_id"],
            "ground_truth": gt,
            "prediction": pred,
            "risk_score": round(p_high, 6),
            "threshold": 0.5,
            "error_type": error_type,
            "cost": 5 if error_type == "false_negative" else (1 if error_type == "false_positive" else 0),
            "model": "Qwen3.5-4B-SFT",
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results


def compute_best_threshold(results, fn_cost=5, fp_cost=1):
    """Find optimal threshold on validation set."""
    scores = np.array([r["risk_score"] for r in results])
    gts = np.array([r["ground_truth"] for r in results])
    best_t, best_cost = 0.5, float("inf")
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (scores >= t).astype(int)
        cost = 0
        for gt, pred in zip(gts, preds):
            if gt == 1 and pred == 0:
                cost += fn_cost
            elif gt == 0 and pred == 1:
                cost += fp_cost
        if cost < best_cost:
            best_cost = cost
            best_t = t
    return best_t, best_cost


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    all_results = {}

    for seed in SEEDS:
        print(f"\n{'#'*60}")
        print(f"# C3: LoRA SFT — Seed {seed}")
        print(f"{'#'*60}")

        # Train
        adapter_dir, model, tokenizer = train_sft(seed)

        # Inference on valid set
        print("\n  Running valid inference...")
        valid_preds = run_inference(
            model, tokenizer, "valid",
            os.path.join(OUTPUT_BASE, f"german_sft_seed{seed}", "valid_predictions.jsonl")
        )

        # Find optimal threshold
        best_threshold, best_cost = compute_best_threshold(valid_preds)
        print(f"  Best threshold (valid): {best_threshold:.2f}, cost={best_cost}")

        # Inference on test set with optimal threshold
        print("  Running test inference...")
        test_preds = run_inference(
            model, tokenizer, "test",
            os.path.join(OUTPUT_BASE, f"german_sft_seed{seed}", "test_predictions_raw.jsonl")
        )

        # Apply optimal threshold to test predictions
        test_preds_adjusted = []
        for r in test_preds:
            pred = 1 if r["risk_score"] >= best_threshold else 0
            gt = r["ground_truth"]
            error_type = None
            if gt == 1 and pred == 0:
                error_type = "false_negative"
            elif gt == 0 and pred == 1:
                error_type = "false_positive"
            test_preds_adjusted.append({
                **r,
                "prediction": pred,
                "threshold": round(best_threshold, 2),
                "error_type": error_type,
                "cost": 5 if error_type == "false_negative" else (1 if error_type == "false_positive" else 0),
            })

        with open(os.path.join(OUTPUT_BASE, f"german_sft_seed{seed}", "test_predictions.jsonl"), "w") as f:
            for r in test_preds_adjusted:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Compute metrics
        gt = [r["ground_truth"] for r in test_preds_adjusted]
        y_pred = [r["prediction"] for r in test_preds_adjusted]
        scores = [r["risk_score"] for r in test_preds_adjusted]
        metrics = eval_compute_metrics(gt, y_pred, scores)
        all_results[f"SFT_seed{seed}"] = metrics
        print(f"  Test: acc={metrics['accuracy']:.4f}, high_risk_recall={metrics['high_risk_recall']:.4f}, roc_auc={metrics['roc_auc']}, cost={metrics['cost']}")

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Merge results across seeds (average)
    avg_acc = np.mean([all_results[f"SFT_seed{s}"]["accuracy"] for s in SEEDS])
    avg_roc = np.mean([all_results[f"SFT_seed{s}"]["roc_auc"] for s in SEEDS])
    avg_cost = np.mean([all_results[f"SFT_seed{s}"]["cost"] for s in SEEDS])
    avg_recall = np.mean([all_results[f"SFT_seed{s}"]["high_risk_recall"] for s in SEEDS])
    print(f"\n  SFT Average: acc={avg_acc:.4f}, roc_auc={avg_roc:.4f}, high_risk_recall={avg_recall:.4f}, cost={avg_cost:.1f}")

    # Save aggregate results
    aggregate = {
        "model": "Qwen3.5-4B-SFT",
        "training_samples": 700,
        "seeds": SEEDS,
        "per_seed": all_results,
        "average": {
            "accuracy": round(avg_acc, 4),
            "roc_auc": round(avg_roc, 4),
            "high_risk_recall": round(avg_recall, 4),
            "cost": round(avg_cost, 1),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(OUTPUT_BASE, "sft_results.json"), "w") as f:
        json.dump(aggregate, f, indent=2)

    print("\n" + "=" * 60)
    print("C3 SFT Complete. Results saved to outputs/sft/")
    print("=" * 60)


if __name__ == "__main__":
    main()
