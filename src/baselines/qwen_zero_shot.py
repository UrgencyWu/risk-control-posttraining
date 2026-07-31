"""B2: Qwen3.5-4B zero-shot baseline using a first-token class score.

No training is performed.  The score compares the next-token log-probability
of the first token of ``low`` and ``high`` after the prompt; it is *not* the
conditional likelihood of the complete strings ``low risk`` and ``high risk``.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.metrics import (
    load_ground_truth, compute_metrics, save_predictions, generate_metrics_table,
    select_cost_threshold,
)

MODEL_ID = os.environ.get("RISK_CONTROL_MODEL_ID", "/data/share/model/Qwen3.5-4B")
SFT_DIR = "data/processed/german/sft"
OUTPUT_DIR = "outputs/baselines"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_id=MODEL_ID):
    """Load Qwen3.5-4B with explicit single-GPU placement."""
    print(f"  Loading {model_id} on CPU (GPU OOM)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def compute_risk_scores(model, tokenizer, messages):
    """
    Compute a first-token class score p(high) from next-token log-probabilities.

    The normalisation is over the first token of ``low`` and ``high`` only.
    See docs/EVALUATION_PROTOCOL.md for the complete score definition.
    """
    # Build a Qwen ChatML prompt from system + user messages.
    text = tokenizer.apply_chat_template(
        messages[:-1],  # system + user, exclude assistant
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits[0, -1, :]  # last token logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Score the first continuation token of the two class labels.
    low_single = tokenizer.encode("low", add_special_tokens=False)
    high_single = tokenizer.encode("high", add_special_tokens=False)

    # Score: use first token logprob for "low" vs "high"
    s_low = log_probs[low_single[0]].item()
    s_high = log_probs[high_single[0]].item()

    # Normalize to probabilities
    p_low = np.exp(s_low) / (np.exp(s_low) + np.exp(s_high))
    p_high = np.exp(s_high) / (np.exp(s_low) + np.exp(s_high))

    return p_high, {"s_low": s_low, "s_high": s_high, "p_low": p_low, "p_high": p_high}


def run_qwen_zero_shot():
    """Run the Qwen3.5-4B zero-shot baseline with first-token class scoring."""
    model, tokenizer = load_model_and_tokenizer()

    results = {}

    for split in ("valid", "test"):
        sft_path = f"{SFT_DIR}/{split}.jsonl"
        print(f"  Running {split} set ({sft_path})...")

        sft_records = []
        with open(sft_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    sft_records.append(json.loads(line))

        print(f"  Loaded {len(sft_records)} samples")

        ground_truth = []
        risk_scores = []
        predictions = []
        pred_records = []

        for i, sft_r in enumerate(sft_records):
            messages = sft_r["messages"]
            metadata = sft_r["metadata"]
            gt = metadata["risk_label"]

            try:
                p_high, logprob_info = compute_risk_scores(model, tokenizer, messages)
                risk_scores.append(p_high)
            except Exception as e:
                print(f"    Warning: sample {metadata['sample_id']} failed: {e}")
                risk_scores.append(float("nan"))

            ground_truth.append(gt)

            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(sft_records)}...")

        # Select threshold (on valid set) or use default 0.5
        if split == "valid":
            valid_scores_clean = [s for s in risk_scores if not np.isnan(s)]
            valid_gt_clean = [ground_truth[i] for i, s in enumerate(risk_scores) if not np.isnan(s)]
            threshold, _ = select_cost_threshold(valid_scores_clean, valid_gt_clean)
            print(f"  Optimal threshold ({split}): {threshold:.2f}")
        else:
            # Use threshold from valid
            threshold = getattr(run_qwen_zero_shot, "_threshold", 0.5)

        if split == "valid":
            run_qwen_zero_shot._threshold = threshold

        # Apply threshold
        for i, (score, gt) in enumerate(zip(risk_scores, ground_truth)):
            if np.isnan(score):
                pred = 0  # default to low risk on failure
                valid_rate_flag = False
            else:
                pred = 1 if score >= threshold else 0
                valid_rate_flag = True

            error_type = None
            if gt == 1 and pred == 0:
                error_type = "false_negative"
            elif gt == 0 and pred == 1:
                error_type = "false_positive"

            predictions.append(pred)
            pred_records.append({
                "sample_id": sft_records[i]["metadata"]["sample_id"],
                "ground_truth": gt,
                "prediction": pred,
                "risk_score": round(float(score), 6) if not np.isnan(score) else None,
                "threshold": round(threshold, 2),
                "error_type": error_type,
                "cost": 5 if error_type == "false_negative" else (1 if error_type == "false_positive" else 0),
                "model": "Qwen3.5-4B-zero-shot",
                "valid": valid_rate_flag,
            })

        save_predictions(pred_records, f"{OUTPUT_DIR}/qwen_zero_shot_{split}.jsonl")
        valid_count = sum(1 for r in pred_records if r["valid"])
        valid_rate = valid_count / len(pred_records) if pred_records else 0

        metrics = compute_metrics(ground_truth, predictions, risk_scores)
        metrics["valid_rate"] = round(valid_rate, 4)
        results["Qwen-ZeroShot"] = metrics
        print(f"  {split}: acc={metrics['accuracy']:.4f}, high_risk_recall={metrics['high_risk_recall']:.4f}, "
              f"roc_auc={metrics['roc_auc']}, cost={metrics['cost']}, valid_rate={valid_rate:.4f}")

    return results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== B2: Qwen3.5-4B Zero-shot Baseline ===")
    results = run_qwen_zero_shot()
    print("\n" + generate_metrics_table(results))
