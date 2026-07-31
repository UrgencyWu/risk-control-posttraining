#!/bin/bash
#SBATCH --job-name=csft51
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/csft51_%j.out
#SBATCH --error=logs/csft51_%j.err

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate qwen9B

echo "=== C6v2 E1: Cost-sensitive SFT 5:1 ==="
nvidia-smi -L

mkdir -p logs outputs/dpo
cd /home/wushaohua/data/risk-control-posttraining

ADAPTER_DIR=outputs/dpo/german_cost_sft_fn5_fp1

echo "--- Training ---"
python3 src/training/cost_sensitive_sft.py --fn_weight 5 --fp_weight 1

echo "--- Inference ---"
python3 src/training/pref_infer.py --adapter "$ADAPTER_DIR/best_adapter" --output "$ADAPTER_DIR"

echo "--- Done ---"
