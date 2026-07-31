#!/bin/bash
#SBATCH --job-name=adpo11
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/adpo11_%j.out
#SBATCH --error=logs/adpo11_%j.err

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate qwen9B

echo "=== C6v2 E2: Anchored DPO 1:1 ==="
nvidia-smi -L

mkdir -p logs outputs/dpo
cd /home/wushaohua/data/risk-control-posttraining

python3 src/training/anchored_risk_dpo.py --fn_weight 1 --fp_weight 1 --lambda_anchor 1.0

ADAPTER_DIR=outputs/dpo/german_anchored_riskdpo_fn1_fp1
python3 src/training/pref_infer.py --adapter "$ADAPTER_DIR/best_adapter" --output "$ADAPTER_DIR"

echo "--- Done ---"
