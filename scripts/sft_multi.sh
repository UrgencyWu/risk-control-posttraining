#!/bin/bash
#SBATCH --job-name=sft_multi
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/sft_multi_%j.out
#SBATCH --error=logs/sft_multi_%j.err

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate qwen9B

echo "=== C5v3-A: Multi-dataset SFT (German + Australian) ==="
nvidia-smi -L

mkdir -p logs outputs/sft
cd /home/wushaohua/data/risk-control-posttraining
python3 src/training/sft_multi.py

echo "--- Inference ---"
ADAPTER_DIR=outputs/sft/german_multi
python3 src/training/pref_infer.py --adapter "$ADAPTER_DIR/best_adapter" --output "$ADAPTER_DIR"

echo "--- Done ---"
