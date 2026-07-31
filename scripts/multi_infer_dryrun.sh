#!/bin/bash
#SBATCH --job-name=midry
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/multi_dry_%j.out
#SBATCH --error=logs/multi_dry_%j.err

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate qwen9B

echo "=== C5v3-A.1: Multi-dataset Inference — DRY RUN ==="
nvidia-smi -L

cd /home/wushaohua/data/risk-control-posttraining
python3 src/training/multi_infer_logprobs.py --dry_run 10
