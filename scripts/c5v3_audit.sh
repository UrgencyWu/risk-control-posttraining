#!/bin/bash
#SBATCH --job-name=c5v3_audit
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/c5v3_audit_%j.out
#SBATCH --error=logs/c5v3_audit_%j.err

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate qwen9B

echo "=== C5v3 Audit ==="
nvidia-smi -L

cd /home/wushaohua/data/risk-control-posttraining
python3 src/evaluation/c5v3_audit.py

echo "--- Done ---"
