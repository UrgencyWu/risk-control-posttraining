#!/bin/bash
#SBATCH --job-name=c5v2_check
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/c5v2_%j.out
#SBATCH --error=logs/c5v2_%j.err

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate qwen9B

echo "=== C5v2: Implementation Check + Hard Preference Construction ==="
nvidia-smi -L

cd /home/wushaohua/data/risk-control-posttraining
python3 src/training/c5v2_check.py
