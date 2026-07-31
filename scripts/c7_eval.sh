#!/bin/bash
#SBATCH --job-name=c7_eval
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/c7_%j.out
#SBATCH --error=logs/c7_%j.err

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate qwen9B

echo "=== C7: Final Evaluation ==="
nvidia-smi -L

cd /home/wushaohua/data/risk-control-posttraining
python3 src/evaluation/c7_final.py

echo "--- Done ---"
