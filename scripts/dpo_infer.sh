#!/bin/bash
#SBATCH --job-name=dpo_infer
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/dpo_infer_%j.out
#SBATCH --error=logs/dpo_infer_%j.err

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate qwen9B

echo "=== C5 DPO Inference: $1 ==="
nvidia-smi -L

cd /home/wushaohua/data/risk-control-posttraining
python3 src/training/pref_infer.py --adapter "$1" --output "$2"
