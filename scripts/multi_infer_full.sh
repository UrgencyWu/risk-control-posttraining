#!/bin/bash
#SBATCH --job-name=mifull
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/multi_full_%j.out
#SBATCH --error=logs/multi_full_%j.err

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate qwen9B

echo "=== C5v3-A.2: Multi-dataset Full Inference (1182 train + 169 valid) ==="
nvidia-smi -L

cd /home/wushaohua/data/risk-control-posttraining
python3 src/training/multi_infer_logprobs.py
