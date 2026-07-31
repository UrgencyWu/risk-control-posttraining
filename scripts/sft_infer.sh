#!/bin/bash
#SBATCH --job-name=sft_infer
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/sft_infer_%j.out
#SBATCH --error=logs/sft_infer_%j.err

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate qwen9B

echo "=== C3: SFT Inference ==="
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

cd /home/wushaohua/data/risk-control-posttraining
python3 src/training/sft_infer.py
