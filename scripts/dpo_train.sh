#!/bin/bash
#SBATCH --job-name=dpo_german
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/dpo_%j.out
#SBATCH --error=logs/dpo_%j.err

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate qwen9B

echo "=== C5: DPO Training ==="
echo "Job ID: $SLURM_JOB_ID  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L
echo ""

mkdir -p logs outputs/dpo
cd /home/wushaohua/data/risk-control-posttraining
python3 src/training/dpo_train.py --mode dpo --seed 42
