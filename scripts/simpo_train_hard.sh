#!/bin/bash
#SBATCH --job-name=simpo_hard
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/simpo_hard_%j.out
#SBATCH --error=logs/simpo_hard_%j.err

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate qwen9B

echo "=== C5v2: SimPO (Hard Preference) ==="
echo "Job: $SLURM_JOB_ID  GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L

mkdir -p logs outputs/dpo
cd /home/wushaohua/data/risk-control-posttraining

ADAPTER_DIR=outputs/dpo/german_simpo_seed42_hard
PREF_DATA=data/processed/german/preference/preference_train_hard.jsonl

echo "--- Training ---"
python3 src/training/dpo_train.py --mode simpo --pref_data "$PREF_DATA" --seed 42 --tag hard

echo "--- Inference ---"
python3 src/training/pref_infer.py --adapter "$ADAPTER_DIR/best_adapter" --output "$ADAPTER_DIR"

echo "--- Done ---"
