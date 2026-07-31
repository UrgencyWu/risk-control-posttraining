#!/bin/bash
#SBATCH --job-name=c5v3_pipe
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/c5v3_pipe_%j.out
#SBATCH --error=logs/c5v3_pipe_%j.err

source /home/wushaohua/miniconda3/etc/profile.d/conda.sh
conda activate qwen9B

echo "============================================"
echo " C5v3 Pipeline: A.2 → A.3 → A.4"
echo " Job: $SLURM_JOB_ID  GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"
nvidia-smi -L

mkdir -p logs outputs/preference_inference data/processed/preference_multidataset outputs/dpo
cd /home/wushaohua/data/risk-control-posttraining

# ======== A.2: Full Inference ========
echo ""
echo "======== A.2: Full Inference (train 1182 + valid 169) ========"
python3 src/training/multi_infer_logprobs.py
if [ $? -ne 0 ]; then
    echo "A.2 FAILED"; exit 1
fi

# ======== A.3 + A.4: Build Preference + Train DPO/SimPO ========
echo ""
echo "======== A.3 + A.4: Hard Preference + DPO/SimPO ========"
python3 src/training/c5v3_pipeline.py --step all --mode both
if [ $? -ne 0 ]; then
    echo "A.3/A.4 FAILED"; exit 1
fi

echo ""
echo "======== Pipeline Complete ========"
