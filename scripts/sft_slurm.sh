#!/bin/bash
#SBATCH --job-name=german_sft
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== C3: LoRA SFT — German Credit (seeds 42, 7) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
which python3
nvidia-smi -L
echo ""

mkdir -p "$REPOSITORY_ROOT/logs"

cd "$REPOSITORY_ROOT"
"$PYTHON_BIN" -m src.training.sft_lora_manual
