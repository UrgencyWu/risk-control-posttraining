#!/bin/bash
#SBATCH --job-name=simpo_german
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/simpo_%j.out
#SBATCH --error=logs/simpo_%j.err

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== C5: SimPO Training ==="
echo "Job ID: $SLURM_JOB_ID  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L
echo ""

cd "$REPOSITORY_ROOT"
mkdir -p logs outputs/dpo
"$PYTHON_BIN" -m src.training.dpo_train --mode simpo --seed 42
