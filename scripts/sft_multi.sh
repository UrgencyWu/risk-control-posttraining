#!/bin/bash
#SBATCH --job-name=sft_multi
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/sft_multi_%j.out
#SBATCH --error=logs/sft_multi_%j.err

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== C5v3-A: Multi-dataset SFT (German + Australian) ==="
nvidia-smi -L

cd "$REPOSITORY_ROOT"
mkdir -p logs outputs/sft
"$PYTHON_BIN" -m src.training.sft_multi

echo "--- Inference ---"
ADAPTER_DIR=outputs/sft/german_multi
"$PYTHON_BIN" -m src.training.pref_infer --adapter "$ADAPTER_DIR/best_adapter" --output "$ADAPTER_DIR"

echo "--- Done ---"
