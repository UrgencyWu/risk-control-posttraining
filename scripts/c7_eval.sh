#!/bin/bash
#SBATCH --job-name=c7_eval
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/c7_%j.out
#SBATCH --error=logs/c7_%j.err

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== C7: Leakage-safe evaluation of frozen predictions ==="
cd "$REPOSITORY_ROOT"
"$PYTHON_BIN" -m src.evaluation.c7_final

echo "--- Done ---"
