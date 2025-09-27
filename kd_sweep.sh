#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  bash kd_sweep.sh compare_k"
  echo "  bash kd_sweep.sh compare_methods <k_percent>"
}

[[ $# -lt 1 ]] && usage && exit 1

MODE="$1"
K_FIXED="${2:-20}"

if [[ "$MODE" == "compare_k" ]]; then
  # Sweep k = 0..100 step 10 (top-k-tok, except k=0 as vanilla)
  for K in 0 20 50; do
    if [[ "$K" -eq 0 ]]; then
      sbatch train.slurm vanilla "$K"
    else
      sbatch train.slurm top-k-tok "$K"
    fi
  done

elif [[ "$MODE" == "compare_methods" ]]; then
  # Run all three methods at fixed k
  for METHOD in vanilla top-k-tok random rs-kd; do
    sbatch train.slurm "$METHOD" "$K_FIXED"
  done

else
  usage; exit 1
fi
