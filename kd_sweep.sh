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
  # Sweep k = 0..100 step 10 (top-k-tok, except k=100 as vanilla)
  for K in 1 2 5 10 12 15 20 25 30 40 50 75 100; do
    if [[ "$K" -eq 100 ]]; then
      sbatch train.slurm vanilla "$K" light
    else
      sbatch train.slurm top-k-tok "$K" light
    fi
  done

elif [[ "$MODE" == "compare_methods" ]]; then
  # Run all three methods at fixed k
  for METHOD in vanilla top-k-tok random pos-rs-kd; do
    sbatch train.slurm "$METHOD" "$K_FIXED"
  done

else
  usage; exit 1
fi
