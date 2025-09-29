#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  bash kd_sweep.sh compare_k"
  echo "  bash kd_sweep.sh compare_methods <k_percent>"
  echo "  bash kd_sweep.sh anneal_compare_methods <k_percent>"
  echo "  bash kd_sweep.sh anneal_method <method> <k_percent>"
}

[[ $# -lt 1 ]] && usage && exit 1

MODE="$1"
K_FIXED="${2:-20}"

# One label to group all runs from this invocation
export KD_SWEEP_NAME="${KD_SWEEP_NAME:-$(date +%Y%m%d_%H%M)-$MODE}"
# Optional W&B defaults propagated to jobs
export WANDB_PROJECT="${WANDB_PROJECT:-selective-entropy-knowledge-distillation}"
# export WANDB_ENTITY="your_team"

if [[ "$MODE" == "compare_k" ]]; then
  METHOD="${2:-top-k-tok}"
  # Sweep k = 0..100 step 10 (top-k-tok, except k=100 as vanilla)
  for K in 0 1 2 5 10 12 15 20 25 30 40 50 75 100; do
    if [[ "$K" -eq 100 ]]; then
      sbatch train.slurm vanilla "$K" light
    else
      sbatch train.slurm "$METHOD" "$K" light
    fi
  done

elif [[ "$MODE" == "coarse_k" ]]; then
  # --- New mode: quick 1-epoch pass over a default K list
  METHOD="${2:-top-k-tok}"
  K_LIST=(0 1 2 5 10 12 15 20 25 30 40 50 75 100)
  # Submit with EKD_EPOCHS=1 to override train.slurm default
  for K in "${K_LIST[@]}"; do
    if [[ "$K" -eq 100 ]]; then
      sbatch --export=ALL,EKD_EPOCHS=1 train.slurm vanilla "$K"
    else
      sbatch --export=ALL,EKD_EPOCHS=1 train.slurm "$METHOD" "$K"
    fi
  done

elif [[ "$MODE" == "compare_methods" ]]; then
  # Run all three methods at fixed k
  for METHOD in vanilla top-k-tok random pos-rs-kd; do
    sbatch train.slurm "$METHOD" "$K_FIXED"
  done

elif [[ "$MODE" == "anneal_compare_methods" ]]; then
  # Run all methods with temperature annealing enabled
  for METHOD in vanilla top-k-tok random pos-rs-kd; do
    sbatch train.slurm "$METHOD" "$K_FIXED" anneal
  done

elif [[ "$MODE" == "anneal_method" ]]; then
  # Run a specific method with annealing at fixed k
  [[ $# -lt 3 ]] && usage && exit 1
  METHOD="$2"
  K_ARG="$3"
  sbatch train.slurm "$METHOD" "$K_ARG" anneal

else
  usage; exit 1
fi
