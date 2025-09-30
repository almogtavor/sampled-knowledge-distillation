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
KD_SWEEP_TAG=$(date +%Y%m%d_%H%M)-$MODE

# One label to group all runs from this invocation
export KD_SWEEP_NAME="${KD_SWEEP_NAME:-$(date +%Y%m%d_%H%M)-$MODE}"
# Optional W&B defaults propagated to jobs
export WANDB_PROJECT="${WANDB_PROJECT:-selective-entropy-knowledge-distillation}"
# export WANDB_ENTITY="your_team"

echo Starting kd_sweep="$KD_SWEEP_TAG" jobs

if [[ "$MODE" == "compare_k" ]]; then
  METHOD="${2:-top-k-tok}"
  # Sweep k = 0..100 step 10 (top-k-tok, except k=100 as vanilla)
  # Run sequentially: wait for each job to finish before submitting the next
  for K in 0 1 2 5 10 12 15 20 25 30 40 50 75 100; do
    if [[ "$K" -eq 100 ]]; then
      echo "[compare_k] Submitting vanilla K=$K and waiting for completion..."
      sbatch --wait train.slurm vanilla "$K" light "$KD_SWEEP_TAG"
    else
      echo "[compare_k] Submitting $METHOD K=$K and waiting for completion..."
      sbatch --wait train.slurm "$METHOD" "$K" light "$KD_SWEEP_TAG"
    fi
  done

elif [[ "$MODE" == "coarse_k" ]]; then
  # --- New mode: quick 1-epoch pass over a default K list
  METHOD="${2:-top-k-tok}"
  K_LIST=(0 1 2 5 10 12 15 20 25 30 40 50 75 100)
  # Submit with EPOCHS=1 to override train.slurm default
  for K in "${K_LIST[@]}"; do
    if [[ "$K" -eq 100 ]]; then
      sbatch --export=ALL,EPOCHS=1 train.slurm vanilla "$K" "$KD_SWEEP_TAG"
    else
      sbatch --export=ALL,EPOCHS=1 train.slurm "$METHOD" "$K" "$KD_SWEEP_TAG"
    fi
  done


elif [[ "$MODE" == "compare_methods" ]]; then
  # Run all three methods at fixed k
  for METHOD in vanilla top-k-tok random pos-rs-kd; do
    sbatch train.slurm "$METHOD" "$K_FIXED" "$KD_SWEEP_TAG"
  done

elif [[ "$MODE" == "anneal_compare_methods" ]]; then
  # Run all methods with temperature annealing enabled
  for METHOD in vanilla top-k-tok random pos-rs-kd; do
    sbatch train.slurm "$METHOD" "$K_FIXED" "$KD_SWEEP_TAG" anneal
  done

elif [[ "$MODE" == "anneal_method" ]]; then
  # Run a specific method with annealing at fixed k
  [[ $# -lt 3 ]] && usage && exit 1
  METHOD="$2"
  K_ARG="$3"
  sbatch train.slurm "$METHOD" "$K_ARG" "$KD_SWEEP_TAG" anneal

elif [[ "$MODE" == "score_weights" ]]; then
  # Sweep score-based top-k weighting combinations at fixed K
  # ex: `bash kd_sweep.sh score_weights 25`
  K_SCORE="${2:-25}"
  METHOD="top-k-tok"
  SCORE_NORM_DEFAULT="${SCORE_NORMALIZE_OVERRIDE:-z}"
  read -r -d '' SCORE_COMBOS <<'EOF'
e_only 1.0 0.0 0.0
balanced 1.0 1.0 1.0
entropy_kl 1.0 0.0 1.0
ce_heavy 0.5 1.5 0.0
kl_heavy 0.5 0.0 1.5
kl_tilt 0.0 0.5 1.5
moderate 0.8 0.8 0.4
EOF
  while read -r LABEL W_ENT W_CE W_KL; do
    [[ -z "$LABEL" ]] && continue
    echo "[score_weights] Submitting $LABEL weights (ent=$W_ENT ce=$W_CE kl=$W_KL)"
    sbatch --export=ALL,SCORE_TOKEN_SELECTION=1,SCORE_NORMALIZE=$SCORE_NORM_DEFAULT,SCORE_ENTROPY_WEIGHT=$W_ENT,SCORE_CE_WEIGHT=$W_CE,SCORE_KL_WEIGHT=$W_KL,WANDB_GROUP=${KD_SWEEP_NAME:-$KD_SWEEP_TAG}-score-$LABEL train.slurm "$METHOD" "$K_SCORE" "" "$KD_SWEEP_TAG"
  done <<< "$SCORE_COMBOS"

else
  usage; exit 1
fi
