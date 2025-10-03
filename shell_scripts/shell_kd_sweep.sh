#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  bash kd_sweep_local.sh compare_k [method]"
  echo "  bash kd_sweep_local.sh compare_k_with_buckets [method]"
  echo "  bash kd_sweep_local.sh coarse_k [method]"
  echo "  bash kd_sweep_local.sh compare_methods_gsm8k <k_percent>"
  echo "  bash kd_sweep_local.sh run_top_k_with_softmax <k_percent>"
  echo "  bash kd_sweep_local.sh compare_methods <k_percent>"
  echo "  bash kd_sweep_local.sh anneal_compare_methods <k_percent>"
  echo "  bash kd_sweep_local.sh anneal_method <method> <k_percent>"
  echo "  bash kd_sweep_local.sh score_weights <k_percent>"
  echo "  bash kd_sweep_local.sh eval_qwens [suite]"
  echo "  bash kd_sweep_local.sh run_all_4m [suite]"
  echo "  bash kd_sweep_local.sh run_sampledkd_4m [suite]"
  echo "  bash kd_sweep_local.sh run_gls_4m [suite]"
}

[[ $# -lt 1 ]] && usage && exit 1

MODE="$1"
K_FIXED="${2:-20}"
KD_SWEEP_TAG="$(date +%Y%m%d_%H%M)-$MODE"

# One label to group all runs from this invocation
export KD_SWEEP_NAME="${KD_SWEEP_NAME:-$(date +%Y%m%d_%H%M)-$MODE}"
# Optional W&B defaults propagated to jobs
export WANDB_PROJECT="${WANDB_PROJECT:-selective-entropy-knowledge-distillation}"
# export WANDB_ENTITY="your_team"

TRAIN="./shell_train_onegpu.sh"
EVAL="./shell_evals_onegpu.sh"
[[ -x "$TRAIN" ]] || { echo "ERROR: $TRAIN not found or not executable"; exit 2; }
[[ -x "$EVAL"   ]] || { echo "ERROR: $EVAL not found or not executable";   exit 2; }

echo "Starting kd_sweep='$KD_SWEEP_TAG' jobs"

if [[ "$MODE" == "compare_k" ]]; then
  METHOD="${2:-top-k-tok}"
  for K in 10 12 15 20 25 30 40 50 75 100; do
    if [[ "$K" -eq 100 ]]; then
      echo "[compare_k] vanilla K=$K"
      bash "$TRAIN" vanilla "$K" light "$KD_SWEEP_TAG"
    else
      echo "[compare_k] $METHOD K=$K"
      bash "$TRAIN" "$METHOD" "$K" light "$KD_SWEEP_TAG"
    fi
  done

elif [[ "$MODE" == "score_weights" ]]; then
  K_SCORE="${2:-25}"
  METHOD="top-k-tok"
  SCORE_NORM_DEFAULT="${SCORE_NORMALIZE_OVERRIDE:-z}"
  declare -a SCORE_CONFIGS=(
    "e_only:1.0:0.0:0.0"
    "balanced:1.0:1.0:1.0"
    "entropy_kl:1.0:0.0:1.0"
  )
  for CONFIG in "${SCORE_CONFIGS[@]}"; do
    IFS=':' read -r LABEL W_ENT W_CE W_KL <<< "$CONFIG"
    echo "[score_weights] $LABEL (ent=$W_ENT ce=$W_CE kl=$W_KL)"
    SCORE_TOKEN_SELECTION=1 \
    SCORE_NORMALIZE="$SCORE_NORM_DEFAULT" \
    SCORE_ENTROPY_WEIGHT="$W_ENT" \
    SCORE_CE_WEIGHT="$W_CE" \
    SCORE_KL_WEIGHT="$W_KL" \
    WANDB_GROUP="${KD_SWEEP_NAME:-$KD_SWEEP_TAG}-score-$LABEL" \
    bash "$TRAIN" "$METHOD" "$K_SCORE" "light" "$KD_SWEEP_TAG"
  done

elif [[ "$MODE" == "eval_qwens" ]]; then
  SUITE="${2:-light}"
  export FINEWEB_TOKENS=4000000
  echo "[eval_qwens] Evaluating baselines (suite=$SUITE)"
  bash "$EVAL" "Qwen/Qwen3-0.6B" "$SUITE" from_hf
  bash "$EVAL" "Qwen/Qwen3-8B" "$SUITE" from_hf
  echo "[eval_qwens] Done."

elif [[ "$MODE" == "run_all_4m" ]]; then
  SUITE="${2:-light}"
  export FINEWEB_TOKENS=4000000
  bash "$EVAL" "Qwen/Qwen3-0.6B" "$SUITE" from_hf
  bash "$EVAL" "Qwen/Qwen3-8B" "$SUITE" from_hf
  echo "[run_all_4m] Training runs (4M tokens each; suite=$SUITE)"
  NO_ELIMINATE_SOFTMAX=1 NO_OFFLINE=1 bash "$TRAIN" top-k-tok 100 light "$KD_SWEEP_TAG"
  NO_ELIMINATE_SOFTMAX=1 NO_OFFLINE=1 bash "$TRAIN" top-k-tok 25 light "$KD_SWEEP_TAG"
  NO_ELIMINATE_SOFTMAX=1 NO_OFFLINE=1 bash "$TRAIN" top-k-tok 75 light "$KD_SWEEP_TAG"
  BUCKET_LOWER_PERCENT=5 BUCKET_UPPER_PERCENT=25 NO_ELIMINATE_SOFTMAX=1 NO_OFFLINE=1 bash "$TRAIN" bucket 0 light "$KD_SWEEP_TAG"
  NO_ELIMINATE_SOFTMAX=1 NO_OFFLINE=1 bash "$TRAIN" random 25 light "$KD_SWEEP_TAG"
  bash "$TRAIN" top-k-tok 25 light "$KD_SWEEP_TAG"
  NO_ELIMINATE_SOFTMAX=1 bash "$TRAIN" top-k-tok 25 light "$KD_SWEEP_TAG"
  bash "$TRAIN" top-k-tok 75 light "$KD_SWEEP_TAG"
  bash "$TRAIN" pos-rs-kd 25 light "$KD_SWEEP_TAG"
  GLS_ENABLED=1 bash "$TRAIN" top-k-tok 25 light "$KD_SWEEP_TAG"
  bash "$TRAIN" linucb 25 light "$KD_SWEEP_TAG"
  SCORE_TOKEN_SELECTION=1 bash "$TRAIN" top-k-tok 25 light "$KD_SWEEP_TAG"
  echo "[run_all_4m] Done."

elif [[ "$MODE" == "run_sampledkd_4m" ]]; then
  SUITE="${2:-light}"
  export FINEWEB_TOKENS=4000000

  echo "[run_sampledkd_4m] Sequence start (4M tokens each; suite=$SUITE)"
  bash "$TRAIN" top-k-tok 100 light "$KD_SWEEP_TAG"
  NO_ELIMINATE_SOFTMAX=1 NO_OFFLINE=1 bash "$TRAIN" top-k-tok 100 light "$KD_SWEEP_TAG"
  bash "$TRAIN" top-k-tok 25 light "$KD_SWEEP_TAG"
  bash "$TRAIN" top-k-tok 75 light "$KD_SWEEP_TAG"
  # SCORE_TOKEN_SELECTION=1 bash "$TRAIN" top-k-tok 25 light "$KD_SWEEP_TAG"
  bash "$TRAIN" pos-rs-kd 25 light "$KD_SWEEP_TAG"
  DATASETS="gsm8k" DATASET_CONFIG="main" PROMPT_COL="question" ANSWER_COL="answer" \
    bash "$TRAIN" top-k-tok 25 light "$KD_SWEEP_TAG"
  bash "$TRAIN" linucb 25 light "$KD_SWEEP_TAG"
  echo "[run_sampledkd_4m] Done."

elif [[ "$MODE" == "run_gls_4m" ]]; then
  SUITE="${2:-light}"
  export FINEWEB_TOKENS=4000000

  echo "[run_gls_4m] GLS experiments (4M tokens each; suite=$SUITE)"
  GLS_ENABLED=1 bash "$TRAIN" top-k-tok 25 light "$KD_SWEEP_TAG"
  SCORE_TOKEN_SELECTION=1 SCORE_NORMALIZE=z SCORE_ENTROPY_WEIGHT=1.0 SCORE_CE_WEIGHT=1.0 SCORE_KL_WEIGHT=1.0 \
    bash "$TRAIN" top-k-tok 25 light "$KD_SWEEP_TAG"
  echo "[run_gls_4m] Done."

else
  usage; exit 1
fi
