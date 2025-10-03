#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  bash kd_sweep.sh compare_k"
  echo "  bash kd_sweep.sh compare_methods <k_percent>"
  echo "  bash kd_sweep.sh anneal_compare_methods <k_percent>"
  echo "  bash kd_sweep.sh anneal_method <method> <k_percent>"
  echo "  bash kd_sweep.sh run_all_4m [suite]   # Runs the full 4m-token pipeline in the exact requested order"
  echo "  bash kd_sweep.sh run_gls_4m [suite]   # Runs GLS experiments with balanced scoring at k=25"
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
  for K in 10 12 15 20 25 30 40 50 75 100; do
    if [[ "$K" -eq 100 ]]; then
      echo "[compare_k] Submitting vanilla K=$K and waiting for completion..."
      sbatch --wait train.slurm vanilla "$K" light "$KD_SWEEP_TAG"
    else
      echo "[compare_k] Submitting $METHOD K=$K and waiting for completion..."
      sbatch --wait train.slurm "$METHOD" "$K" light "$KD_SWEEP_TAG"
    fi
  done

elif [[ "$MODE" == "compare_k_with_buckets" ]]; then
  # Run compare_k first, then buckets; both with 4M tokens
  export FINEWEB_TOKENS=4000000

  echo "[compare_k_with_buckets] === compare_k (top-k-tok) with FINEWEB_TOKENS=$FINEWEB_TOKENS ==="
  METHOD="${2:-top-k-tok}"
  for K in 0 1 2 5 10 12 15 20 25 30 40 50 75 100; do
    if [[ "$K" -eq 100 ]]; then
      echo "[compare_k_with_buckets/compare_k] Submitting vanilla K=$K and waiting for completion..."
      sbatch --wait train.slurm vanilla "$K" light "$KD_SWEEP_TAG"
    else
      echo "[compare_k_with_buckets/compare_k] Submitting $METHOD K=$K and waiting for completion..."
      sbatch --wait train.slurm "$METHOD" "$K" light "$KD_SWEEP_TAG"
    fi
  done

  echo "[compare_k_with_buckets] === buckets (bucket) with FINEWEB_TOKENS=$FINEWEB_TOKENS ==="
  METHOD="bucket"
  declare -a BUCKETS=(
    "5 15"
    "10 20"
    "5 30"
    "15 35"
    "15 50"
    "25 75"
  )
  for R in "${BUCKETS[@]}"; do
    read -r L U <<< "$R"
    export BUCKET_LOWER_PERCENT="$L"
    export BUCKET_UPPER_PERCENT="$U"
    TAG="${KD_SWEEP_TAG:-compare_k_with_buckets}_${L}-${U}"

    echo "[compare_k_with_buckets/buckets] Submitting $METHOD L=$L U=$U and waiting for completion..."
    sbatch --wait train.slurm "$METHOD" "0" light "$TAG"
  done


elif [[ "$MODE" == "coarse_k" ]]; then
  # --- New mode: quick 1-epoch pass over a default K list
  METHOD="${2:-top-k-tok}"
  K_LIST=(0 1 2 5 10 12 15 20 25 30 40 50 75 100)
  # Submit with EPOCHS=1 to override train.slurm default
  for K in "${K_LIST[@]}"; do
    if [[ "$K" -eq 100 ]]; then
      sbatch --export=ALL,EPOCHS=1 train.slurm vanilla "$K" light "$KD_SWEEP_TAG"
    else
      sbatch --export=ALL,EPOCHS=1 train.slurm "$METHOD" "$K" light "$KD_SWEEP_TAG"
    fi
  done

elif [[ "$MODE" == "compare_methods_gsm8k" ]]; then
  # Run all three methods at fixed k
  for METHOD in top-k-tok; do
    sbatch train.slurm "$METHOD" "$K_FIXED" light "$KD_SWEEP_TAG"
  done

elif [[ "$MODE" == "run_top_k_with_softmax" ]]; then
  # Run all three methods at fixed k
  for METHOD in entropy-top-k-with-softmax; do
    sbatch train.slurm "$METHOD" "$K_FIXED" light "$KD_SWEEP_TAG"
  done

elif [[ "$MODE" == "compare_methods" ]]; then
  # Run all three methods at fixed k
  for METHOD in vanilla top-k-tok random pos-rs-kd; do
    sbatch --wait --export=ALL,EPOCHS=1 train.slurm "$METHOD" "$K_FIXED" light "$KD_SWEEP_TAG"
  done

elif [[ "$MODE" == "anneal_compare_methods" ]]; then
  # Run all methods with temperature annealing enabled
  for METHOD in vanilla top-k-tok random pos-rs-kd; do
    sbatch train.slurm "$METHOD" "$K_FIXED" light "$KD_SWEEP_TAG" anneal
  done

elif [[ "$MODE" == "anneal_method" ]]; then
  # Run a specific method with annealing at fixed k
  [[ $# -lt 3 ]] && usage && exit 1
  METHOD="$2"
  K_ARG="$3"
  sbatch train.slurm "$METHOD" "$K_ARG" light "$KD_SWEEP_TAG" anneal

elif [[ "$MODE" == "score_weights" ]]; then
  # Sweep score-based top-k weighting combinations at fixed K
  # ex: `bash kd_sweep.sh score_weights 25`
  K_SCORE="${2:-25}"
  METHOD="top-k-tok"
  SCORE_NORM_DEFAULT="${SCORE_NORMALIZE_OVERRIDE:-z}"
  
  # Define weight combinations: label ent_weight ce_weight kl_weight
  declare -a SCORE_CONFIGS=(
    "e_only:1.0:0.0:0.0"
    "balanced:1.0:1.0:1.0"
    "entropy_kl:1.0:0.0:1.0"
    # "ce_heavy:0.5:1.5:0.0"
    # "kl_heavy:0.5:0.0:1.5"
    # "kl_tilt:0.0:0.5:1.5"
    # "moderate:0.8:0.8:0.4"
  )
  
  for CONFIG in "${SCORE_CONFIGS[@]}"; do
    IFS=':' read -r LABEL W_ENT W_CE W_KL <<< "$CONFIG"
    echo "[score_weights] Submitting $LABEL weights (ent=$W_ENT ce=$W_CE kl=$W_KL)"
    sbatch --wait --export=ALL,SCORE_TOKEN_SELECTION=1,SCORE_NORMALIZE=$SCORE_NORM_DEFAULT,SCORE_ENTROPY_WEIGHT=$W_ENT,SCORE_CE_WEIGHT=$W_CE,SCORE_KL_WEIGHT=$W_KL,WANDB_GROUP=${KD_SWEEP_NAME:-$KD_SWEEP_TAG}-score-$LABEL train.slurm "$METHOD" "$K_SCORE" "light" "$KD_SWEEP_TAG"
  done


elif [[ "$MODE" == "eval_qwens" ]]; then
  SUITE="${2:-light}"

  export FINEWEB_TOKENS=4000000

  echo "[eval_qwens] Evaluating baselines (suite=$SUITE)"
  sbatch --wait evals.slurm "Qwen/Qwen3-0.6B" "$SUITE" from_hf
  sbatch --wait evals.slurm "Qwen/Qwen3-8B" "$SUITE" from_hf

  echo "[eval_qwens] All jobs submitted and completed in sequence."

elif [[ "$MODE" == "run_all_4m" ]]; then
  # Full pipeline: evaluate baselines, then run the requested trainings in EXACT order.
  # Optional arg 2: suite for evals (default: light)
  SUITE="${2:-light}"

  # Use 4m tokens for all FineWeb-Edu training runs
  export FINEWEB_TOKENS=4000000

  echo "[run_all_4m] Evaluating baselines (suite=$SUITE)"
  # Pass a mode flag 'from_hf' to tell evals to treat MODEL_PATH as a HF hub ID
  # sbatch --wait evals.slurm "Qwen/Qwen3-8B" "$SUITE" from_hf
  # sbatch --wait evals.slurm "Qwen/Qwen3-0.6B" "$SUITE" from_hf

  echo "[run_all_4m] Training runs (4M tokens each)"
  # 1) FullKD Qwen8B->0.6B on 4M tokens Basline
  # sbatch --export=ALL,NO_ELIMINATE_SOFTMAX=1,NO_OFFLINE=1 train.slurm top-k-tok 100 light "$KD_SWEEP_TAG"

  # 2) Token Selective KD k=25
  sbatch --export=ALL,NO_ELIMINATE_SOFTMAX=1,NO_OFFLINE=1 train.slurm top-k-tok 25 light "$KD_SWEEP_TAG"

  # 3) Token Selective KD k=75
  sbatch --export=ALL,NO_ELIMINATE_SOFTMAX=1,NO_OFFLINE=1 train.slurm top-k-tok 75 light "$KD_SWEEP_TAG"

  # 4) Token Selective KD Bucket top 5% of the tokens to top 25%
  sbatch --export=ALL,BUCKET_LOWER_PERCENT=5,BUCKET_UPPER_PERCENT=25,NO_ELIMINATE_SOFTMAX=1,NO_OFFLINE=1 train.slurm bucket 0 light "$KD_SWEEP_TAG"

  # 6) Token Selective KD Random k=25 Baseline
  sbatch --export=ALL,NO_ELIMINATE_SOFTMAX=1,NO_OFFLINE=1 train.slurm random 25 light "$KD_SWEEP_TAG"

  # 7) SampledKD k=25 (top-k with cached elimination)
  sbatch train.slurm top-k-tok 25 light "$KD_SWEEP_TAG"

  # 8) SampledKD k=25 with softmax (no cache/no elim)
  sbatch --export=ALL,NO_ELIMINATE_SOFTMAX=1 train.slurm top-k-tok 25 light "$KD_SWEEP_TAG"

  # 9) SampledKD k=75
  sbatch train.slurm top-k-tok 75 light "$KD_SWEEP_TAG"

  # 11) SampledKD Pos RS-KD k=25
  sbatch train.slurm pos-rs-kd 25 light "$KD_SWEEP_TAG"

  # # 12) SampledKD Trained also on GSM8K k=25 (GSM8K-only run)
  # # Note: train.slurm supports DATASETS / DATASET_CONFIG / PROMPT_COL / ANSWER_COL overrides
  # sbatch --export=ALL,DATASETS="gsm8k",DATASET_CONFIG="main",PROMPT_COL="question",ANSWER_COL="answer" \
  #   train.slurm top-k-tok 25 light "$KD_SWEEP_TAG"

  # 13) SampledKD LinUCB k=25
  sbatch train.slurm linucb 25 light "$KD_SWEEP_TAG"

  # 10) SampledKD Score k=25 (enable score-based selection)
  sbatch --export=ALL,SCORE_TOKEN_SELECTION=1 train.slurm top-k-tok 25 light "$KD_SWEEP_TAG"

  echo "[run_all_4m] All jobs submitted and completed in sequence."


elif [[ "$MODE" == "run_sampledkd_4m" ]]; then
  # Full pipeline: evaluate baselines, then run the requested trainings in EXACT order.
  # Optional arg 2: suite for evals (default: light)
  SUITE="${2:-light}"

  # Use 4m tokens for all FineWeb-Edu training runs
  export FINEWEB_TOKENS=4000000

  echo "[run_all_4m] Evaluating baselines (suite=$SUITE)"
  # 7) SampledKD k=100 (top-k with cached elimination)
  sbatch --wait train.slurm top-k-tok 100 light "$KD_SWEEP_TAG"

  # 8) SampledKD k=100 with softmax (no cache/no elim)
  sbatch --wait --export=ALL,NO_ELIMINATE_SOFTMAX=1,NO_OFFLINE=1 train.slurm top-k-tok 100 light "$KD_SWEEP_TAG"
  
  # 9) SampledKD k=25
  sbatch --wait train.slurm top-k-tok 25 light "$KD_SWEEP_TAG"

  # 9) SampledKD k=75
  sbatch --wait train.slurm top-k-tok 75 light "$KD_SWEEP_TAG"

  # 10) SampledKD Score k=25 (enable score-based selection)
  # sbatch --wait --export=ALL,SCORE_TOKEN_SELECTION=1 train.slurm top-k-tok 25 light "$KD_SWEEP_TAG"

  # 11) SampledKD Pos RS-KD k=25
  sbatch --wait train.slurm pos-rs-kd 25 light "$KD_SWEEP_TAG"

  # 12) SampledKD Trained also on GSM8K k=25 (GSM8K-only run)
  # Note: train.slurm supports DATASETS / DATASET_CONFIG / PROMPT_COL / ANSWER_COL overrides
  sbatch --wait --export=ALL,DATASETS="gsm8k",DATASET_CONFIG="main",PROMPT_COL="question",ANSWER_COL="answer" \
    train.slurm top-k-tok 25 light "$KD_SWEEP_TAG"

  # 13) SampledKD LinUCB k=25
  sbatch --wait train.slurm linucb 25 light "$KD_SWEEP_TAG"

  echo "[run_all_4m] All jobs submitted and completed in sequence."

elif [[ "$MODE" == "run_gls_4m" ]]; then
  # GLS (Global-Level Selection) experiments with balanced scoring
  # Optional arg 2: suite for evals (default: light)
  SUITE="${2:-light}"

  # Use 4m tokens for all FineWeb-Edu training runs
  export FINEWEB_TOKENS=4000000

  echo "[run_gls_4m] Running GLS experiments with balanced scoring (4M tokens each, suite=$SUITE)"
  
  # 1) GLS top-k-tok k=25 with balanced score-based selection (entropy:1.0, ce:1.0, kl:1.0)
  echo "[run_gls_4m] 1) GLS top-k-tok k=25 with balanced scoring"
  sbatch --wait --export=ALL,GLS_ENABLED=1 train.slurm top-k-tok 25 light "$KD_SWEEP_TAG"

  # 2) Standard (per-example) top-k-tok k=25 with balanced score-based selection for comparison
  echo "[run_gls_4m] 2) Standard top-k-tok k=25 with balanced scoring (no GLS)"
  sbatch --wait --export=ALL,SCORE_TOKEN_SELECTION=1,SCORE_NORMALIZE=z,SCORE_ENTROPY_WEIGHT=1.0,SCORE_CE_WEIGHT=1.0,SCORE_KL_WEIGHT=1.0 \
    train.slurm top-k-tok 25 light "$KD_SWEEP_TAG"

  echo "[run_gls_4m] All GLS jobs submitted and completed in sequence."

else
  usage; exit 1
fi
  