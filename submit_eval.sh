#!/usr/bin/env bash
# Usage: ./submit_eval.sh <vanilla|ekd> <checkpoint_filename.pt> [base_model_dir] [wait_secs=90]
set -euo pipefail

RUN_TAG=${1:?need tag vanilla|ekd}
CKPT_FILE=${2:?need checkpoint filename}
BASE_MODEL_DIR=${3:-"Qwen/Qwen3-0.6B"}
WAIT_SECS=${4:-90}

submit_and_wait () {
  local gpus=$1
  local wait_secs=$2
  echo "→ Trying with $gpus GPU(s)..."
  JOBID=$(sbatch --gpus="$gpus" evals.slurm "$RUN_TAG" "$CKPT_FILE" "$BASE_MODEL_DIR" | awk '{print $NF}')
  echo "Submitted job $JOBID (gpus=$gpus). Waiting up to ${wait_secs}s..."

  deadline=$((SECONDS + wait_secs))
  while squeue -j "$JOBID" -h -o %T >/dev/null 2>&1; do
    state=$(squeue -j "$JOBID" -h -o %T)
    if [[ "$state" == "RUNNING" || "$state" == "R" ]]; then
      echo "✓ Job $JOBID is RUNNING with $gpus GPU(s)."
      exit 0
    fi
    if (( SECONDS >= deadline )); then
      echo "✗ Job $JOBID still pending after ${wait_secs}s, cancelling..."
      scancel "$JOBID" || true
      return 1
    fi
    sleep 5
  done

  echo "Job $JOBID disappeared from queue unexpectedly."
  return 1
}

# Try 3 GPUs, then 2, then 1
submit_and_wait 3 "$WAIT_SECS" || \
submit_and_wait 2 "$WAIT_SECS" || \
submit_and_wait 1 "$WAIT_SECS" || \
echo "All attempts failed."
