#!/usr/bin/env bash
# Usage: ./submit_eval.sh <vanilla|ekd> <checkpoint_filename.pt> {heavy|light} [base_model_dir] [wait_secs=90]
set -euo pipefail

RUN_TAG=${1:?need tag vanilla|ekd}
CKPT_FILE=${2:?need checkpoint filename}
SUITE=${3:?need suite heavy|light}
BASE_MODEL_DIR=${4:-"Qwen/Qwen3-0.6B"}
WAIT_SECS=${5:-90}

submit_and_wait () {
  local gpus=$1
  local wait_secs=$2
  echo "→ Trying with $gpus GPU(s)..."
  JOBID=$(sbatch --gpus="$gpus" evals.slurm "$RUN_TAG" "$CKPT_FILE" "$SUITE" "$BASE_MODEL_DIR" | awk '{print $NF}')
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

# Examples:
# ./submit_eval.sh ekd my_checkpoint.pt light Qwen/Qwen3-0.6B
# ./submit_eval.sh ekd my_checkpoint.pt heavy Qwen/Qwen3-0.6B 120
