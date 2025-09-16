#!/bin/bash
#SBATCH --job-name=ekd-eval
#SBATCH --partition=studentkillable
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=logs/eval.%j.log
#SBATCH --error=logs/eval.%j.log

set -euo pipefail

# Usage: sbatch evals.slurm <vanilla|ekd> <checkpoint_filename.pt> [base_model_dir] [fast]
# Example: sbatch evals.slurm ekd checkpoint_epoch1_step4527.pt Qwen/Qwen2-0.5B
# Fast mode: sbatch evals.slurm ekd checkpoint_epoch1_step4527.pt Qwen/Qwen2-0.5B fast

cd /home/joberant/NLP_2425b/$USER/ekd

RUN_TAG=${1:-}
CKPT_FILE=${2:-}
BASE_MODEL_DIR=${3:-"Qwen/Qwen3-0.6B"}
FAST_MODE=${4:-}

if [[ -z "$RUN_TAG" || -z "$CKPT_FILE" ]]; then
	echo "Error: missing args. Usage: sbatch evals.slurm <vanilla|ekd> <checkpoint_filename.pt> [base_model_dir]"
	exit 2
fi

if [[ "$RUN_TAG" != "vanilla" && "$RUN_TAG" != "ekd" ]]; then
	echo "Error: RUN_TAG must be 'vanilla' or 'ekd'"
	exit 2
fi

# Directories
VANILLA_DIR=kd_vanilla_run_out_model/checkpoints
EKD_DIR=kd_ekd_run_out_model/checkpoints
CKPT_DIR=$([[ "$RUN_TAG" == "vanilla" ]] && echo "$VANILLA_DIR" || echo "$EKD_DIR")
CKPT_PATH="$CKPT_DIR/$CKPT_FILE"

echo "Evaluating tag=$RUN_TAG checkpoint=$CKPT_PATH base=$BASE_MODEL_DIR"

# Temps and caches
export TMPDIR=/home/joberant/NLP_2425b/$USER/ekd/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export PIP_CACHE_DIR=/home/joberant/NLP_2425b/$USER/ekd/.pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" logs eval_runs

export HF_HOME=/home/joberant/NLP_2425b/$USER/ekd/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=/home/joberant/NLP_2425b/$USER/ekd/hf-datasets
export TORCH_HOME=$HF_HOME/torch
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Activate env  
source fastenv310_3_new/bin/activate
python -c "import torch; print('CUDA?', torch.cuda.is_available())"

# Ensure dependencies for evaluation are present (best-effort)
echo "Installing evaluation dependencies..."
pip install lm-eval lighteval evalplus wandb tensorboard --quiet || echo "Some eval packages failed to install, continuing anyway"

# Build evaluation command with optional fast mode
EVAL_ARGS="--base_model_dir $BASE_MODEL_DIR --device cuda:0 --work_dir eval_runs --tag $RUN_TAG --checkpoint_path $CKPT_PATH"
if [[ "$FAST_MODE" == "fast" ]]; then
    EVAL_ARGS="$EVAL_ARGS --fast_eval"
    echo "Running FAST evaluation mode (subset of tasks)"
fi

# Run evaluation for the single checkpoint
srun -u python ekd/evaluations/eval.py $EVAL_ARGS | tee "eval_runs/${RUN_TAG}.${SLURM_JOB_ID}.log"

# Extract LaTeX table to a separate file if present in log
awk '/\\begin{table}/, /\\end{table}/' "eval_runs/${RUN_TAG}.${SLURM_JOB_ID}.log" > "eval_runs/${RUN_TAG}.table.${SLURM_JOB_ID}.tex" || true
echo "Done. Logs: eval_runs/${RUN_TAG}.${SLURM_JOB_ID}.log"

