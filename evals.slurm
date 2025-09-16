#!/bin/bash
#SBATCH --job-name=ekd-eval
#SBATCH --partition=studentkillable
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1                     # default to 1 so the job can always start
#SBATCH --output=logs/eval.%j.log
#SBATCH --error=logs/eval.%j.log

set -euo pipefail
cd /home/joberant/NLP_2425b/$USER/ekd

# ---------- paths / caches ----------
export TMPDIR="$PWD/tmp";                  mkdir -p "$TMPDIR"
export TMP="$TMPDIR"; export TEMP="$TMPDIR"
export HF_HOME="$PWD/huggingface";         mkdir -p "$HF_HOME" "$HF_HOME/hub" "$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$HF_HOME/torch"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ---------- use the pinned training venv ----------
VENV_DIR="$PWD/fastenv310_3_new"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv missing at $VENV_DIR. Run train.slurm once to create a pinned env."
  exit 1
fi
source "$VENV_DIR/bin/activate"
PY="$VENV_DIR/bin/python"

echo "Python: $($PY -V)"
$PY - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available(), "| torch", torch.__version__, "| cuda", getattr(torch.version, "cuda", None))
PY

# ---------- install eval deps (idempotent) ----------
echo "Installing evaluation dependencies (idempotent)..."
pip install -q lm-eval lighteval evalplus wandb tensorboard python-dotenv || true

# ---------- reorder GPUs by free VRAM and advertise to the script ----------
if command -v nvidia-smi >/dev/null 2>&1; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_SET <<< "$CUDA_VISIBLE_DEVICES"
    ORDERED=$(for gi in "${GPU_SET[@]}"; do
      FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gi" 2>/dev/null | head -n1 || echo 0)
      echo "$FREE:$gi"
    done | sort -t: -k1,1nr | awk -F: '{print $2}' | paste -sd, -)
    export CUDA_VISIBLE_DEVICES="$ORDERED"
  else
    export CUDA_VISIBLE_DEVICES="$(
      nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
      | nl -v0 | sort -k2,2nr | awk '{print $1}' | paste -sd, -
    )"
  fi
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "=== GPU VRAM Information ==="
  IFS=',' read -r -a GPU_ARR <<< "$CUDA_VISIBLE_DEVICES"
  for gid in "${GPU_ARR[@]}"; do
    NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$gid" 2>/dev/null || echo "Unknown")
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gid" 2>/dev/null || echo 0)
    TOTL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gid" 2>/dev/null || echo 0)
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gid" 2>/dev/null || echo 0)
    echo "GPU $gid ($NAME): ${FREE} MiB free / ${TOTL} MiB total (${USED} MiB used)"
  done
  echo "=============================="
fi

# ---------- args ----------
# Usage: sbatch evals.slurm <vanilla|ekd> <checkpoint_filename.pt> [base_model_dir]
RUN_TAG=${1:-}
CKPT_FILE=${2:-}
BASE_MODEL_DIR=${3:-"Qwen/Qwen3-0.6B"}

if [[ -z "$RUN_TAG" || -z "$CKPT_FILE" ]]; then
  echo "Error: missing args. Usage: sbatch evals.slurm <vanilla|ekd> <checkpoint_filename.pt> [base_model_dir]"
  exit 2
fi
if [[ "$RUN_TAG" != "vanilla" && "$RUN_TAG" != "ekd" ]]; then
  echo "Error: RUN_TAG must be 'vanilla' or 'ekd'"
  exit 2
fi

VANILLA_DIR=kd_vanilla_run_out_model/checkpoints
EKD_DIR=kd_ekd_run_out_model/checkpoints
CKPT_DIR=$([[ "$RUN_TAG" == "vanilla" ]] && echo "$VANILLA_DIR" || echo "$EKD_DIR")
CKPT_PATH="$CKPT_DIR/$CKPT_FILE"

echo "Evaluating tag=$RUN_TAG checkpoint=$CKPT_PATH base=$BASE_MODEL_DIR"

mkdir -p logs eval_runs

# ---------- run (fan-out across whatever GPUs were allocated) ----------
EVAL_ARGS=(
  --base_model_dir "$BASE_MODEL_DIR"
  --work_dir eval_runs
  --tag "$RUN_TAG"
  --checkpoint_path "$CKPT_PATH"
  --gpu_ids "$CUDA_VISIBLE_DEVICES"
  --wandb_project selective-entropy-knowledge-distillation
)

srun -u "$PY" ekd/evaluations/eval.py "${EVAL_ARGS[@]}" | tee "eval_runs/${RUN_TAG}.${SLURM_JOB_ID}.log"

# extract LaTeX table if present
awk '/\\begin{table}/, /\\end{table}/' "eval_runs/${RUN_TAG}.${SLURM_JOB_ID}.log" > "eval_runs/${RUN_TAG}.table.${SLURM_JOB_ID}.tex" || true
echo "Done. Logs: eval_runs/${RUN_TAG}.${SLURM_JOB_ID}.log"
