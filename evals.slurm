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

# ---------- force eval caches to TMPDIR (avoid $HOME quota) ----------  
export XDG_CACHE_HOME="$TMPDIR/xdg_cache";          mkdir -p "$XDG_CACHE_HOME"
export WANDB_CACHE_DIR="$TMPDIR/wandb_cache";       mkdir -p "$WANDB_CACHE_DIR"  
export WANDB_DIR="$TMPDIR/wandb";                   mkdir -p "$WANDB_DIR"
export EVALPLUS_TRUST_REMOTE_CODE=1

# Optional tiny extras in LIGHT suite (BoolQ, HumanEval) -> set LIGHT_EXTRAS=1 to enable
# export LIGHT_EXTRAS=1

# ---------- use the pinned training venv ----------
VENV_DIR="$PWD/fastenv310_3_new"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv missing at $VENV_DIR. Run train.slurm once to create a pinned env."
  exit 1
fi

source "$VENV_DIR/bin/activate"
PY="$VENV_DIR/bin/python"

# Check Python version but don't fail - just warn if not 3.10
PYTHON_VERSION=$($PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PYTHON_VERSION" == "3.10" ]]; then
  echo "✅ Python version verified: $PYTHON_VERSION"
elif [[ "$PYTHON_VERSION" == "3.12" ]]; then
  echo "⚠️  Using Python $PYTHON_VERSION (prefer 3.10 but 3.12 should work)"
else
  echo "⚠️  Using Python $PYTHON_VERSION (untested version)"
fi

echo "=== ENVIRONMENT DIAGNOSTICS ==="
echo "Date: $(date)"
echo "User: $USER"
echo "Working directory: $(pwd)"
echo "Python: $($PY -V)"
echo "Python executable: $(which python)"
echo "Python path: $($PY -c 'import sys; print(sys.executable)')"
echo "Pip version: $(pip --version)"
echo "Virtual env: $VIRTUAL_ENV"
echo "PATH: $PATH"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "SLURM_NODE_LIST: ${SLURM_NODE_LIST:-N/A}"
echo "SLURM_GPUS: ${SLURM_GPUS:-N/A}"
echo "================================"

$PY - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available(), "| torch", torch.__version__, "| cuda", getattr(torch.version, "cuda", None))
PY

# ---------- install/upgrade eval deps (idempotent) ----------
echo "Installing/upgrading evaluation dependencies from requirements.txt..."
echo "=== PIP INSTALL DIAGNOSTICS ==="
$PY -m pip install -q --upgrade pip wheel setuptools || {
    echo "WARNING: pip/wheel/setuptools upgrade failed"
  $PY -m pip --version
}

echo "Installing from requirements.txt..."
if ! $PY -m pip install -q --upgrade -r requirements.txt; then
    echo "ERROR: pip install failed. Showing detailed error:"
  $PY -m pip install --upgrade -r requirements.txt
    echo "Continuing anyway..."
fi

echo "=== POST-INSTALL PACKAGE CHECK ==="
echo "Key packages installed:"
$PY -m pip show lm-eval lighteval transformers torch | grep -E "Name|Version" || echo "Some packages missing"
echo "==================================="

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
# Usage: sbatch evals.slurm <vanilla|ekd> <checkpoint_filename.pt> <suite: heavy|light> [base_model_dir]
RUN_TAG=${1:-}
CKPT_FILE=${2:-}
SUITE=${3:-}
BASE_MODEL_DIR=${4:-"Qwen/Qwen3-0.6B"}

if [[ -z "$RUN_TAG" || -z "$CKPT_FILE" || -z "$SUITE" ]]; then
  echo "Error: missing args. Usage: sbatch evals.slurm <vanilla|ekd> <checkpoint_filename.pt> <suite: heavy|light> [base_model_dir]"
  exit 2
fi
if [[ "$RUN_TAG" != "vanilla" && "$RUN_TAG" != "ekd" ]]; then
  echo "Error: RUN_TAG must be 'vanilla' or 'ekd'"
  exit 2
fi
if [[ "$SUITE" != "heavy" && "$SUITE" != "light" ]]; then
  echo "Error: SUITE must be 'heavy' or 'light'"
  exit 2
fi

VANILLA_DIR=kd_vanilla_run_out_model/checkpoints
EKD_DIR=kd_ekd_run_out_model/checkpoints
CKPT_DIR=$([[ "$RUN_TAG" == "vanilla" ]] && echo "$VANILLA_DIR" || echo "$EKD_DIR")
CKPT_PATH="$CKPT_DIR/$CKPT_FILE"
echo "Evaluating tag=$RUN_TAG checkpoint=$CKPT_PATH base=$BASE_MODEL_DIR suite=$SUITE"
mkdir -p logs eval_runs

# ---------- run (fan-out across whatever GPUs were allocated) ----------
EVAL_ARGS=(
  --base_model_dir "$BASE_MODEL_DIR"
  --work_dir eval_runs
  --tag "$RUN_TAG"
  --checkpoint_path "$CKPT_PATH"
  --gpu_ids "$CUDA_VISIBLE_DEVICES"
  --suite "$SUITE"
  --wandb_project selective-entropy-knowledge-distillation
)
srun -u "$PY" ekd/evaluations/eval.py "${EVAL_ARGS[@]}" | tee "eval_runs/${RUN_TAG}.${SLURM_JOB_ID}.${SUITE}.log"

# extract LaTeX table if present
awk '/\\begin{table}/, /\\end{table}/' "eval_runs/${RUN_TAG}.${SLURM_JOB_ID}.${SUITE}.log" > "eval_runs/${RUN_TAG}.${SUITE}.table.${SLURM_JOB_ID}.tex" || true
echo "Done. Logs: eval_runs/${RUN_TAG}.${SLURM_JOB_ID}.${SUITE}.log"

# Example usage:
# sbatch evals.slurm vanilla latest_checkpoint.pt light Qwen/Qwen3-0.6B
# sbatch evals.slurm ekd latest_checkpoint.pt heavy Qwen/Qwen3-0.6B
