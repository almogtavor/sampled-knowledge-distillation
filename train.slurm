#!/bin/bash
#SBATCH --job-name=ekd-train
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=4
#SBATCH --output=logs/train.%j.log
#SBATCH --error=logs/train.%j.log
# Ensure W&B agent env vars propagate into job
#SBATCH --export=ALL

set -euo pipefail
cd /home/joberant/NLP_2425b/$USER/ekd

DISTILL_TYPE=${1:-"top-k-tok"}
K_PERCENT=${2:-20}
EVAL_SUITE=${3:-}
KD_SWEEP_TAG=${4:-}
EPOCHS="${5:-${EPOCHS:-3}}"   # default 3, override via env
ANNEAL_FLAG=${6:-""}
echo "Running training with distillation type: $DISTILL_TYPE"
START_TIME=$(date -u)
START_TIME_EPOCH=$(date +%s)
echo "Job started at $START_TIME"

# Convert hyphens to underscores for directory names
DISTILL_TYPE_DIR=$(echo "$DISTILL_TYPE" | tr '-' '_')
OUTPUT_DIR="/home/joberant/NLP_2425b/$USER/ekd/results/kd_${KD_SWEEP_TAG}_out/models/model_${SLURM_JOB_ID}"
EVAL_OUTPUT_DIR="/home/joberant/NLP_2425b/$USER/ekd/results/kd_${KD_SWEEP_TAG}_out/eval"

# Default virtual environment directory
VENV_DIR="$PWD/fastenv310_3_new"

echo "=== Job Parameters ==="
echo "SLURM_JOB_ID      = ${SLURM_JOB_ID:-N/A}"
echo "DISTILL_TYPE      = $DISTILL_TYPE"
echo "K_PERCENT         = $K_PERCENT"
echo "EVAL_SUITE        = ${EVAL_SUITE:-None}"
echo "VENV_DIR          = $VENV_DIR"
echo "OUTPUT_DIR        = $OUTPUT_DIR"
echo "User              = $USER"
echo "GPUs requested    = $SLURM_GPUS"
echo "======================"

# ---------- optional score-based selection overrides ----------
SCORE_TOKEN_SELECTION=${SCORE_TOKEN_SELECTION:-0}
SCORE_NORMALIZE=${SCORE_NORMALIZE:-}
SCORE_ENTROPY_WEIGHT=${SCORE_ENTROPY_WEIGHT:-}
SCORE_CE_WEIGHT=${SCORE_CE_WEIGHT:-}
SCORE_KL_WEIGHT=${SCORE_KL_WEIGHT:-}

EXTRA_ARGS=()
if [[ "$SCORE_TOKEN_SELECTION" == "1" ]]; then
  EXTRA_ARGS+=(--score_token_selection)
fi
if [[ -n "$SCORE_NORMALIZE" ]]; then
  EXTRA_ARGS+=(--score_normalize "$SCORE_NORMALIZE")
fi
if [[ -n "$SCORE_ENTROPY_WEIGHT" ]]; then
  EXTRA_ARGS+=(--score_entropy_weight "$SCORE_ENTROPY_WEIGHT")
fi
if [[ -n "$SCORE_CE_WEIGHT" ]]; then
  EXTRA_ARGS+=(--score_ce_weight "$SCORE_CE_WEIGHT")
fi
if [[ -n "$SCORE_KL_WEIGHT" ]]; then
  EXTRA_ARGS+=(--score_kl_weight "$SCORE_KL_WEIGHT")
fi

# ---------- caches / env ----------
mkdir -p logs
export TMPDIR="$PWD/tmp";               mkdir -p "$TMPDIR"
export HF_HOME="$TMPDIR/hf";            mkdir -p "$HF_HOME/hub" "$HF_HOME/datasets"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
export ACCELERATE_LOG_LEVEL=info
export TRANSFORMERS_VERBOSITY=info
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ---------- Weights & Biases ----------
export WANDB_PROJECT=${WANDB_PROJECT:-selective-entropy-knowledge-distillation}
export WANDB_ENTITY=${WANDB_ENTITY:-selective-entropy-knowledge-distillation}
export WANDB_START_METHOD=${WANDB_START_METHOD:-thread}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DISABLED=${WANDB_DISABLED:-false}
export WANDB_RESUME=${WANDB_RESUME:-allow}
# Group/name/id make the UI nice for shell-based sweeps
export KD_SWEEP_NAME="${KD_SWEEP_NAME:-$(date +%Y%m%d_%H%M)-$DISTILL_TYPE}"
export WANDB_GROUP="${WANDB_GROUP:-${KD_SWEEP_NAME:-manual}-$(echo "$DISTILL_TYPE" | tr '-' '_')-k${K_PERCENT}}"
export WANDB_NOTES="slurm_id=$SLURM_JOB_ID; k=$K_PERCENT; eval=${EVAL_SUITE:-none}; anneal=${ANNEAL_FLAG:-none}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${SLURM_JOB_ID}_${K_PERCENT}_${DISTILL_TYPE}}"

echo "=== Weights & Biases ==="
echo "W&B PROJECT        = $WANDB_PROJECT"
echo "W&B ENTITY         = ${WANDB_ENTITY:-<user>}"
echo "W&B GROUP          = $WANDB_GROUP"
echo "W&B RUN_ID         = $WANDB_RUN_ID"
echo "W&B MODE/DISABLED  = $WANDB_MODE / $WANDB_DISABLED"
echo "======================"

VENV_DIR="$PWD/fastenv310_3_new"

# Check if a custom venv name is passed as a parameter
for arg in "$@"; do
  if [[ "$arg" == --venv=* ]]; then
    VENV_DIR="${arg#--venv=}"
    break
  fi
done

# ---------- pick a python3 available on the node (prefer 3.10 for bnb/triton compat) ----------
PY_SYS="$(command -v python3.10 || command -v python3 || true)"
if [[ -z "${PY_SYS}" ]]; then
  echo "ERROR: python3 not found on this node."
  exit 1
fi
echo "System python: $PY_SYS ($($PY_SYS -V))"

# (Re)create venv if missing or broken
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating venv at $VENV_DIR with system python3..."
  "$PY_SYS" -m venv "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"
echo "Using virtual environment: $VENV_DIR"
$PY -V || { echo "Venv python missing"; exit 1; }

# ---------- ensure pip exists ----------
if ! $PY -m pip -V >/dev/null 2>&1; then
  echo "Bootstrapping pip in venv..."
  $PY -m ensurepip --upgrade || (curl -sS https://bootstrap.pypa.io/get-pip.py | $PY)
fi
$PY -m pip -V || true
$PY -m pip install -q --upgrade pip wheel setuptools

# --- Force coherent CUDA stack (Torch 2.2.2+cu118 / Triton 2.2.0 / BnB 0.43.3)
$PY - <<'PY' >/dev/null 2>&1
import sys, importlib
ok = False
try:
    import torch
    ok = torch.__version__.startswith("2.2.") and getattr(torch.version, "cuda", "").startswith("11.8")
except Exception:
    pass
sys.exit(0 if ok else 1)
PY
if [[ $? -ne 0 ]]; then
  echo "Reinstalling torch/cu118 + triton 2.2.0 + bitsandbytes 0.43.3..."
  $PY -m pip uninstall -y torch torchvision torchaudio triton bitsandbytes || true
  $PY -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
      torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118
  $PY -m pip install --no-cache-dir triton==2.2.0 bitsandbytes==0.43.3
fi


# ---------- install torch/cu118 if missing ----------
if ! $PY - <<'PY' >/dev/null 2>&1
import torch; print(torch.__version__)
PY
then
  echo "Installing torch/cu118 into venv..."
  $PY -m pip install --no-cache-dir --prefer-binary --only-binary=:all: \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118
fi

# ---------- install project requirements (skip torch* to avoid conflicts) ----------
if [[ -f requirements.txt ]]; then
  TMP_REQ="$TMPDIR/req.no.torch.txt"
  grep -v -E '^[[:space:]]*torch(|vision|audio)' requirements.txt > "$TMP_REQ" || true
  $PY -m pip install --no-cache-dir -r "$TMP_REQ"
fi

# Ensure triton and bitsandbytes are installed (defensive)
$PY - <<'PY' >/dev/null 2>&1
import importlib, sys
missing = []
for mod in ("triton", "bitsandbytes"):
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)
sys.exit(0 if not missing else 1)
PY
if [[ $? -ne 0 ]]; then
  echo "Installing triton==2.2.0 and bitsandbytes==0.43.3 (post-reqs)"
  $PY -m pip install --no-cache-dir triton==2.2.0 bitsandbytes==0.43.3
fi

# ---------- GPU reorder by free VRAM ----------
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

# ---------- quick sanity ----------
$PY - <<'PY' || true
import sys
print("sys.executable =", sys.executable)
try:
  import torch; print("torch =", torch.__version__, "cuda", getattr(torch.version, "cuda", None))
except Exception as e:
  print("torch import failed:", e)
PY

# ---------- launch ----------
unset PYTHONPATH PYTHONHOME
"$PY" ekd_distill.py \
  --teacher_model "Qwen/Qwen3-8B" \
  --student_model "Qwen/Qwen3-0.6B" \
  --distill_type "$DISTILL_TYPE" \
  --k_percent "$K_PERCENT" \
  --datasets "fineweb" \
  --fineweb_tokens "${FINEWEB_TOKENS:-1000000}" \
  --epochs "$EPOCHS" \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --checkpoint_steps 250 \
  --keep_checkpoints 3 \
  --max_seq_len 384 \
  --lr 1e-5 \
  --tensorboard_dir "tb/${DISTILL_TYPE}_experiment" \
  --output_dir "$OUTPUT_DIR" \
  --seed "${SEED:-1337}" \
  $(if [[ "${DETERMINISTIC:-0}" == "1" ]]; then echo "--deterministic"; fi) \
  $(if [[ "$ANNEAL_FLAG" == "anneal" ]]; then echo "--anneal_kd_temperature"; fi) \
  "${EXTRA_ARGS[@]}"

END_TIME=$(date -u)
END_TIME_EPOCH=$(date +%s)
ELAPSED=$((END_TIME_EPOCH - START_TIME_EPOCH))

echo "Job started at $START_TIME"
echo "Job finished at $END_TIME"
echo "Total elapsed time: $ELAPSED seconds"

if [[ -n "$EVAL_SUITE" ]]; then
  echo "Submitting eval for model at: $OUTPUT_DIR (suite=$EVAL_SUITE)"
  sbatch --dependency=afterok:${SLURM_JOB_ID} evals.slurm "$OUTPUT_DIR" "$EVAL_SUITE" "$EVAL_OUTPUT_DIR"
  echo "Eval submitted with dependency on train job ${SLURM_JOB_ID}"
else
  echo "No eval suite provided → skipping eval submission."
fi
