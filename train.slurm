#!/bin/bash
#SBATCH --job-name=ekd-train
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=2
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
EPOCHS="${5:-${EPOCHS:-1}}"   # default 1, override via env
ANNEAL_FLAG=${6:-""}
echo "Running training with distillation type: $DISTILL_TYPE"
START_TIME=$(date -u)
START_TIME_EPOCH=$(date +%s)
echo "Job started at $START_TIME"

# Convert hyphens to underscores for directory names
DATE_TAG=$(date +%Y%m%d_%H%M)
DISTILL_TYPE_DIR=$(echo "$DISTILL_TYPE" | tr '-' '_')
OUTPUT_DIR="/home/joberant/NLP_2425b/$USER/ekd/results/kd_${KD_SWEEP_TAG}_out/models/model_${SLURM_JOB_ID}_${DATE_TAG}_${DISTILL_TYPE_DIR}_k${K_PERCENT}"
EVAL_OUTPUT_DIR="/home/joberant/NLP_2425b/$USER/ekd/results/kd_${KD_SWEEP_TAG}_out/eval"

# Default virtual environment directory
VENV_DIR="$PWD/fastenv310_3_new"

echo "=== Job Parameters ==="
echo "SLURM_JOB_ID      = ${SLURM_JOB_ID:-N/A}"
echo "DISTILL_TYPE      = $DISTILL_TYPE"
echo "K_PERCENT         = $K_PERCENT"
echo "EVAL_SUITE        = ${EVAL_SUITE:-None}"
echo "KD_SWEEP_TAG      = $KD_SWEEP_TAG"
echo "DATE_TAG          = $DATE_TAG"
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

# ---------- bucket selection overrides ----------
if [[ "$DISTILL_TYPE" == "bucket" ]]; then
  if [[ -n "${BUCKET_LOWER_PERCENT:-}" ]]; then
    EXTRA_ARGS+=(--bucket_lower_percent "${BUCKET_LOWER_PERCENT}")
  fi
  if [[ -n "${BUCKET_UPPER_PERCENT:-}" ]]; then
    EXTRA_ARGS+=(--bucket_upper_percent "${BUCKET_UPPER_PERCENT}")
  fi
fi

# ---------- caches / env ----------
mkdir -p logs
export TMPDIR="/home/joberant/NLP_2425b/$USER/ekd/tmp"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR"; export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$PWD/tmp/xdg_cache";   mkdir -p "$XDG_CACHE_HOME"
export HF_HOME="$TMPDIR/hf";            mkdir -p "$HF_HOME/hub" "$HF_HOME/datasets"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# W&B caches away from $HOME
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$TMP/wandb_cache}"; mkdir -p "$WANDB_CACHE_DIR"
export WANDB_DIR="${WANDB_DIR:-$TMP/wandb}";            mkdir -p "$WANDB_DIR"
export TORCH_HOME="$TMP/torch";          mkdir -p "$TORCH_HOME"
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
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$TMP/wandb_cache}"; mkdir -p "$WANDB_DATA_DIR"
# Explicitly redirect W&B cache (artifacts, uploads) away from $HOME
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
TRAIN_LOG=$(mktemp)
trap 'rm -f "$TRAIN_LOG"' EXIT
MODEL_FOR_EVAL="$OUTPUT_DIR"
set +e
"$PY" run_distillation.py \
  --teacher_model "Qwen/Qwen3-8B" \
  --student_model "Qwen/Qwen3-0.6B" \
  --distill_type "$DISTILL_TYPE" \
  --k_percent "$K_PERCENT" \
  --datasets "${DATASETS:-fineweb}" \
  --fineweb_tokens "${FINEWEB_TOKENS:-4000000}" \
  --epochs "$EPOCHS" \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  --max_seq_len 250 \
  --lr 1e-5 \
  --tensorboard_dir "tb/${DISTILL_TYPE}_experiment" \
  --output_dir "$OUTPUT_DIR" \
  --seed "${SEED:-1337}" \
  $(if [[ -n "${NO_OFFLINE+1}" ]]; then echo "--no_offline_cache"; fi) \
  $(if [[ "${GLS_ENABLED:-0}" == "1" ]]; then echo "--gls_enabled"; fi) \
  $(if [[ -n "${NO_ELIMINATE_SOFTMAX+1}" ]]; then echo "--no_eliminate_softmax"; fi) \
  $(if [[ "$DISTILL_TYPE" == "entropy-top-k-with-softmax" ]]; then echo "--no_offline_cache --no_eliminate_softmax"; fi) \
  $(if [[ "${DETERMINISTIC:-0}" == "1" ]]; then echo "--deterministic"; fi) \
  $(if [[ "$ANNEAL_FLAG" == "anneal" ]]; then echo "--anneal_kd_temperature"; fi) \
  $(if [[ -n "${DATASET_CONFIG:-}" ]]; then echo "--dataset_config ${DATASET_CONFIG}"; fi) \
  $(if [[ -n "${PROMPT_COL:-}" ]]; then echo "--prompt_col ${PROMPT_COL}"; fi) \
  $(if [[ -n "${ANSWER_COL:-}" ]]; then echo "--answer_col ${ANSWER_COL}"; fi) \
  "${EXTRA_ARGS[@]}" |& tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

if [[ $TRAIN_EXIT -eq 11 ]]; then
  echo "Detected existing training run with pending evaluation; training skipped."
  REG_LINE=$(grep -E "\\[registry\\] duplicate params_hash=" "$TRAIN_LOG" | tail -n1 || true)
  EXISTING_OUTPUT_DIR=""
  if [[ -n "$REG_LINE" ]]; then
    EXISTING_OUTPUT_DIR=${REG_LINE#*output_dir=}
  fi
  if [[ -n "$EXISTING_OUTPUT_DIR" ]]; then
    MODEL_FOR_EVAL="$EXISTING_OUTPUT_DIR"
  else
    echo "Warning: duplicate run reported but existing output_dir was not found; skipping eval submission." >&2
    MODEL_FOR_EVAL=""
  fi
elif [[ $TRAIN_EXIT -eq 10 ]]; then
  echo "Run already trained and evaluated; skipping eval submission."
  MODEL_FOR_EVAL=""
elif [[ $TRAIN_EXIT -ne 0 ]]; then
  echo "Training failed with exit code $TRAIN_EXIT" >&2
  exit $TRAIN_EXIT
fi

rm -f "$TRAIN_LOG"

END_TIME=$(date -u)
END_TIME_EPOCH=$(date +%s)
ELAPSED=$((END_TIME_EPOCH - START_TIME_EPOCH))

echo "Job started at $START_TIME"
echo "Job finished at $END_TIME"
echo "Total elapsed time: $ELAPSED seconds"

if [[ -n "$EVAL_SUITE" ]]; then
  if [[ -n "$MODEL_FOR_EVAL" ]]; then
    echo "Submitting eval for model at: $MODEL_FOR_EVAL (suite=$EVAL_SUITE)"
    sbatch --dependency=afterok:${SLURM_JOB_ID} evals.slurm "$MODEL_FOR_EVAL" "$EVAL_SUITE" "$EVAL_OUTPUT_DIR"
    echo "Eval submitted with dependency on train job ${SLURM_JOB_ID}"
  else
    echo "Skipping eval submission (no pending evaluation required)."
  fi
else
  echo "No eval suite provided → skipping eval submission."
fi

# Also save a copy of the log with a descriptive filename
SAFE_TYPE="$DISTILL_TYPE_DIR"
RICH_LOG="results/logs/train_${DATE_TAG}_${SAFE_TYPE}_k${K_PERCENT}_${SLURM_JOB_ID}.log"
SRC_LOG="logs/train.${SLURM_JOB_ID}.log"
if [[ -f "$SRC_LOG" ]]; then
  cp -f "$SRC_LOG" "$RICH_LOG" || true
  echo "Saved rich log copy to $RICH_LOG"
fi
