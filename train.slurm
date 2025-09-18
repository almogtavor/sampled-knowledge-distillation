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

set -euo pipefail
cd /home/joberant/NLP_2425b/$USER/ekd

DISTILL_TYPE=${1:-"ekd"}
echo "Running training with distillation type: $DISTILL_TYPE"

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

VENV_DIR="$PWD/fastenv310_3_new"

# ---------- pick a python3 available on the node ----------
PY_SYS="$(command -v python3 || true)"
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
OUTPUT_DIR="/home/joberant/NLP_2425b/$USER/ekd/kd_${DISTILL_TYPE}_run_out_model"
unset PYTHONPATH PYTHONHOME
exec "$PY" ekd_distill.py \
  --teacher_model "Qwen/Qwen3-8B" \
  --student_model "Qwen/Qwen3-0.6B" \
  --distill_type "$DISTILL_TYPE" \
  --k_percent 20 \
  --datasets "gsm8k" \
  --dataset_config "main" \
  --prompt_col "question" \
  --answer_col "answer" \
  --epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --checkpoint_steps 250 \
  --keep_checkpoints 3 \
  --max_seq_len 384 \
  --lr 1e-5 \
  --tensorboard_dir "tb/${DISTILL_TYPE}_experiment" \
  --output_dir "$OUTPUT_DIR"