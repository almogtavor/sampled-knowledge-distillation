#!/bin/bash
#SBATCH --job-name=ekd-ablate
#SBATCH --partition=studentkillable
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --output=logs/ablate.%j.log
#SBATCH --error=logs/ablate.%j.log

## Usage: see CONTRIBUTING.md → "Entropy Ablation (Top‑k overlap)" for clean examples.

set -euo pipefail
cd /home/joberant/NLP_2425b/$USER/ekd

# ---------- gather args (forward unknown flags to the Python tool) ----------
USER_ARGS=("$@")

# ---------- paths / caches ----------
mkdir -p logs
export TMPDIR="$PWD/tmp";               mkdir -p "$TMPDIR"
export HF_HOME="$TMPDIR/hf";            mkdir -p "$HF_HOME/hub" "$HF_HOME/datasets"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# ---------- venv ----------
VENV_DIR="$PWD/fastenv310_3_new"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv missing at $VENV_DIR. Run train.slurm once or create it manually."
  exit 1
fi
source "$VENV_DIR/bin/activate"
PY="$VENV_DIR/bin/python"
$PY -V

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
fi

# Resolve a default device for the tool (cuda:0 inside the containerized view)
DEVICE_ARG=("--device" "cuda:0")

# ---------- run ----------
unset PYTHONPATH PYTHONHOME
exec "$PY" tools/ablate_entropy_topk_agreement.py ${DEVICE_ARG[@]} "${USER_ARGS[@]}"
