#!/bin/bash
#SBATCH --job-name=ekd-train
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4                 # one process per GPU
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=4                   # 4 GPUs on a single node
#SBATCH --output=logs/train.%j.log
#SBATCH --error=logs/train.%j.log

set -euo pipefail
cd /home/joberant/NLP_2425b/$USER/ekd

# Get distillation type parameter (default to "ekd" if not provided)
DISTILL_TYPE=${1:-"ekd"}
shift || true
EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  EXTRA_ARGS=("$@")
fi

echo "Running training with distillation type: $DISTILL_TYPE"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"

# Create necessary directories and set essential environment variables
export TMPDIR=/home/joberant/NLP_2425b/$USER/ekd/tmp
mkdir -p "$TMPDIR" huggingface logs

# Set Hugging Face cache directories to use unlimited ekd storage instead of home quota
export HF_HOME=/home/joberant/NLP_2425b/$USER/ekd/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE"

# Use fresh minimal virtual environment  
VENV_DIR="$PWD/fastenv310_3_new"
if [[ -f "$VENV_DIR/bin/activate" && -x "$VENV_DIR/bin/python" ]]; then
  echo "Using virtual environment: $VENV_DIR"
  source "$VENV_DIR/bin/activate"
  python -V
  pip -V
else
  echo "ERROR: Virtual environment not found at $VENV_DIR"
  exit 1
fi

# PyTorch & CUDA memory knobs (no expandable_segments on this GPU gen)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export BNB_CUDA_ALLOW_UNSUPPORTED_DEVICES=1

# ---------------------------
# GPU selection & reorder by free VRAM
# ---------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -r -a GPU_SET <<< "$CUDA_VISIBLE_DEVICES"
    ORDERED=$(for gi in "${GPU_SET[@]}"; do
      FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gi" | head -n1 || echo 0)
      echo "$FREE:$gi"
    done | sort -t: -k1,1nr | awk -F: '{print $2}' | paste -sd, -)
    export CUDA_VISIBLE_DEVICES="$ORDERED"
    echo "Reordered allocated GPUs by free mem: $CUDA_VISIBLE_DEVICES"
  else
    export CUDA_VISIBLE_DEVICES=$(
      nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
      | nl -v0 | sort -k2,2nr | awk '{print $1}' | paste -sd, -
    )
    echo "Selected GPUs by free mem: $CUDA_VISIBLE_DEVICES"
  fi

  echo "=== GPU VRAM Information ==="
  IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
  for gpu_id in "${GPU_ARRAY[@]}"; do
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits -i "$gpu_id" || echo "Unknown")
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" || echo 0)
    TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" || echo 0)
    USED_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" || echo 0)
    echo "GPU $gpu_id ($GPU_NAME): ${FREE_MEM} MiB free / ${TOTAL_MEM} MiB total (${USED_MEM} MiB used)"
  done
  echo "=============================="
fi

# ---------------------------
# NCCL / threading
# ---------------------------
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4
export TORCH_SHOW_CPP_STACKTRACES=1

# ---------------------------
# Launch training
# ---------------------------
unset PYTHONPATH PYTHONHOME

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "Launching torchrun with user-supplied args"
  torchrun --standalone --nproc_per_node="${SLURM_NTASKS_PER_NODE}" \
    ekd_distill.py \
    --distill_type "$DISTILL_TYPE" \
    "${EXTRA_ARGS[@]}"
else
  echo "No EXTRA_ARGS detected → using presets from run_train.sh (distill_type=$DISTILL_TYPE)"
  bash run_train.sh "$DISTILL_TYPE"
fi
