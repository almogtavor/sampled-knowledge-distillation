#!/bin/bash
#SBATCH --job-name=ekd-train
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=4                 # request four GPUs to allow teacher sharding
#SBATCH --output=logs/train.%j.log
#SBATCH --error=logs/train.%j.log

set -euo pipefail
cd /home/joberant/NLP_2425b/$USER/ekd

# Get distillation type parameter (default to "ekd" if not provided)
DISTILL_TYPE=${1:-"ekd"}

echo "Running training with distillation type: $DISTILL_TYPE"

# Create necessary directories and set essential environment variables
export TMPDIR=/home/joberant/NLP_2425b/$USER/ekd/tmp
mkdir -p "$TMPDIR" huggingface logs

# Use local tmp directory for fast offloads and caches  
export HF_HOME="$TMPDIR/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
export ACCELERATE_LOG_LEVEL=info
export TRANSFORMERS_VERBOSITY=info

# Prevent tokenizer hanging issues
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME/hub" "$HF_DATASETS_CACHE"

# Fallback cache to unlimited ekd storage (but scratch is preferred)
mkdir -p /home/joberant/NLP_2425b/$USER/ekd/huggingface/hub

# Use fresh minimal virtual environment  
VENV_DIR="$PWD/fastenv310_3_new"

if [[ -f "$VENV_DIR/bin/activate" ]] && [[ -x "$VENV_DIR/bin/python" ]]; then
    echo "Using virtual environment: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
    python -V
    pip -V
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Please create and install dependencies into fastenv310_3_new before running this job"
    exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MIN_FREE_MEM_MB=${MIN_FREE_MEM_MB:-9000}  # default for 12GB GPUs; override via env if needed
export REQUEUE_MAX_ATTEMPTS=${REQUEUE_MAX_ATTEMPTS:-15}
export REQUEUE_SLEEP_MIN=${REQUEUE_SLEEP_MIN:-20}
export REQUEUE_SLEEP_MAX=${REQUEUE_SLEEP_MAX:-60}

# Reorder allocated GPUs by free memory (descending) and export CUDA_VISIBLE_DEVICES accordingly
if command -v nvidia-smi >/dev/null 2>&1; then
	if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
		mapfile -t GPU_SET < <(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n')
		# Build an array of "freeMem:gpuIdx"
		ORDERED=$(for gi in "${GPU_SET[@]}"; do \
			FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gi" 2>/dev/null | head -n1 || echo 0); \
			echo "$FREE:$gi"; \
		done | sort -t: -k1,1nr | awk -F: '{print $2}' | paste -sd, -)
		export CUDA_VISIBLE_DEVICES="$ORDERED"
		echo "Reordered allocated GPUs by free mem: $CUDA_VISIBLE_DEVICES"
	else
		# No Slurm restriction; sort all GPUs
		export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | nl -v0 | sort -k2,2nr | awk '{print $1}' | paste -sd, -)
		echo "Selected GPUs by free mem: $CUDA_VISIBLE_DEVICES"
	fi

	# Log detailed GPU VRAM information
	echo "=== GPU VRAM Information ==="
	IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
	for gpu_id in "${GPU_ARRAY[@]}"; do
		GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "Unknown")
		FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo 0)
		TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo 0)
		USED_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo 0)
		echo "GPU $gpu_id ($GPU_NAME): ${FREE_MEM} MiB free / ${TOTAL_MEM} MiB total (${USED_MEM} MiB used)"
	done
	echo "=============================="

	# Check free memory threshold; requeue if too low
	MAX_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null | sort -nr | head -1 || echo 0)
	echo "Max free GPU memory among selected: ${MAX_FREE} MiB (threshold ${MIN_FREE_MEM_MB} MiB)"
	if [[ ${MAX_FREE:-0} -lt ${MIN_FREE_MEM_MB:-0} ]]; then
		REQUEUE_FILE="$TMPDIR/.requeue_count"
		COUNT=0
		if [[ -f "$REQUEUE_FILE" ]]; then COUNT=$(cat "$REQUEUE_FILE"); fi
		if [[ $COUNT -lt $REQUEUE_MAX_ATTEMPTS ]]; then
			COUNT=$((COUNT+1))
			echo $COUNT > "$REQUEUE_FILE"
			# random backoff between min/max
			DELAY=$((RANDOM % (REQUEUE_SLEEP_MAX-REQUEUE_SLEEP_MIN+1) + REQUEUE_SLEEP_MIN))
			echo "Insufficient free GPU memory; requeuing attempt $COUNT after ${DELAY}s..."
			sleep "$DELAY" || true
			scontrol requeue "$SLURM_JOB_ID" || true
			exit 0
		else
			echo "Free memory below threshold but requeue limit reached; continuing anyway."
		fi
	fi
fi

# Run training via run_train.sh
source "$VENV_DIR/bin/activate"
unset PYTHONPATH PYTHONHOME  # Clear any system Python paths
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

# Ensure Hugging Face cache environment variables are set for training
export HF_HOME=/home/joberant/NLP_2425b/$USER/ekd/huggingface
export HUGGINGFACE_HUB_CACHE=/home/joberant/NLP_2425b/$USER/ekd/huggingface/hub
export HF_DATASETS_CACHE=/home/joberant/NLP_2425b/$USER/ekd/huggingface/datasets

/bin/bash run_train.sh "$DISTILL_TYPE"
