#!/bin/bash
#SBATCH --job-name=ekd-train
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1                 # request a GPU
#SBATCH --output=logs/train.%j.log
#SBATCH --error=logs/train.%j.log

set -euo pipefail
cd /home/joberant/NLP_2425b/$USER/ekd

# Get distillation type parameter (default to "ekd" if not provided)
DISTILL_TYPE=${1:-"ekd"}

echo "Running training with distillation type: $DISTILL_TYPE"

# temp + pip
export TMPDIR=/home/joberant/NLP_2425b/$USER/ekd/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export PIP_CACHE_DIR=/home/joberant/NLP_2425b/$USER/ekd/.pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# 🤗 caches
export HF_HOME=/home/joberant/NLP_2425b/$USER/ekd/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=/home/joberant/NLP_2425b/$USER/ekd/hf-datasets
export TORCH_HOME=$HF_HOME/torch
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

# activate and run
source fastenv310_2/bin/activate
python -c "import torch; print('CUDA?', torch.cuda.is_available())"
srun -u bash run_train.sh "$DISTILL_TYPE"
