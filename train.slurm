#!/bin/bash
#SBATCH --job-name=ekd-train
#SBATCH --partition=studentkillable       # or studentbatch if allowed
#SBATCH --time=1-00:00:00                 # adjust
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1                          # request a GPU
#SBATCH --output=logs/train.%j.log
#SBATCH --error=logs/train.%j.log

set -euo pipefail
cd /home/joberant/NLP_2425b/$USER/ekd

# temp + pip
export TMPDIR=/vol/joberant_nobck/data/NLP_368307701_2425b/almogt/ekd/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export PIP_CACHE_DIR=/vol/joberant_nobck/data/NLP_368307701_2425b/almogt/ekd/.pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# 🤗 caches
export HF_HOME=/vol/joberant_nobck/data/NLP_368307701_2425b/almogt/ekd/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=/vol/joberant_nobck/data/NLP_368307701_2425b/almogt/ekd/hf-datasets
export TORCH_HOME=$HF_HOME/torch
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

# activate and run
source fastenv310/bin/activate
python -c "import torch; print('CUDA?', torch.cuda.is_available())"
srun -u bash run_train.sh   # or whatever your run_train.sh calls
