#!/bin/bash
#SBATCH --job-name=ekd-train
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ekd-train.%j.out
#SBATCH --error=logs/ekd-train.%j.err

set -euo pipefail

cd /home/joberant/NLP_2425b/$USER/ekd

echo "=== EKD TRAINING LOOP ==="
echo "Starting at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"

# Check GPU availability
nvidia-smi || echo "No GPU available"

# Use large project volume for temp and cache (same pattern as setup_fast.slurm)
mkdir -p /home/joberant/NLP_2425b/$USER/ekd/tmp
mkdir -p /home/joberant/NLP_2425b/$USER/ekd/tmp/.pip-cache
export TMPDIR=/home/joberant/NLP_2425b/$USER/ekd/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export PIP_CACHE_DIR=/home/joberant/NLP_2425b/$USER/ekd/tmp/.pip-cache

# Set HuggingFace cache directories
export HF_HOME=$TMPDIR/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$HF_HOME/torch

# Create cache directories
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME $PIP_CACHE_DIR logs

# Avoid user-site leaks and noisy pip checks
export PYTHONNOUSERSITE=1
export PIP_USER=no
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1

# Activate the fastenv environment
echo "Activating fastenv..."
if [ -f /home/joberant/NLP_2425b/$USER/ekd/fastenv/bin/activate ]; then
    source /home/joberant/NLP_2425b/$USER/ekd/fastenv/bin/activate
    echo "✅ Virtual environment activated"
    
    # Verify environment
    echo "Python executable: $(which python)"
    echo "Python version: $(python --version)"
    
    # Test PyTorch installation
    echo "Testing PyTorch..."
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
else
    echo "❌ fastenv not found at /home/joberant/NLP_2425b/$USER/ekd/fastenv/bin/activate"
    exit 1
fi

# Show what we're about to run
echo "Contents of run_train.sh:"
cat run_train.sh || echo "run_train.sh not found!"
echo "--- End of run_train.sh ---"

# Training loop - keep trying until run_train.sh succeeds
attempt=1
max_attempts=10

while [ $attempt -le $max_attempts ]; do
    echo "=== Training Attempt $attempt at $(date) ==="
    
    # Try running run_train.sh with full output
    echo "Running: bash run_train.sh"
    if bash run_train.sh 2>&1; then
        echo "✅ SUCCESS! run_train.sh completed successfully on attempt $attempt"
        echo "Training finished at $(date)"
        exit 0
    else
        exit_code=$?
        echo "❌ Attempt $attempt failed with exit code $exit_code"
        echo "--- Last 20 lines of any available logs ---"
        
        # Show recent log files if they exist
        find logs/ -name "*.out" -o -name "*.err" 2>/dev/null | head -3 | while read logfile; do
            echo "=== $logfile ==="
            tail -20 "$logfile" 2>/dev/null || echo "Could not read $logfile"
        done
        
        if [ $attempt -lt $max_attempts ]; then
            echo "Waiting 30 seconds before retry..."
            sleep 30
        fi
    fi
    
    attempt=$((attempt + 1))
done

echo "❌ All $max_attempts attempts failed"
echo "Final attempt finished at $(date)"
exit 1
