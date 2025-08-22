#!/bin/bash
#SBATCH --job-name=setup-fast
#SBATCH --partition=studentkillable
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/setup-fast.%j.out
#SBATCH --error=logs/setup-fast.%j.err

set -euo pipefail

cd /home/joberant/NLP_2425b/$USER/ekd

# Use large project volume for temp + pip cache
mkdir -p /home/joberant/NLP_2425b/$USER/ekd/tmp
mkdir -p /home/joberant/NLP_2425b/$USER/ekd/tmp/.pip-cache
export TMPDIR=/home/joberant/NLP_2425b/$USER/ekd/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export PIP_CACHE_DIR=/home/joberant/NLP_2425b/$USER/ekd/tmp/.pip-cache

# Avoid user-site leaks and noisy pip checks
export PYTHONNOUSERSITE=1
export PIP_USER=no
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Use a new venv name to avoid conflicts
VENV_NAME="fastenv"

# Pick a consistent Python (prefer 3.12 if present)
if command -v python3.12 >/dev/null 2>&1; then
  PY_BIN=python3.12
else
  PY_BIN=python3
fi

if [ ! -x "$VENV_NAME/bin/python" ]; then
  echo "Creating real venv: $VENV_NAME"
  "$PY_BIN" -m venv "$VENV_NAME" --without-pip
  source "$VENV_NAME/bin/activate"

  # Verify temp configuration with venv's Python
  python - <<'PY'
import os, tempfile
print("TMPDIR=", os.getenv("TMPDIR"))
print("TEMP=", os.getenv("TEMP"))
print("TMP=", os.getenv("TMP"))
print("tempdir=", tempfile.gettempdir())
PY
  
  echo "Installing pip via curl bootstrap..."
  curl -sS https://bootstrap.pypa.io/get-pip.py | python
  
  # Verify pip installation
  if ! python -m pip --version >/dev/null 2>&1; then
    echo "pip not found, bootstrapping..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    rm -f get-pip.py
  fi

  # Upgrade pip and install basic packages
  python -m pip install --upgrade pip setuptools wheel
  
  # Detect CUDA version and set appropriate PyTorch index
  TORCH_IDX="https://download.pytorch.org/whl/cpu"
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n1)
    echo "Detected CUDA version: ${CUDA_VER:-unknown}"
    case "$CUDA_VER" in
      12.*) TORCH_IDX="https://download.pytorch.org/whl/cu124" ;;
      11.8*) TORCH_IDX="https://download.pytorch.org/whl/cu118" ;;
      *) echo "Unknown CUDA version, using CPU wheels" ;;
    esac
  else
    echo "No nvidia-smi found, using CPU wheels"
  fi
  
  echo "Installing packages from $TORCH_IDX ..."
  # Install PyTorch first from the appropriate index (fallback to CPU if it fails)
  if ! python -m pip install --prefer-binary --only-binary=:all: \
       --index-url "$TORCH_IDX" \
       torch; then
    echo "GPU wheel install failed, falling back to CPU wheels..."
    python -m pip install --prefer-binary --only-binary=:all: \
       --index-url https://download.pytorch.org/whl/cpu \
       torch
  fi
  
  # Install other packages from default index
  python -m pip install --prefer-binary transformers datasets accelerate sentencepiece rich tiktoken
  
  echo "Marking environment as ready..."
  touch .ekd_fastenv_ready
  
else
  echo "Venv $VENV_NAME already exists, activating..."
  source "$VENV_NAME/bin/activate"

  # Verify temp configuration with venv's Python
  python - <<'PY'
import os, tempfile
print("TMPDIR=", os.getenv("TMPDIR"))
print("TEMP=", os.getenv("TEMP"))
print("TMP=", os.getenv("TMP"))
print("tempdir=", tempfile.gettempdir())
PY

  # ensure pip exists in this venv
  python -m pip --version >/dev/null 2>&1 || { 
    python -m ensurepip --upgrade || { 
      curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
      python /tmp/get-pip.py
    }
  }
  python -m pip install -U pip setuptools wheel

  # detect CUDA and set TORCH_IDX (same logic as above)
  TORCH_IDX="https://download.pytorch.org/whl/cpu"
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n1)
    echo "Detected CUDA version: ${CUDA_VER:-unknown}"
    case "$CUDA_VER" in
      12.*) TORCH_IDX="https://download.pytorch.org/whl/cu124" ;;
      11.8*) TORCH_IDX="https://download.pytorch.org/whl/cu118" ;;
      *) echo "Unknown CUDA version, using CPU wheels" ;;
    esac
  else
    echo "No nvidia-smi found, using CPU wheels"
  fi

  # install torch if missing, with GPU index then CPU fallback
  python -c 'import torch' 2>/dev/null || {
    echo "torch not found, installing from $TORCH_IDX ..."
    python -m pip install --prefer-binary --only-binary=:all: \
      --index-url "$TORCH_IDX" torch \
      || python -m pip install --prefer-binary --only-binary=:all: \
         --index-url https://download.pytorch.org/whl/cpu torch
  }

  # install transformers stack if missing
  python -c 'import transformers' 2>/dev/null || {
    echo "transformers not found, installing ..."
    python -m pip install --prefer-binary transformers datasets accelerate sentencepiece rich tiktoken
  }
fi

# Test installation
echo "Testing PyTorch installation..."
python - <<'PY'
import torch, transformers
print("✓ PyTorch:", torch.__version__)
print("✓ Transformers:", transformers.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✓ CUDA device count:", torch.cuda.device_count())
    print("✓ Current device:", torch.cuda.current_device())
PY

echo "✓ Environment setup complete!"
echo "To use this environment in other jobs:"
echo "source /home/joberant/NLP_2425b/$USER/ekd/$VENV_NAME/bin/activate"
