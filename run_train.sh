#!/bin/bash
set -euo pipefail

# This script is called by train.slurm.
# The virtual environment should already be activated.

# Get distillation type parameter (default to "ekd" if not provided)
DISTILL_TYPE=${1:-"ekd"}
echo "Training with distillation type: $DISTILL_TYPE"

OUTPUT_DIR="/home/joberant/NLP_2425b/$USER/ekd/kd_${DISTILL_TYPE}_run_out_model"

# Common arguments
COMMON_ARGS=(
  --teacher_model "Qwen/Qwen3-8B"
  --student_model "Qwen/Qwen3-0.6B"
  --distill_type "$DISTILL_TYPE"
  --top_k_percent 20
  --datasets "gsm8k"
  --dataset_config "main"
  --prompt_col "question"
  --answer_col "answer"
  --epochs 3
  --batch_size 1
  --gradient_accumulation_steps 16
  --checkpoint_steps 250
  --keep_checkpoints 3
  --max_seq_len 384
  --lr 1e-5
  --tensorboard_dir "tb/${DISTILL_TYPE}_experiment"
  --output_dir "$OUTPUT_DIR"
  --teacher_quant_bits 4        # NEW: safer on 11GB GPUs
)

# Detect if torchrun is already set by slurm script
if [[ -n "${WORLD_SIZE:-}" && "${WORLD_SIZE}" -gt 1 ]]; then
  echo "Launching with NPROC=$WORLD_SIZE"
  torchrun --standalone --nproc_per_node="$WORLD_SIZE" ekd_distill.py "${COMMON_ARGS[@]}"
else
  echo "Launching single process training"
  python3 ekd_distill.py "${COMMON_ARGS[@]}"
fi
