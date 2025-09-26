#!/bin/bash
set -euo pipefail

# This script is called by train.slurm.
# The virtual environment should already be activated.

# Get distillation type parameter (default to "ekd" if not provided)
DISTILL_TYPE=${1:-"ekd"}
echo "Training with distillation type: $DISTILL_TYPE"

# Set output directory based on distillation type
OUTPUT_DIR="/home/joberant/NLP_2425b/$USER/ekd/kd_${DISTILL_TYPE}_run_out_model"

# The python script with all its arguments.
# Adjust arguments here as needed.
python3 ekd_distill.py \
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