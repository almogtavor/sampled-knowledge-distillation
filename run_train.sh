#!/bin/bash
set -euo pipefail

# This script is called by train.slurm.
# The virtual environment should already be activated.

# The python script with all its arguments.
# Adjust arguments here as needed.
python ekd_distill.py \
    --teacher_model "Qwen/Qwen3-8B" \
    --student_model "Qwen/Qwen3-0.6B" \
    --distill_type "ekd" \
    --top_k_percent 20 \
    --datasets "gsm8k" \
    --dataset_config "main" \
    --prompt_col "question" \
    --answer_col "answer" \
    --epochs 2 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_seq_len 512 \
    --lr 1e-5 \
    --output_dir "/home/joberant/NLP_2425b/$USER/kd_ekd_run_out_model"