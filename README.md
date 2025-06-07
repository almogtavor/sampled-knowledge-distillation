# Entropy-guided Knowledge-Distillation (EKD)

Knowledge distillation for LLMs that focuses on high-entropy "fork" tokens where the teacher model makes important decisions. Uses Ollama's Qwen3-8B in 4-bit as the teacher model.

## Quick Start

1. Install dependencies:
```bash
uv venv
# Linux:
source .venv/bin/activate
# Windows
.venv/Scripts/activate
uv pip install -e .
```

2. Start Ollama with Qwen3-8B:
```bash
ollama run qwen3:8b
```

3. Run distillation:
```bash
# Vanilla KD
python ekd_distill.py \
    --teacher_model qwen3:8b \
    --student_model OpenAssistant/falcon-180m \
    --distill_type vanilla \
    --datasets data/aime24.jsonl data/aime25.jsonl \
    --output_dir ./kd_vanilla_run

# Entropy-guided KD
python ekd_distill.py \
    --teacher_model qwen3:8b \
    --student_model OpenAssistant/falcon-180m \
    --distill_type ekd \
    --top_k_percent 20 \
    --datasets data/aime24.jsonl data/aime25.jsonl \
    --output_dir ./kd_ekd_run
```

## Requirements

- Python ≥3.10
- CUDA-enabled GPU (e.g. GTX-1080 Ti)
- Ollama installed and running with Qwen3-8B
- Memory: Keep `batch_size * seq_len ≲ 4 000` to fit in 11GB VRAM

## Dataset Format

Prepare your dataset as JSONL files:
```json
{
  "id": "example_1",
  "question": "Your question here...",
  "answer": "123"
}
```