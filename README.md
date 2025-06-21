# Entropy-guided Knowledge-Distillation (EKD)

Knowledge distillation for LLMs that focuses on high-entropy "fork" tokens where the teacher model makes important decisions. Uses Ollama's Qwen3-8B in 4-bit as the teacher model.

## Quick Start

1. Install dependencies (you have to install [uv](https://docs.astral.sh/uv/) first):
```bash
uv venv
# Linux:
source .venv/bin/activate
# Windows
.venv/Scripts/activate
pip install uv
uv pip install -e .

# Installing CUDA dependencies (adjust to your CUDA version):
uv pip install .[cu118]
```

2. Install [Ollama](https://ollama.com/download) & Start with Qwen3-8B:
```bash
ollama run qwen3:8b
```

3. Run distillation:

We provide two distillation types:
* **vanilla** — KL between teacher & student on **all** tokens.
* **ekd** — KL computed **only** on *fork tokens*, i.e. those whose
  teacher-entropy is in the top-`--top_k_percent` percentile inside each
  example.

We use Ollama's Qwen3-8B in 4-bit as teacher:
```bash
# Vanilla KD
python ekd_distill.py \
    --teacher_model qwen3:8b \
    --student_model Qwen/Qwen3-0.6B \
    --distill_type vanilla \
    --datasets data/aime24.jsonl data/aime25.jsonl \
    --output_dir ./kd_vanilla_run

# Entropy-guided KD
python ekd_distill.py \
    --teacher_model qwen3:8b \
    --student_model Qwen/Qwen3-0.6B \
    --distill_type ekd \
    --top_k_percent 20 \
    --datasets data/aime24.jsonl data/aime25.jsonl \
    --output_dir ./kd_ekd_run
```

## Requirements

- Python >=3.10
- CUDA-enabled GPU
- Ollama installed and running with Qwen3-8B
- Memory: Keep `batch_size * seq_len < 4000` to fit in low GPU memory (11GB VRAM).

## Benchmarks
Out-of-the-box the script supports the **AIME-2024** and **AIME-2025**
math benchmarks. We prepare each dataset as a JSONL file:

```json
{
  "id": "AIME24_q1",
  "question": "Find the sum of ...",
  "answer": "123"
}
```

Place them in a directory and pass `--datasets aime24.jsonl aime25.jsonl`.
The script prompts the teacher to generate a chain-of-thought (CoT)
followed by the boxed final answer, caches the result, and then trains the
student by teacher-forcing.


## Dataset Format

Prepare your dataset as JSONL files:
```json
{
  "id": "example_1",
  "question": "Your question here...",
  "answer": "123"
}
```