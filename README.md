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

We provide several distillation types:
* **vanilla** — KL between teacher & student on **all** tokens.
* **ekd** — KL computed **only** on *fork tokens*, i.e. those whose
  teacher-entropy is in the top-`--k_percent` percentile inside each
  example.
* **bucket** — KL computed on tokens with entropy in a specific percentile range,
  e.g., 70th-80th percentile (excludes both very low and very high entropy tokens).
* **linucb** — a contextual bandit (LinUCB) that observes a 6D feature vector per
  token (teacher entropy, teacher CE, student CE, KL, coarse POS bucket, normalized
  position) and learns online which tokens to distill. It enforces at least one
  token per sequence and logs overlap with the top-entropy quartile alongside reward
  improvements.

Add `--score_token_selection` to the **top-k-tok** or **bucket** modes to rank
tokens by a composite score that mixes teacher entropy, student
cross-entropy, and teacher-student KL. Tune the mix with
`--score_entropy_weight`, `--score_ce_weight`, `--score_kl_weight`, and pick a
normalization via `--score_normalize`.

We use Ollama's Qwen3-8B in 4-bit as teacher:
```bash
# Vanilla KD
python ekd_distill.py \
    --teacher_model Qwen/Qwen3-8B \
    --student_model Qwen/Qwen3-0.6B \
    --distill_type vanilla \
    --datasets gsm8k \
    --dataset_config main \
    --prompt_col question \
    --answer_col answer \
    --output_dir ./kd_vanilla_run

# Entropy-guided KD (top-k token selection)
python ekd_distill.py \
    --teacher_model Qwen/Qwen3-8B \
    --student_model Qwen/Qwen3-0.6B \
    --distill_type top-k-tok \
  --k_percent 20 \
    --datasets gsm8k \
    --dataset_config main \
    --prompt_col question \
    --answer_col answer \
    --output_dir ./kd_top_k_tok_run

# Bucket distillation (e.g., distill on 70th-80th percentile entropy tokens)
python ekd_distill.py \
    --teacher_model Qwen/Qwen3-8B \
    --student_model Qwen/Qwen3-0.6B \
    --distill_type bucket \
    --bucket_lower_percent 70 \
    --bucket_upper_percent 80 \
    --datasets gsm8k \
    --dataset_config main \
    --prompt_col question \
    --answer_col answer \
    --output_dir ./kd_bucket_run

# Top-k selection with score-based ranking (entropy + CE + KL)
python ekd_distill.py \
  --teacher_model Qwen/Qwen3-8B \
  --student_model Qwen/Qwen3-0.6B \
  --distill_type top-k-tok \
  --k_percent 20 \
  --score_token_selection \
  --score_entropy_weight 1.0 \
  --score_ce_weight 1.0 \
  --score_kl_weight 1.0 \
  --datasets gsm8k \
  --dataset_config main \
  --prompt_col question \
  --answer_col answer \
  --output_dir ./kd_score_run

# LinUCB contextual bandit distillation
python ekd_distill.py \
  --teacher_model Qwen/Qwen3-8B \
  --student_model Qwen/Qwen3-0.6B \
  --distill_type linucb \
  --datasets gsm8k \
  --dataset_config main \
  --prompt_col question \
  --answer_col answer \
  --bandit_alpha 0.75 \
  --bandit_threshold 0.0 \
  --bandit_min_tokens 1 \
  --output_dir ./kd_linucb_run
```

Tune LinUCB-specific knobs with `--bandit_alpha`, `--bandit_lambda`, `--bandit_threshold`,
and `--bandit_max_tokens`. The trainer logs per-step selection statistics
(`bandit/selected_tokens`, `bandit/overlap_selected`) and the average reward
achieved by the bandit after each optimizer update (`bandit/avg_reward`,
`bandit/positive_reward_rate`).

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

Prepare datasets as JSONL files:
```json
{
  "id": "example_1",
  "question": "Your question here...",
  "answer": "123"
}
```
