"""
Entropy-guided Knowledge-Distillation (EKD) for LLM reasoning
==============================================================

This script provides **two** distillation modes:

* **vanilla** — KL between teacher & student on **all** tokens.
* **ekd** — KL computed **only** on *fork tokens*, i.e. those whose
  teacher-entropy is in the top-`--top_k_percent` percentile inside each
  example.

It is designed to run on a single consumer-GPU (e.g. GTX-1080 Ti, 11 GB)
by:
* Using Ollama's Qwen3-8B in 4-bit as teacher
* Loading student in 8-bit (bitsandbytes) with CPU offload.
* Allowing small (≤1.3 B) student models (125 M-0.7 B) that comfortably fit.

Benchmarks
----------
Out-of-the-box the script supports the **AIME-2024** and **AIME-2025**
math competitions.  Prepare each dataset as a JSONL file:

    {
      "id": "AIME24_q1",
      "question": "Find the sum of ...",
      "answer": "123"
    }

Place them in a directory and pass `--datasets aime24.jsonl aime25.jsonl`.
The script auto-prompts the teacher to generate a chain-of-thought (CoT)
followed by the boxed final answer, caches the result, and then trains the
student by teacher-forcing.

Quick start (Linux / Python ≥3.10)
----------------------------------
::

    # Linux:
    source .venv/bin/activate
    # Windows
    .venv/Scripts/activate
    uv pip install -e .

    # Start Ollama with Qwen3-8B
    ollama run qwen3:8b

    # vanilla KD
    python ekd_distill.py \
        --teacher_model qwen3:8b \
        --student_model OpenAssistant/falcon-180m \
        --distill_type vanilla \
        --datasets data/aime24.jsonl data/aime25.jsonl \
        --output_dir ./kd_vanilla_run

    # entropy-guided KD
    python ekd_distill.py \
        --teacher_model qwen3:8b \
        --student_model OpenAssistant/falcon-180m \
        --distill_type ekd \
        --top_k_percent 20 \
        --datasets data/aime24.jsonl data/aime25.jsonl \
        --output_dir ./kd_ekd_run

Note ▸ On a GTX-1080 Ti keep `batch_size * seq_len ≲ 4 000` to fit memory.
"""

import argparse
from pathlib import Path

import torch
from transformers.utils import is_torch_cuda_available

from ekd.data.dataset import AIMEJsonl, DistillCollator
from ekd.models.loader import load_model
from ekd.models.ollama_loader import OllamaModel
from ekd.training.distiller import Distiller

def main():
    parser = argparse.ArgumentParser(description="Entropy-guided KD for LLMs")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--distill_type", choices=["vanilla", "ekd"], default="vanilla")
    parser.add_argument("--top_k_percent", type=int, default=20, help="for EKD only")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if not is_torch_cuda_available():
        raise RuntimeError("CUDA-enabled PyTorch required for 1080 Ti training.")

    device = torch.device("cuda")

    print("Loading teacher (Ollama)...")
    teacher = OllamaModel(args.teacher_model)
    tok = teacher.tokenizer  # Use teacher's tokenizer for both models

    print("Loading student...")
    student, _ = load_model(args.student_model, device_map="auto")
    student.resize_token_embeddings(len(tok))  # align vocab if needed

    # build DataLoader
    dataset = AIMEJsonl([Path(p) for p in args.datasets])
    collate = DistillCollator(tok, args.max_seq_len)
    dl = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    distiller = Distiller(
        teacher_model=teacher,
        student_model=student,
        tokenizer=tok,
        dataloader=dl,
        distill_type=args.distill_type,
        top_k_percent=args.top_k_percent,
        lr=args.lr,
        device=device,
    )

    distiller.train(epochs=args.epochs)

    print("Saving student to", args.output_dir)
    student.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main() 