#!/usr/bin/env python3
"""Print the tokenizer vocabulary size for Qwen/Qwen3-0.6B."""

from __future__ import annotations

from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"{MODEL_NAME} vocabulary size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
