import json
from pathlib import Path
from typing import List, Dict

from torch.utils.data import Dataset

PROMPT_TEMPLATE = (
    "You are a helpful problem-solving assistant.\n"
    "Problem: {question}\n"
    "Please think step by step and enclose your final answer in the form \\boxed{{answer}}.\n"
    "Solution:"
)


class AIMEJsonl(Dataset):
    """Lazy-loads AIME JSONL and returns prompt strings."""

    def __init__(self, paths: List[Path]):
        self.examples: List[Dict[str, str]] = []
        for p in paths:
            with open(p) as f:
                for line in f:
                    obj = json.loads(line)
                    self.examples.append({
                        "id": obj.get("id", ""),
                        "prompt": PROMPT_TEMPLATE.format(question=obj["question"]),
                        "answer": obj.get("answer", "")
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DistillCollator:
    """Collates batches for distillation training.
    
    Uses dynamic padding (padding=True) to pad each batch to its longest sequence,
    minimizing wasted computation and memory. This is optimal for OOM reduction.
    """

    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        prompts = [ex["prompt"] for ex in batch]
        # Dynamic padding: pads to max length in THIS batch, not global max_len
        # This significantly reduces memory usage compared to always padding to max_len
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
