import json
from pathlib import Path
from typing import List, Dict

import torch

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
    """Collates packed token windows into batched tensors."""

    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        def _to_tensor(values, dtype):
            if torch.is_tensor(values):
                return values.to(dtype=dtype)
            return torch.tensor(values, dtype=dtype)

        input_ids = torch.stack([_to_tensor(item["input_ids"], torch.long) for item in batch])
        attention_mask = torch.stack([
            _to_tensor(item.get("attention_mask", torch.ones(self.max_len)), torch.long)
            for item in batch
        ])

        labels = torch.stack([
            _to_tensor(item.get("labels", item["input_ids"]), torch.long)
            for item in batch
        ])

        kd_mask = torch.stack([
            _to_tensor(item.get("kd_mask"), torch.bool)
            for item in batch
        ])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "kd_mask": kd_mask,
        }


class PackedTokenDataset(Dataset):
    """Concatenates documents with EOS tokens and chunks into fixed-length windows."""

    def __init__(self, texts: List[str], tokenizer, max_seq_len: int, drop_remainder: bool = True):
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id for packing")

        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.drop_remainder = bool(drop_remainder)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        stream: List[int] = []
        eos_id = tokenizer.eos_token_id
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(eos_id)
            stream.extend(tokens)

        if len(stream) == 0:
            self.input_ids = torch.empty((0, self.max_seq_len), dtype=torch.long)
            self.labels = torch.empty((0, self.max_seq_len), dtype=torch.long)
            self.kd_mask = torch.empty((0, self.max_seq_len), dtype=torch.bool)
            self.attention_mask = torch.empty((0, self.max_seq_len), dtype=torch.long)
            return

        total_len = len(stream)
        if drop_remainder:
            total_len = (total_len // self.max_seq_len) * self.max_seq_len
            stream = stream[:total_len]
        else:
            remainder = total_len % self.max_seq_len
            if remainder:
                pad_len = self.max_seq_len - remainder
                stream.extend([eos_id] * pad_len)
                total_len = len(stream)

        if total_len == 0:
            self.input_ids = torch.empty((0, self.max_seq_len), dtype=torch.long)
            self.labels = torch.empty((0, self.max_seq_len), dtype=torch.long)
            self.kd_mask = torch.empty((0, self.max_seq_len), dtype=torch.bool)
            self.attention_mask = torch.empty((0, self.max_seq_len), dtype=torch.long)
            return

        ids_tensor = torch.tensor(stream, dtype=torch.long)
        eos_mask = ids_tensor == eos_id
        boundary_mask = torch.zeros_like(ids_tensor, dtype=torch.bool)
        boundary_mask[1:] = eos_mask[:-1]
        valid_targets = ~(eos_mask | boundary_mask)

        labels_tensor = ids_tensor.clone()
        labels_tensor[~valid_targets] = -100

        num_windows = total_len // self.max_seq_len
        self.input_ids = ids_tensor.view(num_windows, self.max_seq_len)
        self.labels = labels_tensor.view(num_windows, self.max_seq_len)
        self.kd_mask = valid_targets.view(num_windows, self.max_seq_len)
        self.attention_mask = torch.ones_like(self.input_ids, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.input_ids.size(0))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "kd_mask": self.kd_mask[idx],
        }
