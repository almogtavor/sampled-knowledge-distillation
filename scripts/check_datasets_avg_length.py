from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np  # optional, not required here
import os
from itertools import islice

# --- settings ---
hf_dataset = os.environ.get("FINEWEB_DATASET", "HuggingFaceFW/fineweb")  # or "HuggingFaceFW/fineweb-edu"
hf_config = os.environ.get("FINEWEB_CONFIG", None)  # e.g., "CC-MAIN-2024-18"
tokenizer_name = os.environ.get("TOKENIZER", "gpt2")  # replace with your training tokenizer
sample_size = int(os.environ.get("N_DOCS", "20000"))  # first N docs to stream
batch_size = int(os.environ.get("BATCH_SIZE", "128"))  # tokenize in batches to keep it fast
text_col = "text"

# --- tokenizer ---
tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
tok_kwargs = dict(add_special_tokens=False)

# --- streaming dataset (no full local download) ---
if hf_config is None:
    ds = load_dataset(hf_dataset, split="train", streaming=True)
else:
    ds = load_dataset(hf_dataset, hf_config, split="train", streaming=True)

def batched(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

# --- iterate over first sample_size examples and compute running average ---
total_tokens = 0
count = 0

for batch in batched(islice(ds, sample_size), batch_size):
    texts = [(ex.get(text_col) or "") for ex in batch]
    enc = tok(texts, **tok_kwargs)
    lens = [len(ids) for ids in enc["input_ids"]]
    total_tokens += sum(lens)
    count += len(lens)

avg_len = total_tokens / max(1, count)
print(f"{hf_dataset} average token length over first {sample_size} docs: {avg_len:.2f} tokens (tokenizer={tokenizer_name})")



# For GSN8K
# from datasets import load_dataset
# from transformers import AutoTokenizer

# tok = AutoTokenizer.from_pretrained("gpt2")   # Replace with your tokenizer
# ds = load_dataset("gsm8k", "main", split="train")

# lengths = [len(tok.encode(ex["question"] + ex["answer"])) for ex in ds]
# import numpy as np
# print("GSM8K token-length percentiles (combined question + answer):")
# for p in [50, 90, 95, 99, 100]:
#     print(f"  {p}th percentile: {np.percentile(lengths, p):.1f} tokens")


# # Result for gsm8k was:
# # GSM8K token-length percentiles (combined question + answer):
# #   50th percentile: 140.0 tokens
# #   90th percentile: 227.0 tokens
# #   95th percentile: 257.4 tokens
# #   99th percentile: 314.3 tokens
# #   100th percentile: 433.0 tokens