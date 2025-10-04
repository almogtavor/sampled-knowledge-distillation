from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import islice
import numpy as np

# --- config ---
data_files = {
    "train": [
        "hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10/000_0000*.parquet",
    ]
}
tokenizer_name = "gpt2"
target_n = 20_000
batch_size = 256

tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
tok_kwargs = dict(add_special_tokens=False)

# Stream directly from Parquet (no repo-wide listing)
ds = load_dataset("parquet", data_files=data_files, split="train", streaming=True)

def batched(iterator, size):
    buf = []
    for ex in iterator:
        buf.append(ex)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf

count = 0
lengths = []

for batch in batched(islice(ds, target_n), batch_size):
    texts = [ex["text"] for ex in batch if "text" in ex]
    enc = tok(texts, **tok_kwargs)
    lens = [len(ids) for ids in enc["input_ids"]]
    lengths.extend(lens)
    count += len(lens)

lengths = np.array(lengths)
mask = lengths <= 256
short_lengths = lengths[mask]

print(f"Docs processed: {count}")
print(f"Docs ≤256 tokens: {mask.sum()} ({mask.mean()*100:.2f}% of docs)")

print("\nPercentiles for docs ≤256 tokens:")
for p in [50, 90, 95, 99, 100]:
    if len(short_lengths) > 0:
        val = np.percentile(short_lengths, p)
        print(f"  {p}th percentile: {val:.1f} tokens")
    else:
        print(f"  {p}th percentile: N/A (no docs under 256)")


# For fineweb filter 256:
# Percentiles for docs ≤256 tokens:
#   50th percentile: 152.0 tokens
#   90th percentile: 232.0 tokens
#   95th percentile: 244.0 tokens
#   99th percentile: 254.0 tokens
#   100th percentile: 256.0 tokens
# Docs processed: 20000
# Average token length (first 20000 docs): 740.79 tokens

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
 
 # Now for HellaSwag
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

tok = AutoTokenizer.from_pretrained("gpt2")
ds = load_dataset("hellaswag", "default", split="train")

lengths = []
for ex in ds:
    gold_ending = ex["endings"][int(ex["label"])]
    text = ex["ctx"] + " " + gold_ending
    lengths.append(len(tok.encode(text)))

print("HellaSwag token-length percentiles (ctx + gold ending):")
for p in [50, 90, 95, 99, 100]:
    print(f"  {p}th percentile: {np.percentile(lengths, p):.1f} tokens")


# Result for HellaSwag was:
# HellaSwag token-length percentiles (ctx + gold ending):
#   50th percentile: 91.0 tokens
#   90th percentile: 121.0 tokens
#   95th percentile: 126.0 tokens
#   99th percentile: 133.0 tokens
#   100th percentile: 155.0 tokens