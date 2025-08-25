from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")   # Replace with your tokenizer
ds = load_dataset("gsm8k", "main", split="train")

lengths = [len(tok.encode(ex["question"] + ex["answer"])) for ex in ds]
import numpy as np
print("GSM8K token-length percentiles (combined question + answer):")
for p in [50, 90, 95, 99, 100]:
    print(f"  {p}th percentile: {np.percentile(lengths, p):.1f} tokens")


# Result for gsm8k was:
# GSM8K token-length percentiles (combined question + answer):
#   50th percentile: 140.0 tokens
#   90th percentile: 227.0 tokens
#   95th percentile: 257.4 tokens
#   99th percentile: 314.3 tokens
#   100th percentile: 433.0 tokens