import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

PRETRAINED_MODEL_PATH = "../../kd_vanilla_run"

# 1) Load student & tokenizer (no .to())
student = AutoModelForCausalLM.from_pretrained(
    PRETRAINED_MODEL_PATH,
    device_map="auto",    # <-- let bitsandbytes + accelerate place on GPU
    load_in_8bit=True     # ensure 8-bit mode
)
tok = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, use_fast=True)
tok.pad_token = tok.eos_token

# 2) Prepare validation set
ds = load_dataset("Maxwell-Jia/AIME_2024", split="validation")

# 3) Compute perplexity
student.eval()
total_nll = 0.0
total_tokens = 0

for ex in tqdm(ds):
    text = ex["prompt"] + " " + ex["answer"]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc.input_ids.to(student.device)          # student.device is set by device_map
    attention = enc.attention_mask.to(student.device)

    with torch.no_grad():
        outputs = student(input_ids, attention_mask=attention, labels=input_ids)
        nll = outputs.loss.item() * input_ids.numel()
        total_nll += nll
        total_tokens += input_ids.numel()

perplexity = torch.exp(torch.tensor(total_nll / total_tokens))
print(f"Validation Perplexity: {perplexity:.2f}")
