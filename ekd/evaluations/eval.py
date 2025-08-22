import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm

PRETRAINED_MODEL_PATH = "../../kd_vanilla_run"

# Configure 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

#] Load student & tokenizer (no .to())
student = AutoModelForCausalLM.from_pretrained(
    PRETRAINED_MODEL_PATH,
    device_map="auto",
    quantization_config=bnb_config  # use BitsAndBytesConfig instead of load_in_8bit
)
tok = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, use_fast=True)
tok.pad_token = tok.eos_token

print("Testing model with simple input...")
test_text = "Hello world"
test_enc = tok(test_text, return_tensors="pt")
test_input_ids = test_enc.input_ids.to(student.device)

with torch.no_grad():
    test_outputs = student(test_input_ids, labels=test_input_ids)
    print(f"Test loss: {test_outputs.loss}")

    if torch.isnan(test_outputs.loss):
        print("ERROR: Model produces NaN even on simple input!")
        print("This suggests the model weights are corrupted.")
        exit(1)

# 2) Prepare evaluation set
ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")

# Let's examine the first example
first_example = ds[0]
print(f"First example problem: {first_example['Problem'][:100]}...")
print(f"First example answer: {first_example['Answer']}")

# 3) Compute perplexity with better error handling
student.eval()
total_nll = 0.0
total_tokens = 0
valid_examples = 0

for i, ex in enumerate(tqdm(ds)):
    try:
        # Combine problem and answer
        text = str(ex["Problem"]) + " " + str(ex["Answer"])

        # Check text length
        if len(text.strip()) == 0:
            print(f"Warning: Empty text at example {i}")
            continue

        # Tokenize with padding and attention mask
        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # Reduce max length to avoid memory issues
            padding=True
        )

        input_ids = enc.input_ids.to(student.device)
        attention_mask = enc.attention_mask.to(student.device)

        # Check for valid tokens
        if input_ids.numel() == 0:
            print(f"Warning: No tokens at example {i}")
            continue

        with torch.no_grad():
            outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss

            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at example {i}: {loss}")
                print(f"Text preview: {text[:100]}...")

                # Try without labels to see if that helps
                outputs_no_labels = student(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs_no_labels.logits
                print(f"Logits stats - min: {logits.min()}, max: {logits.max()}, mean: {logits.mean()}")
                continue

            # Count only non-padded tokens
            num_tokens = attention_mask.sum().item()

            if num_tokens == 0:
                print(f"Warning: Zero valid tokens at example {i}")
                continue

            nll = loss.item() * num_tokens
            total_nll += nll
            total_tokens += num_tokens
            valid_examples += 1

            # Debug first few examples
            if i < 3:
                print(f"Example {i}: loss={loss:.4f}, tokens={num_tokens}, text_len={len(text)}")

    except Exception as e:
        print(f"Error processing example {i}: {e}")
        continue

print(f"\nResults:")
print(f"Valid examples: {valid_examples}/{len(ds)}")
print(f"Total NLL: {total_nll}")
print(f"Total tokens: {total_tokens}")

if total_tokens > 0 and valid_examples > 0:
    avg_nll = total_nll / total_tokens
    perplexity = torch.exp(torch.tensor(avg_nll))
    print(f"Average NLL: {avg_nll:.4f}")
    print(f"Perplexity on train split: {perplexity:.2f}")
else:
    print("Error: No valid examples processed!")
    print("Possible issues:")
    print("1. Model weights are corrupted")
    print("2. Tokenizer/model mismatch")
    print("3. Data format issues")
    print("4. Quantization problems")