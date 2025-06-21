import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.utils import is_torch_bf16_available

def load_model(model_name: str, device_map: str = "auto"):
    """Load model in 8‑bit to save GPU memory."""
    quant_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=None)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16 if is_torch_bf16_available() else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer 