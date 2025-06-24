import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.utils import is_torch_bf16_available

def load_model(model_name: str, device_map: str = "auto", quant_bits: int = 4):
    """Load model in 8‑bit to save GPU memory."""
    if quant_bits == 4:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if is_torch_bf16_available() else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
    elif quant_bits == 8:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError("quant_bits must be 4 or 8")

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
