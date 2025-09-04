import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

def load_model(model_name: str, device_map: str = "auto", quant_bits: int = None):
    """Load model with optional quantization to save GPU memory."""
    
    # Configure quantization if specified
    quantization_config = None
    if quant_bits in [4, 8]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=(quant_bits == 4),
            load_in_8bit=(quant_bits == 8),
            bnb_4bit_quant_type="nf4" if quant_bits == 4 else None,
            bnb_4bit_compute_dtype=torch.float16 if quant_bits == 4 else None,
            bnb_4bit_use_double_quant=True if quant_bits == 4 else None,
        )
    
    # Use torch.float16 for CPU to save memory, auto for GPU
    torch_dtype = torch.float16 if device_map == "cpu" else "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,  # Use less CPU memory during loading
        trust_remote_code=True,  # For some models like Qwen
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
