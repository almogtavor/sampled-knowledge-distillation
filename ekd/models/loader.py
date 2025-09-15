import torch
from typing import Union, Optional, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

def load_model(
    model_name: str,
    device_map: Union[str, int, Dict[str, int]] = "auto",
    quant_bits: int = None,
    max_memory: Optional[Dict[Union[int, str], str]] = None,
):
    """Load model with optional quantization to save GPU memory."""
    
    # Configure quantization if specified
    quantization_config = None
    if quant_bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant_bits == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Use torch.float16 for better memory efficiency
    torch_dtype = torch.float16

    # Optimize device mapping strategy
    device_map_arg = device_map
    if quantization_config is not None and device_map != "cpu":
        device_map_arg = "auto"
    elif isinstance(device_map, int):
        # If a specific GPU index was requested, map the whole model to that GPU
        device_map_arg = {"": device_map}
    elif device_map == "auto":
        # Use auto for optimal sharding across available GPUs
        device_map_arg = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map_arg,
        max_memory=max_memory,
        dtype=torch_dtype,
        quantization_config=quantization_config,
        low_cpu_mem_usage=False,  # Disable to avoid hanging issues
        trust_remote_code=True,  # For some models like Qwen
    )
    print("Model loaded successfully, now loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    print("Tokenizer loaded, setting pad token...")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer ready!")

    return model, tokenizer
