"""Ollama model loading and inference utilities."""

import json
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer
from ollama import Client

class OllamaModel:
    """Wrapper for Ollama models to match HuggingFace interface."""
    
    def __init__(self, model_name: str = "qwen3:8b"):
        self.client = Client()
        self.model_name = model_name
        # Use a compatible tokenizer for tokenization
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-8B")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate text using Ollama model."""
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "num_predict": max_tokens,
                "temperature": 0.0,  # Deterministic for distillation
            }
        )
        return response['response']

    def __call__(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Mimic HuggingFace model interface for distillation."""
        # Convert input_ids to text
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Generate responses
        responses = [self.generate(prompt) for prompt in prompts]
        
        # Tokenize responses
        response_tokens = self.tokenizer(
            responses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=input_ids.size(1)
        )
        
        # Create a dummy output object that matches HuggingFace interface
        class DummyOutput:
            def __init__(self, logits):
                self.logits = logits
        
        # Convert to logits (one-hot for the actual tokens)
        vocab_size = self.tokenizer.vocab_size
        batch_size, seq_len = response_tokens.input_ids.size()
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        
        # Set logits to high value for actual tokens
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = response_tokens.input_ids[b, s]
                if token_id != self.tokenizer.pad_token_id:
                    logits[b, s, token_id] = 100.0  # High logit for actual token
        
        return DummyOutput(logits) 