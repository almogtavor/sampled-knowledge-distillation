"""Minimal utilities for benchmark tasks."""

import re
from typing import Dict, Any


def process_results(doc: Dict[str, Any], results: list) -> Dict[str, int]:
    """Process results for numeric exact match with LaTeX normalization."""
    response = results[0] if results else ""
    
    # Extract from $...$ if present
    dollar_matches = re.findall(r'\$([^$]+)\$', response)
    if dollar_matches:
        response = dollar_matches[-1]
    
    # Extract from \boxed{...} if present
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        response = boxed_match.group(1)
    
    # Clean and normalize
    response = response.strip().replace(' ', '').replace(',', '')
    
    # Get target answer
    target = str(doc.get('answer', ''))
    target = target.strip().replace(' ', '').replace(',', '')
    
    return {"exact_match": 1 if response == target else 0}


def process_docs(dataset):
    """Normalize dataset docs for SVAMP."""
    def _process_row(row):
        question = row.get('question', '') or row.get('Body', '')
        answer = row.get('answer', '') or row.get('Answer', '')
        
        return {
            "input": f"Question: {question}\nAnswer:",
            "answer": str(answer)
        }
    
    # Map first available split to train/test
    split_name = next(iter(dataset.keys()))
    processed = dataset[split_name].map(_process_row)
    return {"train": processed, "test": processed}