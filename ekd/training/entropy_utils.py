import torch


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy (the regular Entropy formula) in natural logarithm (base e).
    Used find high-entropy "fork" tokens / low-entropy certain tokens.

    logits: [seq_len, vocab]
    returns: [seq_len]"""
    probs = torch.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-9)).sum(-1)  # [L, V] -> [L] where L is sequence length, V is vocabulary size


# truncated entropy lower bound (Top-k + Tail), base-e (natural log)
def truncated_entropy_topk_tail(logits: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    H_hat = - sum_{j=1..k} p_j ln p_j - p_tail ln p_tail
    logits: [L, V]
    returns: [L]  (natural log base; multiply by log(2) if you want bits later)
    """
    # stable top-k on logits then convert to probabilities
    topk_vals, _ = torch.topk(logits, k=k, dim=-1)  # [L, k]
    # compute logsumexp for normalization using the same k window via trick:
    # We need full Z = logsumexp over V; approximate Z via topk + tail mass estimate:
    # Here we'll get probabilities by softmax over ALL via logits (safe enough with autocast disabled offline).
    # This function is only used offline/cached, so do the exact softmax:
    p_full = torch.softmax(logits, dim=-1)  # [L, V]
    p_topk, _ = torch.topk(p_full, k=k, dim=-1)  # [L, k]
    p_sum_topk = p_topk.sum(-1)  # [L]
    p_tail = (1.0 - p_sum_topk).clamp_min(0.0)  # [L]
    # -sum p log p on topk
    h_topk = -(p_topk * (p_topk.clamp_min(1e-12)).log()).sum(-1)  # [L]
    # tail bucket
    h_tail = -(p_tail * (p_tail.clamp_min(1e-12)).log())  # [L]
    return h_topk + h_tail  # [L]
