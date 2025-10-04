from __future__ import annotations

import torch


def kl_from_vocab_samples(
    t_logp_sel: torch.Tensor, # [S] teacher log-probs at sampled tokens
    s_logp_sel: torch.Tensor, # [S] student log-probs at same tokens
    q_sel: torch.Tensor,  # [S] proposal probabilities used to draw the samples
    self_norm: bool = True,
    T_kd: float = 1.0,
) -> torch.Tensor:
    """
    Estimate KL(P||S) = sum_j p_j (log p_j - log s_j)
    with samples j ~ q. Use weights w_j = p_j / q_j.

    If self_norm=True (default): return sum_j (w_j / sum w) * (log p_j - log s_j)   (low-variance, slightly biased)
    Else (unbiased Horvitz-Thompson): return mean_j w_j * (log p_j - log s_j) with expectation over q.

    Inputs are 1D tensors for a single position. Returns a scalar tensor.
    """
    with torch.no_grad():
        # teacher probabilities at T=1 (cache stores t_logp_sel at T=1)
        p1 = t_logp_sel.exp()
        # retemper to current KD temperature: p_T ∝ p_1^{1/T_kd}
        gamma = 1.0 / max(1e-12, T_kd)
        pT_un = p1.pow(gamma)
        w = pT_un / q_sel.clamp_min(1e-12)  # [S]
        if self_norm:
            w = w / w.sum().clamp_min(1e-12)
    # approximate log p_T on sampled ids up to a constant
    logpT_approx = torch.log(pT_un) - torch.log(pT_un.sum().clamp_min(1e-12))
    diff = (logpT_approx - s_logp_sel)  # [S]
    if self_norm:
        return (w * diff).sum()
    else:
        # Importance expectation E_q[w * diff] ≈ (1/S) * sum w * diff
        return (w * diff).mean()
