from __future__ import annotations

import torch
import torch.nn.functional as F


class KDCoreMixin:
    @staticmethod
    def _kl_loss(log_p: torch.Tensor, log_q: torch.Tensor):
        """KL(P||Q) where log_p are teacher log-probs, log_q are student log-probs."""
        log_q32 = log_q.float()
        log_p32 = log_p.float()
        kl = F.kl_div(log_q32, log_p32, log_target=True, reduction="none").sum(-1)
        return kl.to(log_q.dtype)

    @staticmethod
    def _proposal_sample_negatives(V: int, M: int, device: torch.device) -> torch.Tensor:
        """Draw M uniform 'negatives' from [0, V). Overlap with U/avoid is allowed."""
        if M <= 0 or V <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.randint(low=0, high=V, size=(M,), device=device, dtype=torch.long)

    @staticmethod
    def _student_log_probs_sampled(z_sel: torch.Tensor, q_sel: torch.Tensor, T: float) -> torch.Tensor:
        """Return log softmax over the sampled set S with importance correction: log s(i) ∝ z_i/T - log q(i)."""
        z_corr = z_sel / T - torch.log(q_sel.clamp_min(1e-12))
        logZ = torch.logsumexp(z_corr, dim=-1, keepdim=False)
        return z_corr - logZ
