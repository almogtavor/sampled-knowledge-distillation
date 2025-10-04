from __future__ import annotations

import torch


def ce_elimination_estimator(
    s_rows: torch.Tensor,
    ids_M_shared: torch.Tensor,
    y_rows: torch.Tensor,
) -> torch.Tensor:
    """Lower-bound cross-entropy over {y} ∪ M computed at temperature 1.

    This estimator is intentionally biased low because it only integrates over
    the shared negatives ``M`` plus the gold token ``y``. It is suitable for
    relative ranking or heuristic scoring, but it should not be interpreted as
    the true cross-entropy unless ``M`` spans the full vocabulary.

    Args:
        s_rows: Student logits for the selected positions, shape ``[P, V]`` at
            temperature 1 (no scaling by KD temperature).
        ids_M_shared: Shared negative token ids. Accepts shape ``[M]`` (same
            negatives for all rows) or ``[P, M]`` for per-row negatives.
        y_rows: Gold token ids for each position, shape ``[P]``.

    Returns:
        Tensor of shape ``[P]`` with the lower-bound cross-entropy estimates.
    """

    z_y = s_rows.gather(1, y_rows.view(-1, 1)).squeeze(1)                     # [P]

    if ids_M_shared.numel() == 0:
        z_M = s_rows.new_zeros((s_rows.size(0), 0))
    elif ids_M_shared.dim() == 1:
        z_M = s_rows[:, ids_M_shared]
    else:
        z_M = torch.gather(s_rows, 1, ids_M_shared)

    z_all = torch.cat([z_y.view(-1, 1), z_M], dim=1)
    logZ = torch.logsumexp(z_all, dim=1)
    return -(z_y - logZ)
