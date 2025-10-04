import torch

from sampledkd.distill.ce_estimators import ce_elimination_estimator


def _true_ce(s_rows: torch.Tensor, y_rows: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(s_rows, dim=-1)
    return -log_probs[torch.arange(s_rows.size(0)), y_rows]


def test_ce_elimination_full_vocab_matches_true_ce():
    torch.manual_seed(0)
    P, V = 3, 12

    s_rows = torch.randn(P, V)
    y_rows = torch.randint(0, V, (P,))

    ids_M = torch.empty((P, V - 1), dtype=torch.long)
    for i in range(P):
        y = int(y_rows[i].item())
        ids = torch.cat([torch.arange(y), torch.arange(y + 1, V)])
        ids_M[i] = ids

    ce_hat = ce_elimination_estimator(s_rows, ids_M, y_rows)
    ce_true = _true_ce(s_rows, y_rows)
    assert torch.allclose(ce_hat, ce_true, atol=1e-6, rtol=1e-6)


def test_ce_elimination_is_lower_bound():
    torch.manual_seed(1)
    P, V = 32, 2048

    s_rows = torch.randn(P, V)
    y_rows = torch.randint(0, V, (P,))
    ids_M_shared = torch.randperm(V, dtype=torch.long)[:256]

    ce_hat = ce_elimination_estimator(s_rows, ids_M_shared, y_rows)
    ce_true = _true_ce(s_rows, y_rows)

    assert torch.all(ce_hat <= ce_true + 1e-6)


def test_ce_elimination_detects_temperature_mismatch():
    torch.manual_seed(2)
    P, V = 16, 512
    temperature = 2.0

    s_rows = torch.randn(P, V)
    y_rows = torch.randint(0, V, (P,))
    ids_M_shared = torch.randperm(V, dtype=torch.long)[:128]

    ce_true = _true_ce(s_rows, y_rows)
    ce_hat_correct = ce_elimination_estimator(s_rows, ids_M_shared, y_rows)
    ce_hat_bad = ce_elimination_estimator(s_rows / temperature, ids_M_shared, y_rows)

    assert torch.all(ce_hat_correct <= ce_true + 1e-6)
    assert not torch.allclose(ce_hat_bad, ce_true, atol=0.05, rtol=0.05)


def test_ce_elimination_supports_backward():
    torch.manual_seed(3)
    P, V = 6, 256

    s_rows = torch.randn(P, V, requires_grad=True)
    y_rows = torch.randint(0, V, (P,))
    ids_M_shared = torch.randperm(V, dtype=torch.long)[:64]

    ce_hat = ce_elimination_estimator(s_rows, ids_M_shared, y_rows)
    loss = ce_hat.mean()
    loss.backward()

    assert s_rows.grad is not None
    assert torch.isfinite(s_rows.grad).all()
