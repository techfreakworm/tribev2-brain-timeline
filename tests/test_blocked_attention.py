import torch
import torch.nn as nn
from tribescore import fast_encode


def _full_eager(q, k, v, scaling):
    w = torch.matmul(q, k.transpose(-1, -2)) * scaling
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(w, v).transpose(1, 2).contiguous()
    return out


def test_blocked_attention_matches_full_eager():
    torch.manual_seed(0)
    B, H, N, d = 1, 4, 1100, 32   # N not a multiple of block (512) -> exercises the tail
    q = torch.randn(B, H, N, d); k = torch.randn(B, H, N, d); v = torch.randn(B, H, N, d)
    scaling = d ** -0.5
    ref = _full_eager(q, k, v, scaling)
    mod = nn.Module(); mod.training = False
    out, weights = fast_encode._blocked_eager_attention_forward(
        mod, q, k, v, None, scaling, dropout=0.0)
    assert weights is None
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4), \
        f"max|Δ|={(out-ref).abs().max().item():.2e}"
