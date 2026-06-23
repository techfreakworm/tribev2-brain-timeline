import torch
import torch.nn as nn
from tribescore import fast_encode

class _Layer(nn.Module):
    def forward(self, x):
        return x

class _Encoder(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layer = nn.ModuleList([_Layer() for _ in range(n)])

class _HFModel(nn.Module):
    def __init__(self, n=40):
        super().__init__()
        self.encoder = _Encoder(n)

def test_registers_one_hook_per_encoder_layer():
    m = _HFModel(n=40)
    n = fast_encode._register_layer_empty_cache_hooks(m, every=1)
    assert n == 40
    assert all(len(l._forward_hooks) == 1 for l in m.encoder.layer)

def test_idempotent_no_double_registration():
    m = _HFModel(n=40)
    fast_encode._register_layer_empty_cache_hooks(m, every=1)
    n2 = fast_encode._register_layer_empty_cache_hooks(m, every=1)  # second call
    assert n2 == 0
    assert all(len(l._forward_hooks) == 1 for l in m.encoder.layer)  # not stacked

def test_no_encoder_layers_returns_zero():
    assert fast_encode._register_layer_empty_cache_hooks(nn.Linear(2, 2), every=1) == 0
