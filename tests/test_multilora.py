"""Correctness checks for the per-subject multi-LoRA routing.

Runs fully offline (no model / dataset download): exercises the routing math,
per-sample gradient isolation, and module injection on a tiny fake model.

Run:  conda run -n venv python tests/test_multilora.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from multilora import (
    MultiSubjectLoraLinear,
    inject_multi_subject_lora,
    set_subject_ids,
)


def _reference_delta(x, sid, lora_A, lora_B, scaling):
    """Manual per-sample LoRA delta (dropout disabled in eval)."""
    out = []
    for k in range(x.shape[0]):
        A = lora_A[sid[k]]                 # (r, in)
        B = lora_B[sid[k]]                 # (out, r)
        out.append(scaling * (x[k] @ A.T) @ B.T)
    return torch.stack(out)


def test_routing_matches_manual_reference():
    torch.manual_seed(0)
    base = nn.Linear(7, 5, bias=True)
    layer = MultiSubjectLoraLinear(base, num_subjects=4, rank=3, alpha=6,
                                   dropout=0.0)
    # Make adapters non-trivial (B is zero-init by design).
    with torch.no_grad():
        layer.lora_B.normal_()
    layer.eval()

    x = torch.randn(6, 9, 7)                       # (B, N, in)
    sid = torch.tensor([0, 3, 1, 1, 2, 0])
    set_subject_ids(sid)
    out = layer(x)
    set_subject_ids(None)

    expected = base(x) + _reference_delta(x, sid, layer.lora_A, layer.lora_B,
                                          layer.scaling)
    assert torch.allclose(out, expected, atol=1e-5), \
        (out - expected).abs().max().item()


def test_subjects_get_independent_outputs():
    torch.manual_seed(1)
    layer = MultiSubjectLoraLinear(nn.Linear(8, 8, bias=False),
                                   num_subjects=3, rank=4, alpha=4, dropout=0.0)
    with torch.no_grad():
        layer.lora_B.normal_()
    layer.eval()

    x = torch.randn(1, 8).repeat(3, 1).reshape(3, 1, 8)  # same input, 3 subjects
    set_subject_ids(torch.tensor([0, 1, 2]))
    out = layer(x)
    set_subject_ids(None)
    # Distinct adapters -> distinct rows for an identical input.
    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out[1], out[2])


def test_single_backward_only_touches_present_subjects():
    torch.manual_seed(2)
    layer = MultiSubjectLoraLinear(nn.Linear(6, 6, bias=False),
                                   num_subjects=5, rank=2, alpha=2, dropout=0.0)
    with torch.no_grad():
        layer.lora_B.normal_()                 # so A also receives gradient
    layer.train()

    x = torch.randn(4, 3, 6)
    present = torch.tensor([1, 3, 1, 3])       # subjects 0,2,4 absent
    set_subject_ids(present)
    layer(x).pow(2).mean().backward()
    set_subject_ids(None)

    for s in (0, 2, 4):
        assert torch.count_nonzero(layer.lora_B.grad[s]) == 0, f"absent {s} got grad"
        assert torch.count_nonzero(layer.lora_A.grad[s]) == 0, f"absent {s} got grad"
    for s in (1, 3):
        assert torch.count_nonzero(layer.lora_B.grad[s]) > 0, f"present {s} no grad"


def test_global_plus_subject_matches_manual_reference():
    torch.manual_seed(4)
    base = nn.Linear(7, 5, bias=True)
    layer = MultiSubjectLoraLinear(base, num_subjects=3, rank=2, alpha=4,
                                   dropout=0.0, global_rank=3, global_alpha=6)
    with torch.no_grad():
        layer.lora_B.normal_()
        layer.global_B.normal_()
    layer.eval()

    x = torch.randn(5, 4, 7)
    sid = torch.tensor([0, 2, 1, 0, 2])
    set_subject_ids(sid)
    out = layer(x)
    set_subject_ids(None)

    gdelta = layer.global_scaling * (x @ layer.global_A.T) @ layer.global_B.T
    sdelta = _reference_delta(x, sid, layer.lora_A, layer.lora_B, layer.scaling)
    expected = base(x) + gdelta + sdelta
    assert torch.allclose(out, expected, atol=1e-5), \
        (out - expected).abs().max().item()


def test_global_grads_all_samples_subject_still_isolated():
    torch.manual_seed(5)
    layer = MultiSubjectLoraLinear(nn.Linear(6, 6, bias=False), num_subjects=5,
                                   rank=2, alpha=2, dropout=0.0,
                                   global_rank=2, global_alpha=2)
    with torch.no_grad():
        layer.lora_B.normal_()
        layer.global_B.normal_()
    layer.train()

    x = torch.randn(4, 3, 6)
    present = torch.tensor([1, 3, 1, 3])           # subjects 0,2,4 absent
    set_subject_ids(present)
    layer(x).pow(2).mean().backward()
    set_subject_ids(None)

    # Global adapter is shared -> trained by the whole batch.
    assert torch.count_nonzero(layer.global_B.grad) > 0
    # Per-subject adapters stay isolated even with the global one active.
    for s in (0, 2, 4):
        assert torch.count_nonzero(layer.lora_B.grad[s]) == 0
    for s in (1, 3):
        assert torch.count_nonzero(layer.lora_B.grad[s]) > 0


class _FF(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d),
                                 nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        return self.net(x)


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.to_qkv = nn.Linear(d, d)
        self.to_out = nn.Linear(d, d)

    def forward(self, x):
        return self.to_out(self.to_qkv(x))


class _FakeReve(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList(
            [nn.ModuleList([_Attn(d), _FF(d)]) for _ in range(2)]
        )
        self.unrelated = nn.Linear(d, d)          # must stay frozen, not wrapped
        self.final_layer = nn.Sequential(nn.Linear(d, 2))

    def forward(self, x):
        for attn, ff in self.transformer.layers:
            x = ff(attn(x))
        return self.final_layer(x)


def test_injection_targets_and_freezing():
    model = _FakeReve(d=8)
    model, n = inject_multi_subject_lora(model, num_subjects=3, rank=2)
    # 2 layers * (to_qkv, to_out, net.1, net.3) = 8
    assert n == 8, n

    wrapped = [m for m in model.modules() if isinstance(m, MultiSubjectLoraLinear)]
    assert len(wrapped) == 8

    assert not any(p.requires_grad for p in model.unrelated.parameters())
    assert all(p.requires_grad for p in model.final_layer.parameters())
    for w in wrapped:
        assert not any(p.requires_grad for p in w.base.parameters())
        assert w.lora_A.requires_grad and w.lora_B.requires_grad


def test_injected_model_forward_routes():
    torch.manual_seed(3)
    model = _FakeReve(d=8)
    model, _ = inject_multi_subject_lora(model, num_subjects=4, rank=2, alpha=8)
    for w in model.modules():
        if isinstance(w, MultiSubjectLoraLinear):
            with torch.no_grad():
                w.lora_B.normal_()
    model.eval()

    x = torch.randn(1, 5, 8).repeat(2, 1, 1)
    set_subject_ids(torch.tensor([0, 1]))
    out = model(x)
    set_subject_ids(None)
    assert out.shape == (2, 5, 2)
    assert not torch.allclose(out[0], out[1])   # different subjects -> different


def main():
    tests = [
        test_routing_matches_manual_reference,
        test_subjects_get_independent_outputs,
        test_single_backward_only_touches_present_subjects,
        test_global_plus_subject_matches_manual_reference,
        test_global_grads_all_samples_subject_still_isolated,
        test_injection_targets_and_freezing,
        test_injected_model_forward_routes,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
