"""Per-subject LoRA with per-sample routing in a single forward/backward.

A stack of LoRA adapters (one per subject) lives in each wrapped Linear. The
adapter applied to each sample of a (subject-mixed) batch is selected by its
subject id, so one forward + one backward updates only the adapters of the
subjects present in the batch -- no per-subject loop.
"""

import math

import torch
import torch.nn as nn


class _SubjectCtx:
    """Process-wide holder for the per-sample subject ids of the batch currently
    flowing through the model. The train/eval loop sets it right before each
    forward so the multi-LoRA layers can route without touching the backbone's
    forward signature (it is remote-code, loaded with trust_remote_code)."""

    subject_ids = None


_CTX = _SubjectCtx()


def set_subject_ids(ids):
    _CTX.subject_ids = ids


class MultiSubjectLoraLinear(nn.Module):
    """Frozen nn.Linear + one LoRA adapter per subject (routed per sample).

    Optionally also carries a single global LoRA adapter, shared by every
    sample, applied on top of the per-subject one in the same forward. The
    layer delta is then  global_scaling * (x Ag) Bg  +  scaling * (x As) Bs ,
    with As/Bs selected per sample by subject id.
    """

    def __init__(self, base: nn.Linear, num_subjects: int, rank: int,
                 alpha: int = 32, dropout: float = 0.05,
                 global_rank: int = 0, global_alpha: int = 32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        self.global_rank = global_rank
        self.dropout = nn.Dropout(dropout)

        in_f, out_f = base.in_features, base.out_features
        a = torch.empty(num_subjects, rank, in_f)
        for s in range(num_subjects):
            nn.init.kaiming_uniform_(a[s], a=math.sqrt(5))
        self.lora_A = nn.Parameter(a)                                   # (S, r, in)
        self.lora_B = nn.Parameter(torch.zeros(num_subjects, out_f, rank))  # (S, out, r)

        if global_rank > 0:
            self.global_scaling = global_alpha / global_rank
            ga = torch.empty(global_rank, in_f)
            nn.init.kaiming_uniform_(ga, a=math.sqrt(5))
            self.global_A = nn.Parameter(ga)                            # (gr, in)
            self.global_B = nn.Parameter(torch.zeros(out_f, global_rank))  # (out, gr)

    def forward(self, x):
        out = self.base(x)
        sid = _CTX.subject_ids
        if sid is None and self.global_rank == 0:
            return out
        shape = x.shape
        xf = self.dropout(x).reshape(shape[0], -1, shape[-1])      # (B, N, in)
        delta = xf.new_zeros(xf.shape[0], xf.shape[1], out.shape[-1])
        if self.global_rank > 0:
            ga = torch.einsum("bni,ri->bnr", xf, self.global_A)    # (B, N, gr)
            delta = delta + self.global_scaling * torch.einsum(
                "bnr,or->bno", ga, self.global_B)
        if sid is not None:
            sid = sid.to(x.device)
            xa = torch.einsum("bni,bri->bnr", xf, self.lora_A[sid])  # (B, N, r)
            delta = delta + self.scaling * torch.einsum(
                "bnr,bor->bno", xa, self.lora_B[sid])
        return out + delta.reshape(*shape[:-1], out.shape[-1])


TARGET_SUFFIXES = ("to_qkv", "to_out", "net.1", "net.3")


def _is_target(name: str) -> bool:
    return any(name == s or name.endswith("." + s) for s in TARGET_SUFFIXES)


def inject_multi_subject_lora(model, num_subjects, rank, alpha=32, dropout=0.05,
                              global_rank=0, global_alpha=32):
    """Replace the targeted Linears with MultiSubjectLoraLinear, freeze the
    backbone, keep `final_layer` (the shared head) trainable. With global_rank>0
    each layer also gets a shared global LoRA adapter trained jointly.

    Returns (model, n_replaced)."""
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and _is_target(n)]
    for name, module in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1],
                MultiSubjectLoraLinear(module, num_subjects, rank, alpha, dropout,
                                       global_rank, global_alpha))

    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, MultiSubjectLoraLinear):
            for n, p in m.named_parameters():
                p.requires_grad = not n.startswith("base.")
    for p in model.final_layer.parameters():
        p.requires_grad = True

    return model, len(targets)
