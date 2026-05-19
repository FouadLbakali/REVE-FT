"""LabraM backbone (braindecode/labram-pretrained) adapted to this repo.

LaBraM uses a fixed 128-channel canonical montage, so arbitrary EEG montages
go through braindecode's `InterpolatedLaBraM` wrapper (channel interpolation to
the canonical set). The HF checkpoint was pretrained with 15 temporal patches,
so its `temporal_embedding` is sliced to the current patch count when loading.

`LabramWrapper` exposes the same `(data, pos) -> logits` interface and a
`final_layer` attribute as the REVE path, so the linear-probing engine
(feature caching swaps `final_layer` with Identity) works unchanged. `pos` is
accepted and ignored: LaBraM derives channel positions from `chs_info` at
construction, not per batch.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

LABRAM_REPO = "braindecode/labram-pretrained"
PATCH_SIZE = 200          # LaBraM temporal patch = 1 s @ 200 Hz
LABRAM_SFREQ = 200.0
LABRAM_EMBED_DIM = 200

# Attention (qkv / out proj) + MLP (fc1 / fc2) inside every transformer block.
LORA_TARGET_MODULES = ["qkv", "proj", "mlp.0", "mlp.2"]


def _build_chs_info(ch_names):
    """MNE channel-info list with 10-20 montage positions, for the channel
    interpolation layer."""
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = mne.create_info(list(ch_names), LABRAM_SFREQ, "eeg")
        info.set_montage("standard_1020", match_case=False, on_missing="warn")
    return info["chs"]


def _patched_attention_forward(self, x, return_attention=False, return_qkv=False):
    """braindecode `_Attention.forward` with two changes:

    1. The qkv projection goes through `self.qkv(x)` instead of
       `F.linear(x, self.qkv.weight, ...)`. The original reads `.weight`
       directly, which bypasses any adapter wrapped around `self.qkv` (PEFT
       LoRA or our per-subject multi-LoRA); routing through the module makes
       the qkv adapter actually contribute.
    2. The default path uses `F.scaled_dot_product_attention` instead of an
       explicit `softmax(q @ k.T) @ v`. The explicit form materialises the
       full `(B, heads, N, N)` attention matrix (plus softmax/dropout copies)
       and keeps it for the backward pass; with LaBraM's 128-channel canonical
       montage `N` is large and this is the dominant memory cost (OOM). SDPA
       uses a memory-efficient kernel that never materialises the N×N matrix.
       Numerically equivalent (same scale, same additive relative-position
       bias, same attention dropout).

    The explicit path is retained for `return_attention` / `return_qkv`, which
    need the attention weights and are only used for offline analysis (not
    training)."""
    B, N, _ = x.shape
    qkv_bias = None
    if self.q_bias is not None:
        qkv_bias = torch.cat(
            (self.q_bias,
             torch.zeros_like(self.v_bias, requires_grad=False),
             self.v_bias)
        )
    qkv = self.qkv(x)
    if qkv_bias is not None:
        qkv = qkv + qkv_bias
    qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if self.q_norm is not None:
        q = self.q_norm(q).type_as(v)
    if self.k_norm is not None:
        k = self.k_norm(k).type_as(v)

    attn_bias = None
    if self.relative_position_bias_table is not None:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1,
            -1,
        )
        attn_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

    if not (return_attention or return_qkv):
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    if return_attention:
        return attn
    x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    if return_qkv:
        return x, qkv
    return x


def _route_qkv_through_module(backbone):
    """Bind the patched forward to every `_Attention` instance so the qkv
    projection is computed by calling the module (LoRA-effective). Returns the
    number of attention blocks patched."""
    import types

    from braindecode.models.labram import _Attention

    n = 0
    for m in backbone.modules():
        if isinstance(m, _Attention):
            m.forward = types.MethodType(_patched_attention_forward, m)
            n += 1
    return n


def _load_pretrained(model):
    """Load the pretrained weights, slicing any parameter whose checkpoint
    shape exceeds the current one along the leading dims (the pretrained
    `temporal_embedding` has 15+1 patches; downstream trials have fewer)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sd = load_file(hf_hub_download(LABRAM_REPO, "model.safetensors"))
    own = model.state_dict()
    fixed = {}
    for k, v in sd.items():
        if k in own and own[k].shape != v.shape:
            o = own[k]
            if v.dim() == o.dim() and all(a >= b for a, b in zip(v.shape, o.shape)):
                fixed[k] = v[tuple(slice(0, s) for s in o.shape)].clone()
            # otherwise shape-incompatible -> leave randomly initialized
        else:
            fixed[k] = v
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    # Expected: `final_layer.*` (fresh head, replaced below) and the channel
    # interpolation buffers (not part of the checkpoint).
    return missing, unexpected


class LabramWrapper(nn.Module):
    """braindecode InterpolatedLaBraM behind the REVE-style training API."""

    def __init__(self, ch_names, n_times, n_classes, dropout=0.1):
        super().__init__()
        from braindecode.models import InterpolatedLaBraM

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.backbone = InterpolatedLaBraM(
                chs_info=_build_chs_info(ch_names),
                n_outputs=n_classes,
                n_times=n_times,
                sfreq=LABRAM_SFREQ,
            )
        _load_pretrained(self.backbone)
        n_attn = _route_qkv_through_module(self.backbone)
        print(f"Routed qkv through the module in {n_attn} attention blocks "
              f"(qkv LoRA-effective)")

        # The wrapper owns the trainable head; the backbone returns pooled
        # (B, embed_dim) features once its own head is the identity.
        embed_dim = self.backbone.embed_dim
        self.backbone.final_layer = nn.Identity()
        self.final_layer = nn.Sequential(
            nn.RMSNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, data, pos=None):
        feats = self.backbone(data)          # (B, embed_dim) pooled tokens
        return self.final_layer(feats)


class LabramSpec:
    """Deferred LabraM builder: the montage (chs_info) and trial length are
    only known once the data loaders exist, so `build_model` returns this and
    the linear-probing runner materialises the model."""

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def build(self, ch_names, n_times):
        return LabramWrapper(ch_names, n_times, self.n_classes)
