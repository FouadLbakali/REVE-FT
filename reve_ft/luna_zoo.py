"""LUNA backbone (PulpBio/LUNA, large variant by default) adapted to this repo.

LUNA is topology-agnostic: it takes raw EEG plus 3D electrode coordinates and
projects every channel-token onto a small set of learned queries before running
a temporal RoPE transformer. Channel positions are looked up once from MNE's
standard_1005 montage given the dataset channel names, then kept as a buffer.

`LunaWrapper` exposes the same `(data, pos) -> logits` interface and a
`final_layer` attribute as the REVE / LaBraM paths, so the linear-probing
engine (feature caching swaps `final_layer` with Identity) works unchanged.
`pos` is accepted and ignored (REVE's per-trial positional bank); LUNA derives
its own positional encoding from `channel_locations`. The wrapper also applies
the trial-level channel-wise z-score that the official finetune pipeline uses
right before the forward."""

import warnings

import numpy as np
import torch
import torch.nn as nn

from luna_module.luna import LUNA, ClassificationHeadWithQueries

LUNA_REPO = "PulpBio/LUNA"

# Pretraining variant configs (config/model/LUNA_{base,large,huge}.yaml).
LUNA_VARIANTS = {
    "base": {"patch_size": 40, "embed_dim": 64,  "num_heads": 2, "depth": 8,
             "num_queries": 4, "drop_path": 0.1},
    "large": {"patch_size": 40, "embed_dim": 96, "num_heads": 2, "depth": 10,
              "num_queries": 6, "drop_path": 0.1},
    "huge": {"patch_size": 40, "embed_dim": 128, "num_heads": 2, "depth": 16,
             "num_queries": 8, "drop_path": 0.1},
}


def _build_channel_locations(ch_names):
    """3-D electrode positions in MNE's standard_1005 frame. Returns (C, 3)."""
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = mne.create_info(list(ch_names), 256.0, "eeg")
        info.set_montage("standard_1005", match_case=False, on_missing="warn")
        positions = info.get_montage().get_positions()["ch_pos"]
    locs = torch.tensor(np.array([positions[ch] for ch in ch_names]), dtype=torch.float32)
    if torch.isnan(locs).any():
        bad = [c for c, p in zip(ch_names, locs) if torch.isnan(p).any()]
        raise ValueError(f"No standard_1005 position for: {bad}")
    return locs                                                       # (C, 3)


def _load_pretrained(model, variant):
    """Load PulpBio/LUNA weights for the requested variant, dropping the
    reconstruction-only heads (decoder_head, channel_emb) and the randomly
    initialized classifier (so the new classifier head trains from scratch)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(LUNA_REPO, f"LUNA_{variant}.safetensors")
    sd = load_file(path)
    keep = {}
    for k, v in sd.items():
        # The HF checkpoints store raw model state (no "model." prefix), but
        # be defensive.
        k = k[len("model."):] if k.startswith("model.") else k
        if k.startswith("decoder_head") or k.startswith("channel_emb"):
            continue                              # reconstruction-only, unused
        if k.startswith("classifier"):
            continue                              # checkpoint has no classifier
        keep[k] = v
    missing, unexpected = model.load_state_dict(keep, strict=False)
    # Expected missing: `classifier.*` (fresh head). Anything else is a real
    # mismatch worth reporting.
    classifier_missing = [k for k in missing if k.startswith("classifier")]
    other_missing = [k for k in missing if not k.startswith("classifier")]
    if other_missing:
        print(f"[LUNA] WARN missing keys outside classifier: {other_missing}")
    if unexpected:
        print(f"[LUNA] WARN unexpected keys: {unexpected}")
    print(f"[LUNA] Loaded LUNA_{variant} (classifier reinit: {len(classifier_missing)} tensors)")


class _ChannelWiseNormalize(nn.Module):
    """Per-trial, per-channel z-score. Matches BioFoundation's finetune
    pipeline; applied inside the forward so the data loaders stay agnostic."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):                                      # (B, C, T)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        return (x - mean) / (std + self.eps)


class LunaWrapper(nn.Module):
    """PulpBio/LUNA behind the REVE-style training API.

    The classification head lives in `self.final_layer` (the engine swaps it
    with Identity for LP feature caching, yielding the pooled latent of shape
    (B, num_patches, embed_dim*num_queries)). The head flattens that latent,
    then RMSNorm + Dropout + Linear, matching REVE / LaBraM's flatten head."""

    def __init__(self, ch_names, n_times, n_classes, variant="large",
                 dropout=0.1):
        super().__init__()
        if variant not in LUNA_VARIANTS:
            raise ValueError(f"Unknown LUNA variant {variant!r}, expected "
                              f"one of {list(LUNA_VARIANTS)}")
        cfg = dict(LUNA_VARIANTS[variant])
        self.variant = variant
        self.patch_size = cfg["patch_size"]

        self.backbone = LUNA(num_classes=n_classes, **cfg)
        _load_pretrained(self.backbone, variant)

        # The engine swaps `final_layer` with Identity for LP feature caching;
        # the backbone's own `self.classifier` is set to Identity so the LUNA
        # forward returns the pooled latent untouched, which the head consumes.
        assert isinstance(self.backbone.classifier, ClassificationHeadWithQueries)
        self.backbone.classifier = nn.Identity()

        # Flatten head: flatten the (B, num_patches, embed_dim*num_queries)
        # latent, then RMSNorm + Dropout + Linear. The backbone's official
        # head is dropped above so this head trains from scratch.
        num_patches = n_times // self.patch_size
        dim = num_patches * self.backbone.embed_dim * self.backbone.num_queries
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.RMSNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, n_classes),
        )
        self.normalize = _ChannelWiseNormalize()

        # Pre-computed 3-D electrode positions for this dataset (constant, so
        # carried as a buffer and broadcast over the batch in forward).
        locs = _build_channel_locations(ch_names)                # (C, 3)
        self.register_buffer("channel_locations", locs, persistent=False)
        print(f"[LUNA] {len(ch_names)} channels, n_times={n_times}, "
              f"patches={n_times // self.patch_size}")

    def forward(self, data, pos=None):
        """`pos` is REVE's positional-bank tensor; LUNA derives positions from
        the registered channel_locations buffer, so the arg is accepted only
        for interface compatibility and ignored."""
        del pos
        x = self.normalize(data)
        B = x.shape[0]
        chan_loc = self.channel_locations.unsqueeze(0).expand(B, -1, -1)
        return self.final_layer(self.backbone(x, chan_loc))


class LunaSpec:
    """Deferred LUNA builder: the montage (ch_names) and trial length are only
    known once the data loaders exist, so `build_model` returns this and the
    linear-probing runner materialises the model."""

    def __init__(self, n_classes, variant="large"):
        self.n_classes = n_classes
        self.variant = variant

    def build(self, ch_names, n_times):
        return LunaWrapper(ch_names, n_times, self.n_classes,
                           variant=self.variant)
