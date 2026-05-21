"""Vendored from BioFoundation/models/modules/frequency_embedder.py (Apache-2.0,
ETH Zurich)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from einops import rearrange
from timm.layers import Mlp


class FrequencyFeatureEmbedder(nn.Module):
    """Patch the input along the time axis, take the per-patch FFT magnitude
    and phase, and embed them via an MLP -> (B, C*S, embed_dim) tokens."""

    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        in_features = 2 * (patch_size // 2 + 1)
        self.frequency_to_embed = Mlp(
            in_features=in_features, hidden_features=int(4 * in_features),
            out_features=embed_dim,
        )

    def forward(self, x):
        B, C, T = x.size()
        S = T // self.patch_size
        if T % self.patch_size != 0:                       # pad to a whole patch
            pad_size = self.patch_size - (T % self.patch_size)
            x = F.pad(x, (0, pad_size))
            T = x.size(-1)
            S = T // self.patch_size
        x = x.view(B, C, S, self.patch_size)
        freq_representation = fft.rfft(x, dim=-1)
        magnitude = torch.abs(freq_representation)
        phase = torch.angle(freq_representation)
        freq_features = torch.cat((magnitude, phase), dim=-1)
        embedded = self.frequency_to_embed(freq_features)  # (B, C, S, embed_dim)
        embedded = rearrange(embedded, "B C t D -> B (C t) D")
        return embedded
