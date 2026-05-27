"""Vendored LUNA classification path from BioFoundation/models/LUNA.py
(Apache-2.0, ETH Zurich). The pre-training-only reconstruction decoder and the
SEED channel-name embedding table are dropped -- only the classification path
is exercised here, so num_classes>0 is required."""

import math

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import Mlp, trunc_normal_ as __call_trunc_normal_

from .freq_embedder import FrequencyFeatureEmbedder
from .rope_block import RotaryTransformerBlock


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def nerf_positional_encoding(coords: torch.Tensor, embed_size: int) -> torch.Tensor:
    """coords: (N, C, 3) -> (N, C, embed_size). Sinusoidal NeRF encoding."""
    N, C, dim = coords.shape
    device = coords.device
    freqs = embed_size // (2 * dim)
    leftover = embed_size - freqs * 2 * dim
    freq_bands = 2.0 ** torch.arange(freqs, device=device).float()
    scaled_coords = coords.unsqueeze(-1) * freq_bands.view(1, 1, 1, -1)
    sin_enc = torch.sin(scaled_coords)
    cos_enc = torch.cos(scaled_coords)
    encoded = torch.stack([sin_enc, cos_enc], dim=-1)\
        .permute(0, 1, 3, 2, 4).reshape(N, C, freqs * dim * 2)
    if leftover > 0:
        pad = torch.zeros(N, C, leftover, device=device, dtype=coords.dtype)
        encoded = torch.cat([encoded, pad], dim=-1)
    return encoded


class ClassificationHeadWithQueries(nn.Module):
    """Attention pooling + MLP. Receives (B, num_patches, embed_dim*Q)
    -> (B, num_classes)."""

    def __init__(self, input_dim=8, embed_dim=768, num_queries=8, num_heads=8,
                 num_classes=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = int(embed_dim * num_queries)
        self.decoder_attn = nn.MultiheadAttention(
            self.embed_dim, num_heads, batch_first=True, dropout=0.15
        )
        self.decoder_ffn = Mlp(
            in_features=self.embed_dim, hidden_features=int(self.embed_dim * 4),
            out_features=num_classes, act_layer=nn.GELU, drop=0.15,
        )
        self.learned_agg = nn.Parameter(
            torch.randn(1, 1, self.embed_dim), requires_grad=True
        )

    def forward(self, x):
        decoder_queries = self.learned_agg.repeat(x.shape[0], 1, 1)
        x = self.decoder_attn(query=decoder_queries, key=x, value=x)[0]
        x = x[:, 0, :]
        x = self.decoder_ffn(x)
        return x


class CrossAttentionBlock(nn.Module):
    """Channel-unification module: learned query tokens cross-attend over the
    per-patch channel tokens, then a self-attention refines the queries."""

    def __init__(self, num_queries, input_embed_dim, output_embed_dim, num_heads,
                 dropout_p=0.1, ff_dim=2048, pre_norm=True):
        super().__init__()
        self.num_queries = num_queries
        self.dropout_p = dropout_p
        self.query_embed = nn.Parameter(
            torch.randn(1, num_queries, input_embed_dim), requires_grad=True
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_embed_dim, num_heads=num_heads,
            dropout=dropout_p, batch_first=True,
        )
        self.temparature = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.ffn = Mlp(
            input_embed_dim, ff_dim, output_embed_dim,
            act_layer=nn.GELU, drop=dropout_p, norm_layer=nn.LayerNorm,
        )
        self.keys_norm = nn.LayerNorm(input_embed_dim)
        self.values_norm = nn.LayerNorm(input_embed_dim)
        self.queries_norm = nn.LayerNorm(input_embed_dim)
        self.query_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                input_embed_dim, nhead=num_heads, activation="gelu",
                dim_feedforward=ff_dim, batch_first=True, norm_first=True,
            ),
            num_layers=3,
        )

    def initialize_weights(self):
        nn.init.orthogonal_(self.query_embed, gain=1.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        batch_size, _, _ = x.size()
        queries = self.query_embed.repeat(batch_size, 1, 1)
        queries = self.queries_norm(queries)
        keys = self.keys_norm(x)
        values = self.values_norm(x)
        attention_out, attention_scores = self.cross_attention(
            query=queries, key=keys, value=values
        )
        attention_out = self.ffn(attention_out) + attention_out
        attention_out = self.query_self_attn(attention_out)
        return attention_out, attention_scores


class PatchEmbedNetwork(nn.Module):
    """Per-channel temporal patching with a small 1xK Conv stack."""

    def __init__(self, embed_dim=64, patch_size=40):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = 1
        self.out_channels = int(embed_dim // 4)
        self.groups = 4
        self.kernel_size = int(patch_size // 2)
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=(1, self.kernel_size - 1),
                      stride=(1, self.kernel_size // 2),
                      padding=(0, self.kernel_size // 2 - 1)),
            nn.GroupNorm(self.groups, self.out_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels,
                      kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(self.groups, self.out_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels,
                      kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(self.groups, self.out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = rearrange(x, "B C (S P) -> B (C S) P", P=self.patch_size)
        x = x.unsqueeze(1)
        x = self.proj_in(x)
        x = rearrange(x, "B E CS D -> B CS (D E)")
        return x


class LUNA(nn.Module):
    """LUNA classification path. Channel-unification cross-attention -> stack
    of temporal RoPE transformer blocks -> attention-pooled classifier."""

    def __init__(self, patch_size=40, num_queries=4, embed_dim=64, depth=8,
                 num_heads=2, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 drop_path=0.0, num_classes=2):
        super().__init__()
        assert num_classes > 0, "vendored LUNA only supports the classification path"
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.patch_size = patch_size
        self.patch_embed_size = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.depth = depth

        self.patch_embed = PatchEmbedNetwork(embed_dim=embed_dim, patch_size=patch_size)
        self.freq_embed = FrequencyFeatureEmbedder(embed_dim=embed_dim,
                                                   patch_size=patch_size)
        self.channel_location_embedder = nn.Sequential(
            Mlp(in_features=embed_dim, out_features=embed_dim,
                hidden_features=int(embed_dim * 2),
                act_layer=nn.GELU, drop=0.0, norm_layer=nn.LayerNorm),
        )
        # mask_token is in the pretraining checkpoint but unused at classification.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim),
                                       requires_grad=False)
        self.cross_attn = CrossAttentionBlock(
            num_queries=num_queries, input_embed_dim=embed_dim,
            output_embed_dim=embed_dim, num_heads=num_heads,
            ff_dim=int(mlp_ratio * embed_dim), pre_norm=True,
        )
        block_dim = int(embed_dim * num_queries)
        block_heads = int(num_heads * num_queries)
        self.blocks = nn.ModuleList([
            RotaryTransformerBlock(
                dim=block_dim, num_heads=block_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=drop_path,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(block_dim)
        self.classifier = ClassificationHeadWithQueries(
            input_dim=patch_size, num_queries=num_queries, embed_dim=embed_dim,
            num_classes=num_classes, num_heads=num_heads,
        )
        self.initialize_weights()

    def initialize_weights(self):
        self.cross_attn.initialize_weights()
        trunc_normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def prepare_tokens(self, x_signal, channel_locations):
        """Tokenize the signal. We always run with mask=None (no random
        token-replacement, no location noise), matching the published
        finetune setup minus the all-zero fake mask (which only adds
        location noise without masking anything)."""
        num_channels = channel_locations.shape[1]
        num_patches_per_channel = x_signal.shape[-1] // self.patch_size
        x_patched = self.patch_embed(x_signal)
        freq_embed = self.freq_embed(x_signal)
        x_tokenized = x_patched + freq_embed

        ch_min = torch.min(channel_locations, dim=1, keepdim=True)[0]
        ch_max = torch.max(channel_locations, dim=1, keepdim=True)[0]
        channel_locations = (channel_locations - ch_min) / (ch_max - ch_min + 1e-8)
        channel_locations = nerf_positional_encoding(
            channel_locations, self.patch_embed_size
        )
        channel_locations_emb = self.channel_location_embedder(channel_locations)

        x_tokenized = rearrange(x_tokenized, "B (C t) D -> (B t) C D",
                                C=num_channels)
        channel_locations_emb = channel_locations_emb.repeat(
            num_patches_per_channel, 1, 1
        )
        x_tokenized = x_tokenized + channel_locations_emb
        return x_tokenized

    def forward(self, x_signal, channel_locations):
        """x_signal: (B, C, T); channel_locations: (B, C, 3).
        Returns (B, num_patches, embed_dim*num_queries) when self.classifier is
        nn.Identity (LP feature caching) and (B, num_classes) otherwise."""
        B = x_signal.shape[0]
        x = self.prepare_tokens(x_signal, channel_locations)
        x, _ = self.cross_attn(x)                                    # (B*t, Q, D)
        x = rearrange(x, "(B t) Q D -> B t (Q D)", B=B)              # (B, t, Q*D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.classifier(x)
