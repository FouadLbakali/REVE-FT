import math
from enum import Enum

import torch
from torch import nn

from models.flash_vit_utils import Transformer
from models.mae_eeg import MAE
from models.transformer_eeg import TransformerEncoder


class TileMode(Enum):
    center_weights = "center_weights"
    tile_weights_from_edge = "tile_weights_from_edge"
    tile_weights_from_middle = "tile_weights_from_middle"


def tile_init(
    old: MAE,
    new: MAE,
    tiling_mode: TileMode = TileMode.tile_weights_from_edge,
):
    """
    Please check this function.
    If this goest through the PR, you're a nullosse.
    --------------------------------------------------

    List of trainable modules:
    - encoder: TransformerEncoder
        - Transformer
        - PatchEmbedding
        - MLP4D
    - decoder: Transformer
    - to_pixels: Linear
    - mask_token: nn.Parameter

    (opt)
    - enc_to_dec: Linear
    - pos_enc_to_dec: Linear
    - cls_to_pixels: Sequential (Linear)
    - cls_query_token: nn.Parameter

    """
    # Tile Encoder
    tile_transformer(old.encoder.transformer, new.encoder.transformer, tiling_mode)
    tile_linear(old.encoder.to_patch_embedding[0], new.encoder.to_patch_embedding[0], tiling_mode)
    tile_linear(old.encoder.mlp4d[0], new.encoder.mlp4d[0], tiling_mode)

    # Tile Decoder
    tile_transformer(old.decoder, new.decoder, tiling_mode)

    # Tile to_pixels
    tile_linear(old.to_pixels, new.to_pixels, tiling_mode)

    # Tile mask_token
    with torch.no_grad():
        placeholder_old = old.mask_token.data.clone()
        placeholder_new = new.mask_token.data.clone()

        placeholder_new = _tile_1d(placeholder_old, placeholder_new, tiling_mode)
        new.mask_token = nn.Parameter(
            placeholder_new,
            requires_grad=new.mask_token.requires_grad,
        )

    if isinstance(old.enc_to_dec, nn.Linear):  # else Identity
        tile_linear(old.enc_to_dec, new.enc_to_dec, tiling_mode)
        tile_linear(old.pos_enc_to_dec, new.pos_enc_to_dec, tiling_mode)

    if old.token_avg and new.token_avg:
        with torch.no_grad():
            # dim is (1, 1, self.encoder.embed_dim)
            placeholder_old = old.cls_query_token.data[0, 0].clone()
            placeholder_new = new.cls_query_token.data[0, 0].clone()

            placeholder_new = _tile_1d(placeholder_old, placeholder_new, tiling_mode)
            new.cls_query_token = nn.Parameter(
                placeholder_new.view(1, 1, -1),
                requires_grad=new.cls_query_token.requires_grad,
            )

        # tile cls_to_pixels
        tile_linear(old.cls_to_pixels[0], new.cls_to_pixels[0], tiling_mode)
        tile_linear(old.cls_to_pixels[2], new.cls_to_pixels[2], tiling_mode)


def tile_transformer(
    old: Transformer,
    new: Transformer,
    mode: TileMode = TileMode.center_weights,
):
    pretrained_layers = len(old.layers)
    new_layers = len(new.layers)
    layer_mapping = [math.floor(i * pretrained_layers / new_layers) for i in range(new_layers)]

    for new_idx, old_idx in enumerate(layer_mapping):
        # ModuleList: [Attention, FeedForward]

        # Attention (to_qkv, to_out)
        tile_linear(old.layers[old_idx][0].to_qkv, new.layers[new_idx][0].to_qkv, mode)
        tile_linear(old.layers[old_idx][0].to_out, new.layers[new_idx][0].to_out, mode)

        # FeedForward.net [RMSNorm, Linear, Act, Linear]
        tile_linear(old.layers[old_idx][1].net[1], new.layers[new_idx][1].net[1], mode)
        tile_linear(old.layers[old_idx][1].net[3], new.layers[new_idx][1].net[3], mode)


def tile_linear(
    old: nn.Linear,
    new: nn.Linear,
    mode: TileMode = TileMode.center_weights,
):
    """
    In-place tiling of weights and biases of a linear layer.
    """

    assert isinstance(old, nn.Linear), "Old module must be a Linear layer, got {}".format(type(old))
    assert isinstance(new, nn.Linear), "New module must be a Linear layer, got {}".format(type(new))
    assert (old.bias is None) == (new.bias is None), "Bias must be present in both or absent in both"

    with torch.no_grad():
        new.weight = nn.Parameter(_tile_2d(old.weight, new.weight, mode), requires_grad=new.weight.requires_grad)
        if old.bias is not None:
            new.bias = nn.Parameter(_tile_1d(old.bias, new.bias, mode), requires_grad=new.bias.requires_grad)


# from https://github.com/AnswerDotAI/ModernBERT/blob/main/src/bert_layers/initialization.py


def _tile_1d(pretrained_weights: torch.Tensor, new_weights: torch.Tensor, mode: TileMode) -> torch.Tensor:
    assert pretrained_weights.dim() == 1, "Input tensor must be 1-dimensional"
    input_size = pretrained_weights.shape[0]
    new_size = new_weights.shape[0]
    assert new_size >= input_size, "Desired size must be greater than or equal to input size"

    if mode == TileMode.center_weights:
        offset = (new_size - input_size) // 2
        new_weights[offset : offset + input_size] = pretrained_weights
        return new_weights.clone()
    elif mode == TileMode.tile_weights_from_edge:
        repeat_count = (new_size + input_size - 1) // input_size
        tiled_tensor = pretrained_weights.repeat(repeat_count)
        return tiled_tensor[:new_size].clone()
    elif mode == TileMode.tile_weights_from_middle:
        # Calculate offsets to center the original tensor
        offset = (new_size - input_size) // 2

        # Create a new tensor with the desired size
        result = torch.zeros(new_size, dtype=pretrained_weights.dtype, device=pretrained_weights.device)

        # Place the original tensor in the center
        result[offset : offset + input_size] = pretrained_weights

        # Tile the left and right sides
        for i in range(offset):
            result[offset - 1 - i] = pretrained_weights[input_size - 1 - (i % input_size)]
        for i in range(offset + input_size, new_size):
            result[i] = pretrained_weights[(i - offset) % input_size]
        return result.clone()


def _tile_2d(pretrained_weights: torch.Tensor, new_weights: torch.Tensor, mode: TileMode) -> torch.Tensor:
    assert pretrained_weights.dim() == 2, "Input tensor must be 2-dimensional"  # noqa: PLR2004
    input_height, input_width = pretrained_weights.shape
    new_height, new_width = new_weights.shape
    assert new_height >= input_height, "Desired height must be greater than or equal to input height"
    assert new_width >= input_width, "Desired width must be greater than or equal to input width"

    if mode == TileMode.center_weights:
        height_offset = (new_height - input_height) // 2
        width_offset = (new_width - input_width) // 2
        new_weights[height_offset : height_offset + input_height, width_offset : width_offset + input_width] = (
            pretrained_weights
        )
        return new_weights.clone()
    elif mode == TileMode.tile_weights_from_edge:
        repeat_height = (new_height + input_height - 1) // input_height
        repeat_width = (new_width + input_width - 1) // input_width
        tiled_tensor = pretrained_weights.repeat(repeat_height, repeat_width)
        return tiled_tensor[:new_height, :new_width].clone()
    elif mode == TileMode.tile_weights_from_middle:
        # Calculate offsets to center the original tensor
        height_offset = (new_height - input_height) // 2
        width_offset = (new_width - input_width) // 2

        # Create a new tensor with the desired width and input height
        horizontal_tiled = torch.zeros(
            input_height, new_width, dtype=pretrained_weights.dtype, device=pretrained_weights.device
        )

        # Place the original tensor in the center horizontally
        horizontal_tiled[:, width_offset : width_offset + input_width] = pretrained_weights

        # Tile the left and right sides
        for i in range(width_offset):
            horizontal_tiled[:, i] = horizontal_tiled[
                :, width_offset + input_width - 1 - (width_offset - i - 1) % input_width
            ]
        for i in range(width_offset + input_width, new_width):
            horizontal_tiled[:, i] = horizontal_tiled[:, width_offset + (i - width_offset) % input_width]

        # Now tile vertically
        result = torch.zeros(new_height, new_width, dtype=pretrained_weights.dtype, device=pretrained_weights.device)
        result[height_offset : height_offset + input_height, :] = horizontal_tiled

        # Tile top
        for i in range(height_offset):
            row_to_copy = (input_height - 1) - (i % input_height)
            result[height_offset - 1 - i, :] = horizontal_tiled[row_to_copy, :]

        # Tile bottom
        for i in range(height_offset + input_height, new_height):
            row_to_copy = (i - height_offset) % input_height
            result[i, :] = horizontal_tiled[row_to_copy, :]
        return result.clone()


# Tests


def tiling_test():
    args_small_enc = {
        "patch_size": 200,
        "overlap_size": 20,
        "noise_ratio": 0.0025,
        "embed_dim": 512,
        "depth": 4,
        "heads": 4,
        "mlp_dim_ratio": 2.66,
        "dim_head": 64,
        "use_flash": True,
        "geglu": True,
    }

    args_small_mae = {
        "masking_ratio": 0.55,
        "decoder_dim": 512,
        "decoder_depth": 4,
        "decoder_heads": 4,
        "use_flash": True,
        "geglu": True,
        "token_avg": True,
        "token_avg_lambda": 0.1,
    }

    small_model = TransformerEncoder(**args_small_enc)
    small_mae = MAE(encoder=small_model, **args_small_mae)

    args_large_enc = {
        "patch_size": 200,
        "overlap_size": 20,
        "noise_ratio": 0.0025,
        "embed_dim": 768,
        "depth": 12,
        "heads": 12,
        "mlp_dim_ratio": 2.66,
        "dim_head": 64,
        "use_flash": True,
        "geglu": True,
    }

    args_large_mae = {
        "masking_ratio": 0.55,
        "decoder_dim": 768,
        "decoder_depth": 12,
        "decoder_heads": 12,
        "use_flash": True,
        "geglu": True,
        "token_avg": True,
        "token_avg_lambda": 0.1,
    }

    large_model = TransformerEncoder(**args_large_enc)
    large_mae = MAE(encoder=large_model, **args_large_mae)

    tile_init(small_mae, large_mae, tiling_mode=TileMode.center_weights)


def viz_tiling():
    import matplotlib.pyplot as plt

    TILE_MODE = TileMode.tile_weights_from_edge

    small_linear = nn.Linear(10, 10)
    large_linear = nn.Linear(20, 20)

    # Initialize small_linear with some values
    small_linear.weight.data = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    small_linear.bias.data = torch.arange(10, dtype=torch.float32)

    plt.figure(figsize=(12, 4))
    # Plot small_linear weights
    plt.subplot(1, 3, 1)
    plt.title("Small Linear Weights")
    plt.imshow(small_linear.weight.data.numpy(), cmap="viridis")
    plt.colorbar()

    # Plot large_linear weights before tiling
    plt.subplot(1, 3, 2)
    plt.title("Large Linear Weights (Before)")
    plt.imshow(large_linear.weight.data.numpy(), cmap="viridis")
    plt.colorbar()

    # Tile weights from small_linear to large_linear
    tile_linear(small_linear, large_linear, TILE_MODE)

    # Plot large_linear weights after tiling
    plt.subplot(1, 3, 3)
    plt.title("Large Linear Weights (After)")
    plt.imshow(large_linear.weight.data.numpy(), cmap="viridis")
    plt.colorbar()

    plt.tight_layout()

    plt.savefig("tiling.png")


if __name__ == "__main__":
    tiling_test()
    viz_tiling()
