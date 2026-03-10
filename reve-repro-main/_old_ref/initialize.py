import functools
import logging
import math
from enum import Enum
from typing import Optional

import torch
from torch import nn


@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.warning(*args, **kwargs)


logging.Logger.warning_once = warning_once
logging.Logger.warn_once = warning_once


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


# from .configuration_bert import FlexBertConfig


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


class InitFnType(StrEnum):
    mitchell = "mitchell"
    """
    The strategy suggested to us by Mitchell Wortsman from UW.
    This uses a truncated normal distribution with an adaptive standard deviation that depends
    on the size of the weights as well as the depth of the layer.
    """

    normal = "normal"
    """
    All weights are initialized from the same normal distribution.
    """

    default = "default"
    """
    All weights are initialized with the default HuggingFace Bert method. Set init_std=0.02 to match.
    """

    kaiming_normal = "kaiming_normal"
    """
    All weights are initialized with the Kaiming method from a normal distribution.
    Note this currently won't work with FSDP.
    """

    fan_in = "fan_in"
    """
    "Fan-in variance scaling", i.e. normal with a standard deviation of ``1/sqrt(d_in)`` where ``d_in``
    is the input dimensionality of the kernel.
    """

    full_megatron = "full_megatron"
    """
    This is what metaseq calls "full megatron init". It is the init used for Llama 2.
    """


def init_weights(
    config: None,
    module: nn.Linear,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param layer_dim: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    if config.init_method == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(f"When using the {InitFnType.full_megatron} init, every module must have a type.")

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            # for att_proj (same as QKV), ff_proj
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            # for attn_out, ff_out
            std = config.init_std / math.sqrt(2.0 * config.num_hidden_layers)
        elif type_of_module == ModuleType.emb:
            # positional embeddings (wpe)
            # token embeddings (wte)
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            # final output (ff_out)
            std = config.hidden_size**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")

    if isinstance(module, torch.nn.Parameter):
        nn.init.trunc_normal_(
            module,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    else:
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_method == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.num_hidden_layers))
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_method == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.num_hidden_layers))


class ConfigInit:
    def __init__(
        self, init_method="full_megatron", init_std=0.02, init_cutoff_factor=2.0, hidden_size=512, num_hidden_layers=8
    ):
        self.init_method = init_method
        self.init_std = init_std
        self.init_cutoff_factor = init_cutoff_factor
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


def init_modules(model, config_megatron):
    init_weights(config_megatron, model.mask_token, type_of_module=ModuleType.in_module)
    if model.token_avg:
        init_weights(config_megatron, model.cls_query_token, type_of_module=ModuleType.in_module)
    for name, param in model.named_modules():
        if isinstance(param, nn.Linear):
            # if 'to_patch_embedding' in name:
            #     init_weights(config_megatron,param,type_of_module=ModuleType.emb)
            # elif 'mlp4d' in name:
            #      init_weights(config_megatron,param,type_of_module=ModuleType.emb)
            if "encoder.transformer.layers" in name:
                init_weights(config_megatron, param, type_of_module=ModuleType.in_module)
            elif "decoder.layers" in name:
                init_weights(config_megatron, param, type_of_module=ModuleType.in_module)
            # elif 'to_pixels'  in name:
            #      init_weights(config_megatron,param,type_of_module=ModuleType.final_out)


def init_ft(model, config_megatron,init_cls):
    if init_cls:
        init_weights(config_megatron, model.cls_query_token, type_of_module=ModuleType.out_module)
    for name, param in model.named_modules():
        if isinstance(param, nn.Linear):
            if "linear_head" in name:
                init_weights(config_megatron, param, type_of_module=ModuleType.final_out)


# import torch
# import torch.nn as nn
# from models.flash_vit_utils import *
# from models.patch_embed import FourierEmb4D#,patch_embedding,expand_with_time_dimension,mlp_pos_embedding

# def initialize_model_weights(model):
#     """
#     Apply weight initialization to all layers of a model, including custom layers
#     used in the MAE model and its encoder.
#     Use 0-init for readout layers (final linear layers).
#     """
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             if "to_pixels" in name or "linear_head" in name or "fc_pos" in name:
#                 # For readout layers (e.g., final linear layers in MAE and encoder)
#                 torch.nn.init.constant_(module.weight, 0.0)
#                 if module.bias is not None:
#                     torch.nn.init.constant_(module.bias, 0.0)
#             else:
#                 # Default Xavier initialization for other linear layers
#                 torch.nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     torch.nn.init.constant_(module.bias, 0.0)
#         elif isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
#             # Initialize LayerNorm and RMSNorm layers
#             torch.nn.init.constant_(module.weight, 1.0)
#             if hasattr(module, 'bias') and module.bias is not None:
#                 torch.nn.init.constant_(module.bias, 0.0)
#         elif isinstance(module, nn.Embedding):
#             # Initialize embedding layers (if used)
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#         elif isinstance(module, Attention):
#             # Initialize Attention components
#             if hasattr(module, "to_qkv"):
#                 torch.nn.init.xavier_uniform_(module.to_qkv.weight)
#             if hasattr(module, "to_out"):
#                 torch.nn.init.xavier_uniform_(module.to_out.weight)
#         elif isinstance(module, FeedForward):
#             # Initialize FeedForward layers
#             for submodule in module.net:
#                 if isinstance(submodule, nn.Linear):
#                     torch.nn.init.xavier_uniform_(submodule.weight)
#                     if submodule.bias is not None:
#                         torch.nn.init.constant_(submodule.bias, 0.0)
#         elif isinstance(module, FourierEmb4D):
#             # FourierEmb4D Initialization (if required)
#             if hasattr(module, 'x_param'):
#                 torch.nn.init.normal_(module.x_param, mean=0.0, std=0.02)
#             if hasattr(module, 'y_param'):
#                 torch.nn.init.normal_(module.y_param, mean=0.0, std=0.02)
#             if hasattr(module, 'z_param'):
#                 torch.nn.init.normal_(module.z_param, mean=0.0, std=0.02)
#             if hasattr(module, 'w_param'):
#                 torch.nn.init.normal_(module.w_param, mean=0.0, std=0.02)
#         elif isinstance(module, nn.Sequential):
#             # Sequential layers (for positional embeddings, etc.)
#             for submodule in module:
#                 if isinstance(submodule, nn.Linear):
#                     torch.nn.init.xavier_uniform_(submodule.weight)
#                     if submodule.bias is not None:
#                         torch.nn.init.constant_(submodule.bias, 0.0)


# def initialize_model_weights(model,init_patch,init_mlp,init_out,init_transformer,std=0.01):
#     """
#     Apply weight initialization to all layers of the MAE model and encoder.
#     Implements strategies inspired by the given example.

#     Args:
#         model (nn.Module): The model to initialize.
#         config (object): Configuration containing std, scaling factors, etc.
#     """

#     for name, param in model.named_parameters():
#         #if "to_pixels" in name or "fc_pos" in name:
#         #     # Zero initialization for positional and final readout layers
#             # param.data.zero_()
#             # if "bias" in name:
#             #     torch.nn.init.constant_(param, 0.01)
#             #torch.nn.init.normal_(param.data, mean=0.0, std=1e-3)  # best

#         if init_patch and "to_patch_embedding" in name:
#             # Initialize patch embedding layer
#             nn.init.normal_(param, std=std)
#         elif init_mlp and "mlp4d" in name:
#             nn.init.normal_(param, std=std)

#         elif init_transformer and isinstance(param, Attention):
#             # Initialize Attention components
#             if hasattr(param, "to_qkv"):
#                 torch.nn.init.xavier_uniform_(param.to_qkv.weight)
#             if hasattr(param, "to_out"):
#                 torch.nn.init.xavier_uniform_(param.to_out.weight)
#         elif init_out and isinstance(param, FeedForward):
#             # Initialize FeedForward layers
#             for submodule in param.net:
#                 if isinstance(submodule, nn.Linear):
#                     torch.nn.init.xavier_uniform_(submodule.weight)
#                     if submodule.bias is not None:
#                         torch.nn.init.constant_(submodule.bias, 0.0)

# nn.init.xavier_uniform_(param)
#     if any(skip in name for skip in skip_list):
#         # Specialized initialization for specific layers
#         if "to_patch_embedding" in name:
#             # Initialize patch embedding layer
#             param.data.normal_(mean=0.0, std=config.emb_std)
#         elif "fc_pos" in name or "to_pixels" in name:
#             # Zero initialization for positional and final readout layers
#             param.data.zero_()
#     else:
#         # General initialization
#         if len(param.shape) > 1:  # Weights of Linear layers
#             fan_in = param.shape[1]
#             if "to_qkv" in name or "to_out" in name:
#                 # Scaled initialization for attention projection layers
#                 param.data.normal_(
#                     mean=0.0,
#                     std=config.base_std * (config.n_layer * config.n_embd * 2) ** -0.5
#                 )
#             else:
#                 # Default Xavier-like initialization
#                 param.data.normal_(mean=0.0, std=config.base_std * fan_in ** -0.5)
#         else:  # Bias terms
#             param.data.zero_()

# for name, module in model.named_modules():
#     # LayerNorm and RMSNorm Initialization
#     if isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
#         torch.nn.init.constant_(module.weight, 1.0)
#         if hasattr(module, "bias") and module.bias is not None:
#             torch.nn.init.constant_(module.bias, 0.0)

#     # Fourier embedding initialization
#     if isinstance(module, FourierEmb4D):
#         for attr in ["x_param", "y_param", "z_param", "w_param"]:
#             if hasattr(module, attr):
#                 param = getattr(module, attr)
#                 param.data.normal_(mean=0.0, std=config.emb_std)


# class Config:
#     def __init__(self, emb_std=0.01, base_std=0.01, n_layer=6, n_embd=256):
#         """
#         Configuration for model initialization.

#         Args:
#             emb_std (float): Standard deviation for embedding layers.
#             base_std (float): Base standard deviation for general layers.
#             n_layer (int): Number of transformer layers in the model.
#             n_embd (int): Embedding dimension (e.g., hidden size).
#         """
#         self.emb_std = emb_std
#         self.base_std = base_std
#         self.n_layer = n_layer
#         self.n_embd = n_embd
