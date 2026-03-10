import math
from builtins import print as bprint

import idr_torch
import torch

from models.classifier import ReveClassifier
from models.encoder import REVE


def print(*args, **kwargs):
    if idr_torch.is_master or kwargs.pop("force", False):
        bprint(*args, **kwargs)


def get_flattened_output_dim(config, n_timepoints: int, n_chans: int) -> int:
    """Helper function to compute the flattened output dimension after the transformer."""
    pooling = config.task.classifier.pooling
    embed_dim = config.encoder.transformer.embed_dim

    if pooling in ["last", "all"]:
        return embed_dim

    patch_size = config.encoder.patch_size
    overlap_size = config.encoder.patch_overlap

    n_patches = math.ceil(
        (n_timepoints - patch_size) / (patch_size - overlap_size),
    )

    if (n_timepoints - patch_size) % (patch_size - overlap_size) == 0:
        n_patches += 1

    flat_dim = (n_chans * n_patches + 1) * embed_dim  # +1 for cls token
    return flat_dim


def freeze_model(model: ReveClassifier):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.linear_head.parameters():
        param.requires_grad = True
    model.cls_query_token.requires_grad = True


def unfreeze_model(model: ReveClassifier):
    for param in model.parameters():
        param.requires_grad = True


def load_encoder_checkpoint(encoder: REVE, checkpoint_path: str):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    state_dict = checkpoint.get("model", checkpoint)

    # Remove module. prefix if present (distributed training)
    new_state_dict = {}
    for k, v in state_dict.items():
        k_ = k.replace("module.", "")
        new_state_dict[k_] = v

    # Filter for encoder keys and strip prefix
    encoder_state_dict = {}
    for k, v in new_state_dict.items():
        if k.startswith("encoder."):
            new_key = k.replace("encoder.", "")
            encoder_state_dict[new_key] = v

    if len(encoder_state_dict) == 0:
        print("WARNING: No 'encoder.' keys found in checkpoint. Trying to load as raw encoder weights.")
        encoder_state_dict = new_state_dict

    missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
    print(f"Loaded encoder weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")


def load_cls_query_token(reve_classifier: ReveClassifier, checkpoint_path: str):
    print(f"Loading cls_query_token from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    state_dict = checkpoint.get("model", checkpoint)

    # Remove module. prefix if present (distributed training)
    new_state_dict = {}
    for k, v in state_dict.items():
        k_ = k.replace("module.", "")
        new_state_dict[k_] = v

    cls_key = "cls_query_token"
    if cls_key in new_state_dict:
        reve_classifier.cls_query_token.data.copy_(new_state_dict[cls_key])
        print("Loaded cls_query_token successfully.")
    else:
        print(f"WARNING: {cls_key} not found in checkpoint.")
