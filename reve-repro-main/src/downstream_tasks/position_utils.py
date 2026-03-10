from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModel


@lru_cache(maxsize=1)
def _get_position_model():
    return AutoModel.from_pretrained(
        "brain-bzh/reve-positions",
        trust_remote_code=True,
        dtype="auto",
        cache_dir=".cache",
    )


def load_positions(positions_path=None, electrode_names=None):
    """
    Loads electrode positions.

    Args:
        positions_path (str, optional): Path to .npy file containing positions.
        electrode_names (list[str], optional): List of electrode names.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) containing 3D coordinates.
    """

    if electrode_names is not None and len(electrode_names) > 0:
        print("Loading from electrode names")
        model = _get_position_model()
        with torch.no_grad():
            positions = model(electrode_names)
        return positions.float().cpu()

    if positions_path is not None:
        print("Loading from positions path")
        try:
            positions_ = np.load(positions_path, allow_pickle=True)
            return torch.from_numpy(positions_).float()
        except Exception as e:
            print(f"Failed to load positions from {positions_path}: {e}")
            raise e

    raise ValueError("Either 'electrode_names' or 'positions_path' must be provided.")
