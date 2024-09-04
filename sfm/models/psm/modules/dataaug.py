# -*- coding: utf-8 -*-
import torch
from scipy.spatial.transform import Rotation as R


# @torch.compiler.disable(recursive=False)
def uniform_random_rotation(
    batch_size: int, device="cpu", dtype=torch.float32
) -> torch.Tensor:
    """
    Generate a random rotation matrix using QR decomposition.
    Args:
        batch_size (int): batch size
    Returns:
        torch.Tensor: a random rotation matrix [batch_size, 3, 3]
    """
    # scipy version
    rot_mats = torch.from_numpy(R.random(num=batch_size).as_matrix()).to(
        device=device, dtype=dtype
    )

    return rot_mats
