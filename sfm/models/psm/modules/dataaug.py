# -*- coding: utf-8 -*-
import torch


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
    A = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)
    R, _ = torch.linalg.qr(A)
    R = R * torch.sign(torch.linalg.det(R)).unsqueeze(-1).unsqueeze(-1)
    return R
