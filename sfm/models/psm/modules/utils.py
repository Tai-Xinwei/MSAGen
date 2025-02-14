# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def batched_gather(
    data: torch.Tensor, inds: torch.Tensor, dim: int = 0, no_batch_dims: int = 0
) -> torch.Tensor:
    """Gather data according to indices specify by inds

    Args:
        data (torch.Tensor): the input data
            [..., K, ...]
        inds (torch.Tensor): the indices for gathering data
            [..., N]
        dim (int, optional): along which dimension to gather data by inds (the dim of "K" "N"). Defaults to 0.
        no_batch_dims (int, optional): length of dimensions before the "dim" dimension. Defaults to 0.

    Returns:
        torch.Tensor: gathered data
            [..., N, ...]
    """

    # for the naive case
    if len(inds.shape) == 1 and no_batch_dims == 0 and dim == 0:
        return data[inds]

    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def expressCoordinatesInFrame(
    coordinate: torch.Tensor, frames: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Algorithm 29 Express coordinate in frame

    Args:
        coordinate (torch.Tensor): the input coordinate
            [..., N_atom, 3]
        frames (torch.Tensor): the input frames
            [..., N_frame, 3, 3]
        eps (float): Small epsilon value

    Returns:
        torch.Tensor: the transformed coordinate projected onto frame basis
            [..., N_frame, N_atom, 3]
    """
    # Extract frame atoms
    a, b, c = torch.unbind(frames, dim=-2)  # a, b, c shape: [..., N_frame, 3]
    w1 = F.normalize(a - b, dim=-1, eps=eps)
    w2 = F.normalize(c - b, dim=-1, eps=eps)
    # Build orthonormal basis
    e1 = F.normalize(w1 + w2, dim=-1, eps=eps)
    e2 = F.normalize(w2 - w1, dim=-1, eps=eps)
    e3 = torch.cross(e1, e2, dim=-1)  # [..., N_frame, 3]
    # Project onto frame basis
    d = coordinate[..., None, :, :] - b[..., None, :]  #  [..., N_frame, N_atom, 3]
    x_transformed = torch.cat(
        [
            torch.sum(d * e1[..., None, :], dim=-1, keepdim=True),
            torch.sum(d * e2[..., None, :], dim=-1, keepdim=True),
            torch.sum(d * e3[..., None, :], dim=-1, keepdim=True),
        ],
        dim=-1,
    )  # [..., N_frame, N_atom, 3]
    return x_transformed


def gather_frame_atom_by_indices(
    coordinate: torch.Tensor, frame_atom_index: torch.Tensor, dim: int = -2
) -> torch.Tensor:
    """construct frames from coordinate

    Args:
        coordinate (torch.Tensor):  the input coordinate
            [..., N_atom, 3]
        frame_atom_index (torch.Tensor): indices of three atoms in each frame
            [..., N_frame, 3] or [N_frame, 3]
        dim (torch.Tensor): along which dimension to select the frame atoms
    Returns:
        torch.Tensor: the constructed frames
            [..., N_frame, 3[three atom], 3[three coordinate]]
    """
    if len(frame_atom_index.shape) == 2:
        # the navie case
        x1 = torch.index_select(
            coordinate, dim=dim, index=frame_atom_index[:, 0]
        )  # [..., N_frame, 3]
        x2 = torch.index_select(
            coordinate, dim=dim, index=frame_atom_index[:, 1]
        )  # [..., N_frame, 3]
        x3 = torch.index_select(
            coordinate, dim=dim, index=frame_atom_index[:, 2]
        )  # [..., N_frame, 3]
        return torch.stack([x1, x2, x3], dim=dim)
    else:
        assert (
            frame_atom_index.shape[:dim] == coordinate.shape[:dim]
        ), "batch size dims should match"

    x1 = batched_gather(
        data=coordinate,
        inds=frame_atom_index[..., 0],
        dim=dim,
        no_batch_dims=len(coordinate.shape[:dim]),
    )  # [..., N_frame, 3]
    x2 = batched_gather(
        data=coordinate,
        inds=frame_atom_index[..., 1],
        dim=dim,
        no_batch_dims=len(coordinate.shape[:dim]),
    )  # [..., N_frame, 3]
    x3 = batched_gather(
        data=coordinate,
        inds=frame_atom_index[..., 2],
        dim=dim,
        no_batch_dims=len(coordinate.shape[:dim]),
    )  # [..., N_frame, 3]
    return torch.stack([x1, x2, x3], dim=dim)
