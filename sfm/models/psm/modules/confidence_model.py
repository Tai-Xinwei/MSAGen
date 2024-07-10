# -*- coding: utf-8 -*-
from typing import List, Optional

import torch
from Bio.SVDSuperimposer import SVDSuperimposer
from numpy import flip
from torch import nn


def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers


# evaluation metrics
def _superimpose_np(reference, coords):
    """
    Superimposes coordinates onto a reference by minimizing RMSD using SVD.

    Args:
        reference:
            [N, 3] reference array
        coords:
            [N, 3] array
    Returns:
        A tuple of [N, 3] superimposed coords and the final RMSD.
    """
    sup = SVDSuperimposer()
    sup.set(reference, coords)
    sup.run()
    rot, trans = sup.get_rotran()
    return sup.get_transformed(), sup.get_rms(), rot, trans


def _superimpose_single(reference, coords):
    reference_np = reference.detach().to(torch.float).cpu().numpy()
    coords_np = coords.detach().to(torch.float).cpu().numpy()
    superimposed, rmsd, rot, trans = _superimpose_np(reference_np, coords_np)
    return (
        coords.new_tensor(superimposed),
        coords.new_tensor(rmsd),
        coords.new_tensor(rot),
        coords.new_tensor(trans),
    )


def superimpose(reference, coords, mask):
    """
    Superimposes coordinates onto a reference by minimizing RMSD using SVD.

    Args:
        reference:
            [*, N, 3] reference tensor
        coords:
            [*, N, 3] tensor
        mask:
            [*, N] tensor
    Returns:
        A tuple of [*, N, 3] superimposed coords and [*] final RMSDs, and [*, 3, 3] rots.
    """

    def select_unmasked_coords(coords, mask):
        return torch.masked_select(
            coords,
            (mask > 0.0)[..., None],
        ).reshape(-1, 3)

    batch_dims = reference.shape[:-2]
    flat_reference = reference.reshape((-1,) + reference.shape[-2:])
    flat_coords = coords.reshape((-1,) + reference.shape[-2:])
    flat_mask = mask.reshape((-1,) + mask.shape[-1:])
    superimposed_list = []
    rmsds = []
    for r, c, m in zip(flat_reference, flat_coords, flat_mask):
        r_unmasked_coords = select_unmasked_coords(r, m)
        c_unmasked_coords = select_unmasked_coords(c, m)
        superimposed, rmsd, rot, trans = _superimpose_single(
            r_unmasked_coords, c_unmasked_coords
        )

        # This is very inelegant, but idk how else to invert the masking
        # procedure.
        count = 0
        superimposed_full_size = torch.zeros_like(r)
        for i, unmasked in enumerate(m):
            if unmasked:
                superimposed_full_size[i] = superimposed[count]
                count += 1

        superimposed_list.append(superimposed_full_size)
        rmsds.append(rmsd)

    superimposed_stacked = torch.stack(superimposed_list, dim=0)
    rmsds_stacked = torch.stack(rmsds, dim=0)

    superimposed_reshaped = superimposed_stacked.reshape(batch_dims + coords.shape[-2:])
    rmsds_reshaped = rmsds_stacked.reshape(batch_dims)

    return superimposed_reshaped, rmsds_reshaped


def drmsd(structure_1, structure_2, mask=None):
    def prep_d(structure):
        d = structure[..., :, None, :] - structure[..., None, :, :]
        d = d**2
        d = torch.sqrt(torch.sum(d, dim=-1))
        return d

    d1 = prep_d(structure_1)
    d2 = prep_d(structure_2)

    drmsd = d1 - d2
    drmsd = drmsd**2
    if mask is not None:
        drmsd = drmsd * (mask[..., None] * mask[..., None, :])
    drmsd = torch.sum(drmsd, dim=(-1, -2))
    n = d1.shape[-1] if mask is None else torch.min(torch.sum(mask, dim=-1))
    drmsd = drmsd * (1 / (n * (n - 1))) if n > 1 else (drmsd * 0.0)
    drmsd = torch.sqrt(drmsd)

    return drmsd


def drmsd_np(structure_1, structure_2, mask=None):
    structure_1 = torch.tensor(structure_1)
    structure_2 = torch.tensor(structure_2)
    if mask is not None:
        mask = torch.tensor(mask)

    return drmsd(structure_1, structure_2, mask)


def gdt(p1, p2, mask, cutoffs):
    n = torch.sum(mask, dim=-1)

    p1 = p1.float()
    p2 = p2.float()
    distances = torch.sqrt(torch.sum((p1 - p2) ** 2, dim=-1))
    scores = []
    for c in cutoffs:
        score = torch.sum((distances <= c) * mask, dim=-1) / n
        score = torch.mean(score)
        scores.append(score)

    return sum(scores) / len(scores)


def gdt_ts(p1, p2, mask):
    return gdt(p1, p2, mask, [1.0, 2.0, 4.0, 8.0])


def gdt_ha(p1, p2, mask):
    return gdt(p1, p2, mask, [0.5, 1.0, 2.0, 4.0])


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100


def lddt(
    all_atom_pred_pos: torch.Tensor,  # (B, N, 3)
    all_atom_positions: torch.Tensor,  # (B, N, 3)
    all_atom_mask: torch.Tensor,  # (B, N)
    cutoff: float = 15.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    # use torch built-in function norm to compute pairwise distances
    dmat_true = torch.norm(
        all_atom_positions[..., None, :] - all_atom_positions[..., None, :, :] + eps,
        dim=-1,
        p=2,
    )
    dmat_pred = torch.norm(
        all_atom_pred_pos[..., None, :] - all_atom_pred_pos[..., None, :, :] + eps,
        dim=-1,
        p=2,
    )
    dists_to_score = dmat_true < cutoff
    dists_to_score = dists_to_score * all_atom_mask.unsqueeze(-1)
    dists_to_score = dists_to_score * all_atom_mask.unsqueeze(-2)
    dists_to_score = dists_to_score * (
        1.0
        - torch.eye(all_atom_mask.shape[1], device=all_atom_mask.device).unsqueeze(0)
    )
    # (B, N, N) * (1, N, N)
    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    ) * 0.25  # (B, N, N)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=-1))  # (B, N)
    score = norm * (eps + torch.sum(dists_to_score * score, dim=-1))
    return score


def lddt_loss(
    logits: torch.Tensor,
    ca_atom_pred_pos: torch.Tensor,  # (B, 1, 3)
    ca_atom_positions: torch.Tensor,  # (B, 1, 3)
    ca_atom_mask: torch.Tensor,  # (B, N)
    resolution: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    score = lddt(
        ca_atom_pred_pos, ca_atom_positions, ca_atom_mask, cutoff=cutoff, eps=eps
    )

    # TODO: Remove after initial pipeline testing
    score = torch.nan_to_num(score, nan=torch.nanmean(score))
    score[score < 0] = 0

    score = score.detach()
    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(bin_index, num_classes=no_bins)
    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    # accuracy
    acc = torch.sum(
        (torch.argmax(logits, dim=-1) == bin_index).type(logits.dtype) * ca_atom_mask
    ) / (eps + torch.sum(ca_atom_mask))
    ca_atom_mask = ca_atom_mask.squeeze(-1)
    loss = torch.sum(errors * ca_atom_mask, dim=-1) / (
        eps + torch.sum(ca_atom_mask, dim=-1)
    )

    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))
    # Average over the batch dimension
    loss = torch.mean(loss)
    return loss, acc


def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    asym_id: Optional[torch.Tensor] = None,
    interface: bool = False,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)

    bin_centers = _calculate_bin_centers(boundaries)
    clipped_n = max(torch.sum(residue_weights), 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    n = residue_weights.shape[-1]
    pair_mask = residue_weights.new_ones((n, n), dtype=torch.int32)
    if interface and (asym_id is not None):
        if len(asym_id.shape) > 1:
            assert len(asym_id.shape) <= 2
            batch_size = asym_id.shape[0]
            pair_mask = residue_weights.new_ones((batch_size, n, n), dtype=torch.int32)
        pair_mask *= (asym_id[..., None] != asym_id[..., None, :]).to(
            dtype=pair_mask.dtype
        )

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask * (
        residue_weights[..., None, :] * residue_weights[..., :, None]
    )
    denom = eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
    normed_residue_mask = pair_residue_weights / denom
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights

    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]


class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of bins
        """
        super(TMScoreHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = nn.Linear(self.c_z, self.no_bins)
        # same way as openfold
        with torch.no_grad():
            self.linear.weight.data.fill_(0.0)

    def forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pairwise embedding
        Returns:
            [*, N_res, N_res, no_bins] prediction
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        return logits
