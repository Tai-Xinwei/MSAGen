# -*- coding: utf-8 -*-
import numpy as np
import torch


@torch.jit.script
def mask_after_k_persample(n_sample: int, n_len: int, persample_k: torch.Tensor):
    assert persample_k.shape[0] == n_sample
    assert persample_k.max() <= n_len
    device = persample_k.device
    mask = torch.zeros([n_sample, n_len + 1], device=device)
    mask[torch.arange(n_sample, device=device), persample_k] = 1
    mask = mask.cumsum(dim=1)[:, :-1]
    return mask.type(torch.bool)


class CellExpander:
    def __init__(
        self,
        cutoff=10.0,
        expanded_token_cutoff=256,
        pbc_expanded_num_cell_per_direction=10,
        pbc_multigraph_cutoff=10.0,
    ):
        self.cells = []
        for i in range(
            -pbc_expanded_num_cell_per_direction,
            pbc_expanded_num_cell_per_direction + 1,
        ):
            for j in range(
                -pbc_expanded_num_cell_per_direction,
                pbc_expanded_num_cell_per_direction + 1,
            ):
                for k in range(
                    -pbc_expanded_num_cell_per_direction,
                    pbc_expanded_num_cell_per_direction + 1,
                ):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    self.cells.append([i, j, k])

        self.cell_mask_for_pbc = torch.tensor(self.cells) != 0

        self.cutoff = cutoff

        self.expanded_token_cutoff = expanded_token_cutoff

        self.pbc_multigraph_cutoff = pbc_multigraph_cutoff

    def polynomial(self, dist: torch.Tensor, cutoff: float) -> torch.Tensor:
        """
        Polynomial cutoff function,ref: https://arxiv.org/abs/2204.13639
        Args:
            dist (tf.Tensor): distance tensor
            cutoff (float): cutoff distance
        Returns: polynomial cutoff functions
        """
        ratio = torch.div(dist, cutoff)
        result = (
            1
            - 6 * torch.pow(ratio, 5)
            + 15 * torch.pow(ratio, 4)
            - 10 * torch.pow(ratio, 3)
        )
        return torch.clamp(result, min=0.0)

    def expand(self, pos, pbc, atoms, cell, use_local_attention=True):
        with torch.no_grad():  # CL: make this an option?
            device = pos.device
            batch_size, max_num_atoms = pos.size()[:2]
            cell_tensor = (
                torch.tensor(self.cells, device=pos.device)
                .to(cell.dtype)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            offset = torch.bmm(cell_tensor, cell)  # B x 8 x 3
            expand_pos = pos.unsqueeze(1) + offset.unsqueeze(2)  # B x 8 x T x 3
            expand_pos = expand_pos.view(batch_size, -1, 3)  # B x (8 x T) x 3
            expand_dist = torch.norm(
                pos.unsqueeze(2) - expand_pos.unsqueeze(1), p=2, dim=-1
            )  # B x T x (8 x T)
            expand_mask = (expand_dist < self.cutoff) & (
                expand_dist
                > 1e-5  # CL: this cannot mask out colocated nodes in `expand_pos`
            )  # B x T x (8 x T)
            # multi_ragph_mask = expand_dist < self.pbc_multigraph_cutoff
            expand_mask = torch.masked_fill(
                expand_mask, atoms.eq(0).unsqueeze(-1), False
            )
            expand_mask = (torch.sum(expand_mask, dim=1) > 0) & (
                ~(atoms.eq(0).repeat(1, len(self.cells)))
            )  # B x (8 x T)
            cell_mask = self.cell_mask_for_pbc.to(device=device)
            cell_mask = (
                torch.all(pbc.unsqueeze(1) >= cell_mask.unsqueeze(0), dim=-1)
                .unsqueeze(-1)
                .repeat(1, 1, max_num_atoms)
                .reshape(expand_mask.size())
            )  # B x (8 x T)
            expand_mask &= cell_mask
            expand_len = torch.sum(expand_mask, dim=-1)

            max_expand_len = torch.max(expand_len)

            threshold_num_expanded_token = (
                0
                if max_num_atoms > self.expanded_token_cutoff
                else self.expanded_token_cutoff - max_num_atoms
            )

            # cutoff within expanded_token_cutoff tokens
            if max_expand_len > threshold_num_expanded_token:
                min_expand_dist = expand_dist.masked_fill(expand_dist <= 1e-5, np.inf)
                expand_dist_mask = (
                    atoms.eq(0).unsqueeze(-1) | atoms.eq(0).unsqueeze(1)
                ).repeat(1, 1, len(self.cells))
                min_expand_dist = min_expand_dist.masked_fill_(expand_dist_mask, np.inf)
                min_expand_dist = min_expand_dist.masked_fill_(
                    ~cell_mask.unsqueeze(1), np.inf
                )
                min_expand_dist = torch.min(min_expand_dist, dim=1)[0]
                need_threshold = expand_len > threshold_num_expanded_token

                need_threshold_distances = min_expand_dist[
                    need_threshold
                ]  # B x (8 x T)
                threshold_dist = torch.sort(
                    need_threshold_distances, dim=-1, descending=False
                )[0][:, threshold_num_expanded_token]

                new_expand_mask = min_expand_dist[
                    need_threshold
                ] < threshold_dist.unsqueeze(-1)
                expand_mask[need_threshold] &= new_expand_mask
                expand_len = torch.sum(expand_mask, dim=-1)
                max_expand_len = torch.max(expand_len)

            outcell_index = torch.zeros(
                [batch_size, max_expand_len], dtype=torch.long, device=pos.device
            )
            expand_pos_compressed = torch.zeros(
                [batch_size, max_expand_len, 3], dtype=pos.dtype, device=pos.device
            )
            outcell_all_index = torch.arange(
                max_num_atoms, dtype=torch.long, device=pos.device
            ).repeat(len(self.cells))
            for i in range(batch_size):
                outcell_index[i, : expand_len[i]] = outcell_all_index[expand_mask[i]]
                # assert torch.all(outcell_index[i, :expand_len[i]] < natoms[i])
                expand_pos_compressed[i, : expand_len[i], :] = expand_pos[
                    i, expand_mask[i], :
                ]

            if use_local_attention:
                dist = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(p=2, dim=-1)
                expand_dist_compress = (
                    pos.unsqueeze(2) - expand_pos_compressed.unsqueeze(1)
                ).norm(p=2, dim=-1)
                local_attention_weight = self.polynomial(
                    torch.cat([dist, expand_dist_compress], dim=2),
                    cutoff=self.pbc_multigraph_cutoff,
                )
                is_periodic = pbc.any(dim=-1)
                local_attention_weight = local_attention_weight.masked_fill(
                    ~is_periodic.unsqueeze(-1).unsqueeze(-1), 1.0
                )
                local_attention_weight = local_attention_weight.masked_fill(
                    atoms.eq(0).unsqueeze(-1), 1.0
                )
                expand_mask = mask_after_k_persample(
                    batch_size, max_expand_len, expand_len
                )
                full_mask = torch.cat([atoms.eq(0), expand_mask], dim=-1)
                local_attention_weight = local_attention_weight.masked_fill(
                    atoms.eq(0).unsqueeze(-1), 1.0
                )
                local_attention_weight = local_attention_weight.masked_fill(
                    full_mask.unsqueeze(1), 0.0
                )
                return {
                    "expand_pos": expand_pos_compressed,
                    "outcell_index": outcell_index,
                    "expand_mask": expand_mask,
                    "local_attention_weight": local_attention_weight,
                }
            else:
                return {
                    "expand_pos": expand_pos_compressed,
                    "outcell_index": outcell_index,
                    "expand_mask": mask_after_k_persample(
                        batch_size, max_expand_len, expand_len
                    ),
                    "local_attention_weight": None,
                }
