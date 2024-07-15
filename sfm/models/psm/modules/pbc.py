# -*- coding: utf-8 -*-
import numpy as np
import torch

from sfm.logging.loggers import logger


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

        self.cells = torch.tensor(self.cells)

        self.cell_mask_for_pbc = self.cells != 0

        self.cutoff = cutoff

        self.expanded_token_cutoff = expanded_token_cutoff

        self.pbc_multigraph_cutoff = pbc_multigraph_cutoff

        self.pbc_expanded_num_cell_per_direction = pbc_expanded_num_cell_per_direction

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

    def _get_cell_tensors(self, cell, use_local_attention):
        # fitler impossible offsets according to cell size and cutoff
        def _get_max_offset_for_dim(cell, dim):
            lattice_vec_0 = cell[:, dim, :]
            lattice_vec_1_2 = cell[
                :, torch.arange(3, dtype=torch.long, device=cell.device) != dim, :
            ]
            normal_vec = torch.cross(
                lattice_vec_1_2[:, 0, :], lattice_vec_1_2[:, 1, :], dim=-1
            )
            normal_vec = normal_vec / normal_vec.norm(dim=-1, keepdim=True)
            cutoff = self.pbc_multigraph_cutoff if use_local_attention else self.cutoff
            max_offset = int(
                torch.max(
                    torch.ceil(
                        cutoff
                        / torch.abs(torch.sum(normal_vec * lattice_vec_0, dim=-1))
                    )
                )
            )
            return max_offset

        max_offsets = []
        for i in range(3):
            try:
                max_offset = _get_max_offset_for_dim(cell, i)
            except Exception as e:
                logger.warning(f"{e} with cell {cell}")
                max_offset = self.pbc_expanded_num_cell_per_direction
            max_offsets.append(max_offset)
        max_offsets = torch.tensor(max_offsets, device=cell.device)
        self.cells = self.cells.to(device=cell.device)
        self.cell_mask_for_pbc = self.cell_mask_for_pbc.to(device=cell.device)
        mask = (self.cells.abs() <= max_offsets).all(dim=-1)
        selected_cell = self.cells[mask, :]
        return selected_cell, self.cell_mask_for_pbc[mask, :]

    def expand(
        self,
        pos,
        init_pos,
        pbc,
        num_atoms,
        atoms,
        cell,
        pair_token_type,
        use_local_attention=True,
        use_grad=False,
    ):
        with torch.set_grad_enabled(use_grad):
            batch_size, max_num_atoms = pos.size()[:2]
            cell_tensor, cell_mask = self._get_cell_tensors(cell, use_local_attention)
            cell_tensor = (
                cell_tensor.unsqueeze(0).repeat(batch_size, 1, 1).to(dtype=cell.dtype)
            )
            num_expanded_cell = cell_tensor.size()[1]
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
            expand_mask = torch.masked_fill(
                expand_mask, atoms.eq(0).unsqueeze(-1), False
            )
            expand_mask = (torch.sum(expand_mask, dim=1) > 0) & (
                ~(atoms.eq(0).repeat(1, num_expanded_cell))
            )  # B x (8 x T)
            cell_mask = (
                torch.all(pbc.unsqueeze(1) >= cell_mask.unsqueeze(0), dim=-1)
                .unsqueeze(-1)
                .repeat(1, 1, max_num_atoms)
                .reshape(expand_mask.size())
            )  # B x (8 x T)
            expand_mask &= cell_mask
            expand_len = torch.sum(expand_mask, dim=-1)

            threshold_num_expanded_token = torch.clamp(
                self.expanded_token_cutoff - num_atoms, min=0
            )

            max_expand_len = torch.max(expand_len)

            # cutoff within expanded_token_cutoff tokens
            need_threshold = expand_len > threshold_num_expanded_token
            if need_threshold.any():
                min_expand_dist = expand_dist.masked_fill(expand_dist <= 1e-5, np.inf)
                expand_dist_mask = (
                    atoms.eq(0).unsqueeze(-1) | atoms.eq(0).unsqueeze(1)
                ).repeat(1, 1, num_expanded_cell)
                min_expand_dist = min_expand_dist.masked_fill_(expand_dist_mask, np.inf)
                min_expand_dist = min_expand_dist.masked_fill_(
                    ~cell_mask.unsqueeze(1), np.inf
                )
                min_expand_dist = torch.min(min_expand_dist, dim=1)[0]

                need_threshold_distances = min_expand_dist[
                    need_threshold
                ]  # B x (8 x T)
                threshold_num_expanded_token = threshold_num_expanded_token[
                    need_threshold
                ]
                threshold_dist = torch.sort(
                    need_threshold_distances, dim=-1, descending=False
                )[0]

                threshold_dist = torch.gather(
                    threshold_dist, 1, threshold_num_expanded_token.unsqueeze(-1)
                )

                new_expand_mask = min_expand_dist[need_threshold] < threshold_dist
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
            ).repeat(num_expanded_cell)
            for i in range(batch_size):
                outcell_index[i, : expand_len[i]] = outcell_all_index[expand_mask[i]]
                # assert torch.all(outcell_index[i, :expand_len[i]] < natoms[i])
                expand_pos_compressed[i, : expand_len[i], :] = expand_pos[
                    i, expand_mask[i], :
                ]

            expand_pair_token_type = torch.gather(
                pair_token_type,
                dim=2,
                index=outcell_index.unsqueeze(1)
                .unsqueeze(-1)
                .repeat(1, max_num_atoms, 1, pair_token_type.size()[-1]),
            )
            expand_node_type_edge = torch.cat(
                [pair_token_type, expand_pair_token_type], dim=2
            )

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
                pbc_expand_batched = {
                    "expand_pos": expand_pos_compressed,
                    "outcell_index": outcell_index,
                    "expand_mask": expand_mask,
                    "local_attention_weight": local_attention_weight,
                    "expand_node_type_edge": expand_node_type_edge,
                }
            else:
                pbc_expand_batched = {
                    "expand_pos": expand_pos_compressed,
                    "outcell_index": outcell_index,
                    "expand_mask": mask_after_k_persample(
                        batch_size, max_expand_len, expand_len
                    ),
                    "local_attention_weight": None,
                    "expand_node_type_edge": expand_node_type_edge,
                }

            expand_pos_no_offset = torch.gather(
                pos, dim=1, index=outcell_index.unsqueeze(-1)
            )
            offset = expand_pos_compressed - expand_pos_no_offset
            init_expand_pos_no_offset = torch.gather(
                init_pos, dim=1, index=outcell_index.unsqueeze(-1)
            )
            init_expand_pos = init_expand_pos_no_offset + offset
            init_expand_pos = init_expand_pos.masked_fill(
                pbc_expand_batched["expand_mask"].unsqueeze(-1),
                0.0,
            )

            pbc_expand_batched["init_expand_pos"] = init_expand_pos
            return pbc_expand_batched
