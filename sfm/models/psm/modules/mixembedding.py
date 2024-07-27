# -*- coding: utf-8 -*-
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from sfm.models.psm.invariant.mixture_bias import GaussianLayer, NonLinear
from sfm.models.psm.modules.timestep_encoder import TimeStepEncoder
from sfm.models.psm.psm_config import PSMConfig


class PSMMix3dEmbedding(nn.Module):
    """
    Class for the embedding layer in the PSM model.
    """

    def __init__(self, psm_config: PSMConfig, use_unified_batch_sampler: bool = False):
        """
        Initialize the PSMMixEmbedding class.
        ## [1, 128]: atom type; [129, 159] amino acid type
        """
        super(PSMMix3dEmbedding, self).__init__()

        ## [1, 128]: atom type; [129, 154] amino acid type
        self.embed = nn.Embedding(160, psm_config.encoder_embed_dim)

        # embedding for 2D
        self.atom_feature_embed = nn.Embedding(
            psm_config.num_atom_features, psm_config.encoder_embed_dim
        )

        self.time_step_encoder = TimeStepEncoder(
            psm_config.num_timesteps,
            psm_config.embedding_dim,
            psm_config.diffusion_time_step_encoder_type,
        )

        self.pos_emb = nn.Linear(3, psm_config.num_3d_bias_kernel, bias=False)
        self.pos_feature_emb = nn.Linear(
            psm_config.num_3d_bias_kernel, psm_config.embedding_dim, bias=False
        )

        self.scaling = (psm_config.num_3d_bias_kernel) ** -0.5

        self.psm_config = psm_config
        self.use_unified_batch_sampler = use_unified_batch_sampler

        self.gbf = GaussianLayer(psm_config.num_3d_bias_kernel, psm_config.num_edges)
        self.gbf_proj = NonLinear(
            psm_config.num_3d_bias_kernel, psm_config.num_attention_heads
        )
        self.pos_embedding_proj = nn.Linear(
            psm_config.num_3d_bias_kernel, psm_config.embedding_dim
        )

    @torch.compiler.disable(recursive=False)
    def center_pos(self, expand_pos, expand_mask):
        expand_pos = expand_pos.masked_fill(expand_mask.unsqueeze(-1), 0.0)
        center_pos = torch.sum(expand_pos, dim=1, keepdim=True) / (~expand_mask).sum(
            dim=1
        ).unsqueeze(-1).unsqueeze(-1)
        expand_pos = expand_pos - center_pos
        return expand_pos

    @torch.compiler.disable(recursive=False)
    def _pos_emb(
        self,
        pos: Optional[torch.Tensor],
        expand_pos: torch.Tensor,
        adj: torch.Tensor,
        molecule_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        batched_data: torch.Tensor,
        pbc_expand_batched: Optional[Dict[str, Tensor]] = None,
    ):
        pos = pos.to(self.pos_emb.weight.dtype)

        # inf_pos_mask = pos.eq(float("inf")).any(dim=-1)
        # pos = pos.masked_fill(inf_pos_mask.unsqueeze(-1), 0.0)

        if pbc_expand_batched is not None:
            # assert (
            #     self.use_unified_batch_sampler
            # ), "Only support unified batch sampler for now"
            expand_pos = expand_pos.to(self.pos_emb.weight.dtype)
            expand_pos = torch.cat([pos, expand_pos], dim=1)
            expand_mask = torch.cat(
                [padding_mask, pbc_expand_batched["expand_mask"]], dim=-1
            )

            expand_pos = self.center_pos(expand_pos, expand_mask)
            delta_pos = pos.unsqueeze(2) - expand_pos.unsqueeze(1)
            B, L, expand_L = delta_pos.size()[:3]
            adj = torch.ones(B, L, expand_L, device=adj.device, dtype=torch.bool)
            local_attention_weight = pbc_expand_batched["local_attention_weight"]
            node_type_edge = pbc_expand_batched["expand_node_type_edge"]
        else:
            delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
            expand_mask = padding_mask
            expand_pos = pos
            local_attention_weight = None
            node_type_edge = batched_data["node_type_edge"]

        dist = delta_pos.norm(dim=-1)

        edge_feature = self.gbf(dist, node_type_edge.long())
        graph_attn_bias = self.gbf_proj(edge_feature)
        graph_attn_bias = graph_attn_bias.masked_fill(
            expand_mask.unsqueeze(1).unsqueeze(-1), float("-inf")
        )
        graph_attn_bias = graph_attn_bias.masked_fill(
            padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0
        )
        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2)

        dist = graph_attn_bias.sum(dim=1)
        min_dtype = torch.finfo(dist.dtype).min
        dist = dist.masked_fill(expand_mask.unsqueeze(1), min_dtype)
        dist = dist.masked_fill(padding_mask.unsqueeze(-1), min_dtype)
        if local_attention_weight is not None:
            local_attention_weight = local_attention_weight.to(dtype=pos.dtype)
            dist = dist.masked_fill(local_attention_weight <= 1e-5, min_dtype)

        pos_emb = self.pos_emb(expand_pos).masked_fill(expand_mask.unsqueeze(-1), 0.0)
        # pos_emb = torch.where(inf_pos_mask.unsqueeze(-1), self.unkpos_embedding.weight, pos_emb)

        adj = adj.masked_fill(~molecule_mask.unsqueeze(-1), True)
        dist = dist.masked_fill(~adj, min_dtype)

        dist = torch.nn.functional.softmax(dist.float() * self.scaling, dim=-1)
        if local_attention_weight is not None:
            dist = dist * local_attention_weight
            edge_feature = edge_feature * local_attention_weight.unsqueeze(-1)

        pos_feature_emb = torch.matmul(dist, pos_emb).to(self.pos_emb.weight.dtype)
        pos_feature_emb = self.pos_feature_emb(pos_feature_emb)

        edge_feature = edge_feature.masked_fill(
            expand_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )

        edge_feature = edge_feature.sum(dim=-2)
        pos_feature_emb += self.pos_embedding_proj(edge_feature)

        pos_feature_emb = pos_feature_emb.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        ).to(self.pos_emb.weight.dtype)
        return pos_feature_emb, graph_attn_bias

    def forward(
        self,
        batched_data: Dict,
        time_step: Optional[Tensor] = None,
        clean_mask: Optional[Tensor] = None,
        aa_mask: Optional[Tensor] = None,
        pbc_expand_batched: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Forward pass of the PSMMixEmbedding class.
        Args:
            batched_data: Input data for the forward pass.
        Returns:
            x: The embedding representation.
            padding_mask: The padding mask.
        """
        token_id = batched_data["token_id"]
        padding_mask = token_id.eq(0)  # B x T x 1
        is_periodic = batched_data["is_periodic"]
        molecule_mask = (token_id <= 129) & (~is_periodic.unsqueeze(-1))

        if aa_mask is not None:
            mask_token_type = token_id.masked_fill(
                aa_mask, 157
            )  # 157 is the mask token
        else:
            mask_token_type = token_id

        batched_data["masked_token_type"] = mask_token_type
        x = self.embed(mask_token_type)

        if self.psm_config.use_2d_atom_features and "node_attr" in batched_data:
            atom_feature_embedding = self.atom_feature_embed(
                batched_data["node_attr"][:, :, 1:]
            ).sum(
                dim=-2
            )  # B x T x #ATOM_FEATURE x D -> # B x T x D
            atom_feature_embedding = atom_feature_embedding.masked_fill(
                ~molecule_mask.unsqueeze(-1), 0.0
            )

            # # time raito is 0 at time step == 0, time raito is 1 at time step >= 0.05, linear increase between 0 and 0.05
            time_ratio = torch.clamp(time_step / 0.001, 0.0, 1.0)
            x += atom_feature_embedding * time_ratio.unsqueeze(-1)

        pos_embedding, pos_attn_bias = self._pos_emb(
            batched_data["pos"],
            pbc_expand_batched["expand_pos"]
            if pbc_expand_batched is not None
            else None,
            batched_data["adj"],
            molecule_mask,
            padding_mask,
            batched_data,
            pbc_expand_batched=pbc_expand_batched,
        )

        if time_step is not None:
            time_embed = self.time_step_encoder(time_step, clean_mask)
            x += time_embed

        if "init_pos" in batched_data and (batched_data["init_pos"] != 0.0).any():
            init_pos = batched_data["init_pos"]
            init_pos_embedding, init_pos_attn_bias = self._pos_emb(
                init_pos,
                pbc_expand_batched["init_expand_pos"]
                if pbc_expand_batched is not None
                else None,
                batched_data["adj"],
                molecule_mask,
                padding_mask,
                batched_data,
                pbc_expand_batched=pbc_expand_batched,
            )
            init_pos_mask = (
                (init_pos != 0.0).any(dim=-1, keepdim=False).any(dim=-1, keepdim=False)
            )
            pos_embedding[init_pos_mask, :] += init_pos_embedding[init_pos_mask, :]

            init_pos_attn_bias = init_pos_attn_bias.masked_fill(
                ~init_pos_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0
            )
            pos_attn_bias += init_pos_attn_bias

        x += pos_embedding

        return x, padding_mask, time_embed, pos_attn_bias, pos_embedding


class PSMMix3dDitEmbedding(PSMMix3dEmbedding):
    """
    Class for the embedding layer in the PSM model.
    """

    def forward(
        self,
        batched_data: Dict,
        time_step: Optional[Tensor] = None,
        clean_mask: Optional[Tensor] = None,
        aa_mask: Optional[Tensor] = None,
        pbc_expand_batched: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Forward pass of the PSMMixEmbedding class.
        Args:
            batched_data: Input data for the forward pass.
        Returns:
            x: The embedding representation.
            padding_mask: The padding mask.
        """
        token_id = batched_data["token_id"]
        padding_mask = token_id.eq(0)  # B x T x 1
        is_periodic = batched_data["is_periodic"]
        molecule_mask = (token_id <= 129) & (~is_periodic.unsqueeze(-1))

        if aa_mask is not None:
            mask_token_type = token_id.masked_fill(
                aa_mask, 157
            )  # 157 is the mask token
        else:
            mask_token_type = token_id

        batched_data["masked_token_type"] = mask_token_type
        c = self.embed(mask_token_type)

        if self.psm_config.use_2d_atom_features and "node_attr" in batched_data:
            atom_feature_embedding = self.atom_feature_embed(
                batched_data["node_attr"][:, :, 1:]
            ).sum(
                dim=-2
            )  # B x T x #ATOM_FEATURE x D -> # B x T x D
            atom_feature_embedding = atom_feature_embedding.masked_fill(
                ~molecule_mask.unsqueeze(-1), 0.0
            )

            # # time raito is 0 at time step == 0, time raito is 1 at time step >= 0.05, linear increase between 0 and 0.05
            time_ratio = torch.clamp(time_step / 0.001, 0.0, 1.0)
            c += atom_feature_embedding * time_ratio.unsqueeze(-1)

        pos_embedding, pos_attn_bias = self._pos_emb(
            batched_data["pos"],
            pbc_expand_batched["expand_pos"]
            if pbc_expand_batched is not None
            else None,
            batched_data["adj"],
            molecule_mask,
            padding_mask,
            batched_data,
            pbc_expand_batched=pbc_expand_batched,
        )
        if time_step is not None:
            time_embed = self.time_step_encoder(time_step, clean_mask)

        if "init_pos" in batched_data and (batched_data["init_pos"] != 0.0).any():
            init_pos = batched_data["init_pos"]
            init_pos_embedding, init_pos_attn_bias = self._pos_emb(
                init_pos,
                pbc_expand_batched["init_expand_pos"]
                if pbc_expand_batched is not None
                else None,
                batched_data["adj"],
                molecule_mask,
                padding_mask,
                batched_data,
                pbc_expand_batched=pbc_expand_batched,
            )
            init_pos_mask = (
                (init_pos != 0.0).any(dim=-1, keepdim=False).any(dim=-1, keepdim=False)
            )
            pos_embedding[init_pos_mask, :] = init_pos_embedding[init_pos_mask, :]

            init_pos_attn_bias = init_pos_attn_bias.masked_fill(
                ~init_pos_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0
            )
            pos_attn_bias += init_pos_attn_bias

        pos_embedding += time_embed

        return pos_embedding, padding_mask, time_embed, pos_attn_bias, c
