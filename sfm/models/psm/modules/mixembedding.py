# -*- coding: utf-8 -*-
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from sfm.models.psm.invariant.mixture_bias import GaussianLayer, NonLinear
from sfm.models.psm.modules.timestep_encoder import (
    FourierEmbeddingAF3,
    PositionalEmbeddingEDM,
    TimeStepEncoder,
)
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
        self.psm_config = psm_config

        if psm_config.diffusion_mode == "edm":
            if psm_config.noise_embedding == "positional":
                self.noise_cond_embed_edm = PositionalEmbeddingEDM(
                    num_channels=psm_config.embedding_dim,
                )
            elif psm_config.noise_embedding == "fourier":
                self.noise_cond_embed_edm = FourierEmbeddingAF3(
                    num_channels=psm_config.embedding_dim,
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
        clean_mask: torch.Tensor,
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

            # expand_pos = self.center_pos(expand_pos, expand_mask)
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

        adj = adj.masked_fill(~molecule_mask.unsqueeze(-1), True)
        if clean_mask is not None:
            adj = adj.masked_fill(clean_mask.unsqueeze(-1), True)

        edge_feature = self.gbf(dist, node_type_edge.long())
        graph_attn_bias = self.gbf_proj(edge_feature)
        graph_attn_bias = graph_attn_bias.masked_fill(
            expand_mask.unsqueeze(1).unsqueeze(-1), float("-inf")
        )
        graph_attn_bias = graph_attn_bias.masked_fill(~adj.unsqueeze(-1), float("-inf"))
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
        time_step_1d: Optional[Tensor] = None,
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

        if self.psm_config.diffusion_mode == "edm":
            noise_embed_edm = self.noise_cond_embed_edm(
                batched_data["c_noise"].flatten()
            ).to(x.dtype)
            time_embed = noise_embed_edm.reshape((x.size(0), x.size(1), -1))
        elif time_step is not None:
            time_embed = self.time_step_encoder(time_step, clean_mask)

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
            # time_ratio = torch.clamp(time_step / 0.001, 0.0, 1.0)
            x += atom_feature_embedding  # * time_ratio.unsqueeze(-1)

        pos_embedding, pos_attn_bias = self._pos_emb(
            batched_data["pos"],
            pbc_expand_batched["expand_pos"]
            if pbc_expand_batched is not None
            else None,
            batched_data["adj"],
            clean_mask,
            molecule_mask,
            padding_mask,
            batched_data,
            pbc_expand_batched=pbc_expand_batched,
        )

        x += time_embed

        if "init_pos" in batched_data and (batched_data["init_pos"] != 0.0).any():
            init_pos = batched_data["init_pos"]
            init_pos_embedding, init_pos_attn_bias = self._pos_emb(
                init_pos,
                pbc_expand_batched["init_expand_pos"]
                if pbc_expand_batched is not None
                else None,
                batched_data["adj"],
                clean_mask,
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

    def __init__(self, psm_config: PSMConfig, use_unified_batch_sampler: bool = False):
        super(PSMMix3dDitEmbedding, self).__init__(
            psm_config, use_unified_batch_sampler
        )
        self.gbf = GaussianLayer(
            psm_config.num_3d_bias_kernel // 2, psm_config.num_edges
        )
        self.mol_graph_2d_bond_feat = nn.Embedding(
            2, psm_config.num_3d_bias_kernel // 2
        )

        self.init_pos_emb = nn.Linear(3, psm_config.embedding_dim, bias=False)

        if psm_config.diffusion_mode == "edm":
            if psm_config.noise_embedding == "positional":
                self.noise_cond_embed_edm = PositionalEmbeddingEDM(
                    num_channels=psm_config.embedding_dim,
                )
            elif psm_config.noise_embedding == "fourier":
                self.noise_cond_embed_edm = FourierEmbeddingAF3(
                    num_channels=psm_config.embedding_dim,
                )

    @torch.compiler.disable(recursive=False)
    def _pos_emb(
        self,
        pos: Optional[torch.Tensor],
        expand_pos: torch.Tensor,
        adj: torch.Tensor,
        clean_mask: torch.Tensor,
        molecule_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        batched_data: torch.Tensor,
        pbc_expand_batched: Optional[Dict[str, Tensor]] = None,
    ):
        pos = pos.to(self.pos_emb.weight.dtype)  # / self.psm_config.diffusion_noise_std

        # inf_nan_mask = batched_data["protein_mask"].any(dim=-1)

        if pbc_expand_batched is not None:
            expand_pos = expand_pos.to(
                self.pos_emb.weight.dtype
            )  # / self.psm_config.diffusion_noise_std
            expand_pos = torch.cat([pos, expand_pos], dim=1)
            expand_mask = torch.cat(
                [padding_mask, pbc_expand_batched["expand_mask"]], dim=-1
            )

            # expand_pos = self.center_pos(expand_pos, expand_mask)
            delta_pos = pos.unsqueeze(2) - expand_pos.unsqueeze(1)
            B, L, expand_L = delta_pos.size()[:3]
            adj = torch.ones(B, L, expand_L, device=adj.device, dtype=torch.bool)
            local_attention_weight = pbc_expand_batched["local_attention_weight"]
            node_type_edge = pbc_expand_batched["expand_node_type_edge"]
            expand_molecule_mask = torch.cat(
                [
                    molecule_mask,
                    torch.zeros_like(
                        pbc_expand_batched["expand_mask"], dtype=torch.bool
                    ),
                ],
                dim=-1,
            )
            expand_clean_mask = torch.cat(
                [
                    clean_mask,
                    torch.ones_like(
                        pbc_expand_batched["expand_mask"], dtype=torch.bool
                    ),
                ],
                dim=-1,
            )
        else:
            delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
            padding_mask = padding_mask | batched_data["token_id"].eq(156)
            expand_mask = padding_mask
            expand_molecule_mask = molecule_mask
            expand_clean_mask = clean_mask
            expand_pos = pos
            local_attention_weight = None
            node_type_edge = batched_data["node_type_edge"]

        dist = delta_pos.norm(dim=-1)

        adj = adj.masked_fill(~molecule_mask.unsqueeze(-1), True)

        edge_feature = self.gbf(dist, node_type_edge.long())

        edge_bond_feature = self.mol_graph_2d_bond_feat(adj.long())
        edge_bond_feature = edge_bond_feature.masked_fill(
            ~expand_molecule_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )
        edge_bond_feature = edge_bond_feature.masked_fill(
            ~molecule_mask.unsqueeze(2).unsqueeze(-1), 0.0
        )

        if clean_mask is not None:
            adj = adj.masked_fill(clean_mask.unsqueeze(-1), True)
            edge_bond_feature = edge_bond_feature.masked_fill(
                expand_clean_mask.unsqueeze(1).unsqueeze(-1), 0.0
            )
            edge_bond_feature = edge_bond_feature.masked_fill(
                clean_mask.unsqueeze(2).unsqueeze(-1), 0.0
            )

        edge_feature = torch.cat([edge_feature, edge_bond_feature], dim=-1)

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

        # dist = dist.masked_fill(~adj, min_dtype)

        dist = torch.nn.functional.softmax(dist.float() * self.scaling, dim=-1)
        if local_attention_weight is not None:
            dist = dist * local_attention_weight
            edge_feature = edge_feature * local_attention_weight.unsqueeze(-1)

        pos_feature_emb = torch.matmul(dist, pos_emb).to(self.pos_emb.weight.dtype)
        pos_feature_emb = self.pos_feature_emb(pos_feature_emb)
        # pos_feature_emb = self.pos_feature_emb(pos_emb)

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
        time_step_1d: Optional[Tensor] = None,
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
        molecule_mask = (
            (token_id <= 129) & (token_id > 1) & (~is_periodic.unsqueeze(-1))
        )

        if aa_mask is not None:
            mask_token_type = token_id.masked_fill(
                aa_mask, 157
            )  # 157 is the mask token
        else:
            mask_token_type = token_id

        batched_data["masked_token_type"] = mask_token_type
        condition_embedding = self.embed(mask_token_type)

        pos_embedding, pos_attn_bias = self._pos_emb(
            batched_data["pos"],
            pbc_expand_batched["expand_pos"]
            if pbc_expand_batched is not None
            else None,
            batched_data["adj"],
            clean_mask,
            molecule_mask,
            padding_mask,
            batched_data,
            pbc_expand_batched=pbc_expand_batched,
        )

        if "init_pos" in batched_data and (batched_data["init_pos"] != 0.0).any():
            init_pos = batched_data["init_pos"].to(self.pos_emb.weight.dtype)
            init_pos_embedding = self.init_pos_emb(init_pos)
            # init_pos_embedding, init_pos_attn_bias = self._pos_emb(
            #     init_pos,
            #     pbc_expand_batched["init_expand_pos"]
            #     if pbc_expand_batched is not None
            #     else None,
            #     batched_data["adj"],
            #     clean_mask,
            #     molecule_mask,
            #     padding_mask,
            #     batched_data,
            #     pbc_expand_batched=pbc_expand_batched,
            # )

            init_pos_mask = (
                (init_pos != 0.0).any(dim=-1, keepdim=False).any(dim=-1, keepdim=False)
            )
            pos_embedding[init_pos_mask, :] += init_pos_embedding[init_pos_mask, :]

            # init_pos_attn_bias = init_pos_attn_bias.masked_fill(
            #     ~init_pos_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0
            # )
            # pos_attn_bias += init_pos_attn_bias

        if self.psm_config.diffusion_mode == "edm":
            noise_embed_edm = self.noise_cond_embed_edm(
                batched_data["c_noise"].flatten()
            ).to(condition_embedding.dtype)
            time_embed = noise_embed_edm.reshape(
                (pos_embedding.size(0), pos_embedding.size(1), -1)
            )
        elif time_step is not None:
            time_embed = self.time_step_encoder(time_step, clean_mask)

        if self.psm_config.use_2d_atom_features and "node_attr" in batched_data:
            atom_feature_embedding = self.atom_feature_embed(
                batched_data["node_attr"][:, :, 1:]
            ).sum(
                dim=-2
            )  # B x T x #ATOM_FEATURE x D -> # B x T x D
            atom_feature_embedding = atom_feature_embedding.masked_fill(
                ~molecule_mask.unsqueeze(-1), 0.0
            )

            # # time raito is 0 at time step == 0, time raito is 1 at time step >= 1e-3, linear increase between 0 and 1e-3
            if time_step is not None:
                time_ratio = torch.clamp(time_step / 0.001, 0.0, 1.0)
                pos_embedding += atom_feature_embedding * time_ratio.unsqueeze(-1)
            else:
                condition_embedding += atom_feature_embedding

        return (
            pos_embedding,
            padding_mask,
            time_embed,
            pos_attn_bias,
            condition_embedding + time_embed,
        )


class ProteaEmbedding(PSMMix3dDitEmbedding):
    """
    Class for the embedding layer in the PSM model.
    """

    def __init__(self, psm_config: PSMConfig, use_unified_batch_sampler: bool = False):
        super(ProteaEmbedding, self).__init__(psm_config, use_unified_batch_sampler)

        self.embed = nn.Embedding(160, psm_config.encoder_embed_dim // 2)

        self.pos_embedding_proj = nn.Linear(
            psm_config.num_3d_bias_kernel, psm_config.embedding_dim // 2
        )
        self.pos_feature_emb = nn.Linear(
            psm_config.num_3d_bias_kernel, psm_config.embedding_dim // 2, bias=False
        )
        self.atom_feature_embed = nn.Embedding(
            psm_config.num_atom_features, psm_config.encoder_embed_dim // 2
        )

        self.time_step_encoder = TimeStepEncoder(
            psm_config.num_timesteps,
            psm_config.embedding_dim // 2,
            psm_config.diffusion_time_step_encoder_type,
        )

    def forward(
        self,
        batched_data: Dict,
        time_step: Optional[Tensor] = None,
        time_step_1d: Optional[Tensor] = None,
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
        molecule_mask = (
            (token_id <= 129) & (token_id > 1) & (~is_periodic.unsqueeze(-1))
        )

        if aa_mask is not None:
            token_id.masked_fill(aa_mask, 157)  # 157 is the mask token
        else:
            pass

        condition_embedding = torch.matmul(
            batched_data["one_hot_token_id"], self.embed.weight
        )

        pos_embedding, pos_attn_bias = self._pos_emb(
            batched_data["pos"],
            pbc_expand_batched["expand_pos"]
            if pbc_expand_batched is not None
            else None,
            batched_data["adj"],
            clean_mask,
            molecule_mask,
            padding_mask,
            batched_data,
            pbc_expand_batched=pbc_expand_batched,
        )

        if self.psm_config.diffusion_mode == "edm":
            noise_embed_edm = self.noise_cond_embed_edm(
                batched_data["c_noise"].flatten()
            ).to(condition_embedding.dtype)
            time_embed = noise_embed_edm.reshape(
                (pos_embedding.size(0), pos_embedding.size(1), -1)
            )
        elif time_step is not None:
            time_embed = self.time_step_encoder(time_step, clean_mask)
            time_1d_embed = self.time_step_encoder(time_step_1d, ~aa_mask)

            time_embed = torch.cat([time_embed, time_1d_embed], dim=-1)

        if self.psm_config.use_2d_atom_features and "node_attr" in batched_data:
            atom_feature_embedding = self.atom_feature_embed(
                batched_data["node_attr"][:, :, 1:]
            ).sum(
                dim=-2
            )  # B x T x #ATOM_FEATURE x D -> # B x T x D
            atom_feature_embedding = atom_feature_embedding.masked_fill(
                ~molecule_mask.unsqueeze(-1), 0.0
            )

            # # time raito is 0 at time step == 0, time raito is 1 at time step >= 1e-3, linear increase between 0 and 1e-3
            if time_step is not None:
                time_ratio = torch.clamp(time_step / 0.001, 0.0, 1.0)
                condition_embedding += atom_feature_embedding * time_ratio.unsqueeze(-1)
            else:
                condition_embedding += atom_feature_embedding

        x = torch.cat([pos_embedding, condition_embedding], dim=-1)

        return (
            x,
            padding_mask,
            time_embed,
            pos_attn_bias,
            time_embed,
        )


class PSMLightEmbedding(PSMMix3dDitEmbedding):
    """
    Class for the embedding layer in the PSM model.
    """

    def __init__(self, psm_config: PSMConfig, use_unified_batch_sampler: bool = False):
        super(PSMLightEmbedding, self).__init__(psm_config, use_unified_batch_sampler)

        self.embed = nn.Embedding(160, psm_config.encoder_embed_dim)
        self.chain_id_proj = nn.Embedding(1000, psm_config.encoder_embed_dim)

        self.pos_emb = nn.Linear(3, psm_config.embedding_dim, bias=False)

        self.atom_feature_embed = nn.Embedding(
            psm_config.num_atom_features, psm_config.encoder_embed_dim
        )

        self.time_step_encoder = TimeStepEncoder(
            psm_config.num_timesteps,
            psm_config.embedding_dim,
            psm_config.diffusion_time_step_encoder_type,
        )

        self.mol_bond_emb = nn.Embedding(
            psm_config.num_edges, psm_config.num_3d_bias_kernel, padding_idx=0
        )

        self.bias_proj = NonLinear(
            psm_config.num_3d_bias_kernel, psm_config.num_attention_heads
        )

    @torch.compiler.disable(recursive=False)
    def _pos_emb(
        self,
        pos: Optional[torch.Tensor],
        expand_pos: torch.Tensor,
        adj: torch.Tensor,
        clean_mask: torch.Tensor,
        molecule_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        batched_data: torch.Tensor,
        pbc_expand_batched: Optional[Dict[str, Tensor]] = None,
    ):
        pos = pos.to(self.pos_emb.weight.dtype)  # / self.psm_config.diffusion_noise_std

        # inf_nan_mask = batched_data["protein_mask"].any(dim=-1)
        padding_mask_tmp = padding_mask | batched_data["token_id"].eq(156)

        if pbc_expand_batched is not None:
            expand_pos = expand_pos.to(
                self.pos_emb.weight.dtype
            )  # / self.psm_config.diffusion_noise_std
            expand_pos = torch.cat([pos, expand_pos], dim=1)
            expand_mask = torch.cat(
                [padding_mask_tmp, pbc_expand_batched["expand_mask"]], dim=-1
            )

            # expand_pos = self.center_pos(expand_pos, expand_mask)
            delta_pos = pos.unsqueeze(2) - expand_pos.unsqueeze(1)
            B, L, expand_L = delta_pos.size()[:3]
            adj = torch.ones(B, L, expand_L, device=adj.device, dtype=torch.bool)
            pbc_expand_batched["local_attention_weight"]
            node_type_edge = pbc_expand_batched["expand_node_type_edge"]
            expand_molecule_mask = torch.cat(
                [
                    molecule_mask,
                    torch.zeros_like(
                        pbc_expand_batched["expand_mask"], dtype=torch.bool
                    ),
                ],
                dim=-1,
            )
            expand_clean_mask = torch.cat(
                [
                    clean_mask,
                    torch.ones_like(
                        pbc_expand_batched["expand_mask"], dtype=torch.bool
                    ),
                ],
                dim=-1,
            )
        else:
            delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
            expand_mask = padding_mask_tmp
            expand_molecule_mask = molecule_mask
            expand_clean_mask = clean_mask
            expand_pos = pos
            node_type_edge = batched_data["node_type_edge"]

        dist = delta_pos.norm(dim=-1)

        adj = adj.masked_fill(~molecule_mask.unsqueeze(-1), True)

        edge_feature = self.gbf(dist, node_type_edge.long())

        edge_bond_feature = self.mol_graph_2d_bond_feat(adj.long())
        edge_bond_feature = edge_bond_feature.masked_fill(
            ~expand_molecule_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )
        edge_bond_feature = edge_bond_feature.masked_fill(
            ~molecule_mask.unsqueeze(2).unsqueeze(-1), 0.0
        )

        if clean_mask is not None:
            adj = adj.masked_fill(clean_mask.unsqueeze(-1), True)
            edge_bond_feature = edge_bond_feature.masked_fill(
                expand_clean_mask.unsqueeze(1).unsqueeze(-1), 0.0
            )
            edge_bond_feature = edge_bond_feature.masked_fill(
                clean_mask.unsqueeze(2).unsqueeze(-1), 0.0
            )

        edge_feature = torch.cat([edge_feature, edge_bond_feature], dim=-1)

        graph_attn_bias = self.gbf_proj(edge_feature)
        graph_attn_bias = graph_attn_bias.masked_fill(
            expand_mask.unsqueeze(1).unsqueeze(-1), float("-inf")
        )

        graph_attn_bias = graph_attn_bias.masked_fill(
            padding_mask_tmp.unsqueeze(-1).unsqueeze(-1), 0.0
        )
        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2)

        return graph_attn_bias

    @torch.compiler.disable(recursive=False)
    def _2dedge_emb(
        self,
        adj: torch.Tensor,
        molecule_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        batched_data: torch.Tensor,
    ):
        if molecule_mask.any() or batched_data["is_complex"].any():
            edge_bond_feature = self.mol_bond_emb(
                batched_data["node_type_edge"].squeeze(-1)
            )
            edge_bond_feature = edge_bond_feature.masked_fill(~adj.unsqueeze(-1), 0.0)

            edge_bond_feature = edge_bond_feature.masked_fill(
                ~molecule_mask.unsqueeze(1).unsqueeze(-1), 0.0
            )
            edge_bond_feature = edge_bond_feature.masked_fill(
                ~molecule_mask.unsqueeze(2).unsqueeze(-1), 0.0
            )

            graph_attn_bias = self.bias_proj(edge_bond_feature)
            graph_attn_bias = graph_attn_bias.masked_fill(
                ~molecule_mask.unsqueeze(1).unsqueeze(-1), 0.0
            )
            graph_attn_bias = graph_attn_bias.masked_fill(
                ~molecule_mask.unsqueeze(2).unsqueeze(-1), 0.0
            )

            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2)
        else:
            graph_attn_bias = None

        return graph_attn_bias

    def forward(
        self,
        batched_data: Dict,
        time_step: Optional[Tensor] = None,
        time_step_1d: Optional[Tensor] = None,
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
        chain_id = batched_data["chain_ids"]
        molecule_mask = (
            (token_id <= 129) & (token_id > 1) & (~is_periodic.unsqueeze(-1))
        )

        if aa_mask is not None:
            mask_token_type = token_id.masked_fill(
                aa_mask, 157
            )  # 157 is the mask token
        else:
            mask_token_type = token_id

        if "hot_token_id" not in batched_data:
            batched_data["masked_token_type"] = mask_token_type
            condition_embedding = self.embed(mask_token_type)
        else:
            condition_embedding = torch.matmul(
                batched_data["one_hot_token_id"], self.embed.weight
            )

        pos_embedding = self.pos_emb(
            batched_data["pos"].to(self.pos_emb.weight.dtype)
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # pos_attn_bias = self._pos_emb(
        #     batched_data["pos"],
        #     pbc_expand_batched["expand_pos"]
        #     if pbc_expand_batched is not None
        #     else None,
        #     batched_data["adj"],
        #     clean_mask,
        #     molecule_mask,
        #     padding_mask,
        #     batched_data,
        #     pbc_expand_batched=pbc_expand_batched,
        # )

        graph_attn_bias = self._2dedge_emb(
            batched_data["adj"],
            molecule_mask,
            padding_mask,
            batched_data,
        )

        if self.psm_config.diffusion_mode == "edm":
            noise_embed_edm = self.noise_cond_embed_edm(
                batched_data["c_noise"].flatten()
            ).to(condition_embedding.dtype)
            time_embed = noise_embed_edm.reshape(
                (pos_embedding.size(0), pos_embedding.size(1), -1)
            )
        elif time_step is not None:
            time_embed = self.time_step_encoder(time_step, clean_mask)

        # if time_step_1d is not None:
        #     time_1d_embed = self.time_step_encoder(time_step_1d, ~aa_mask)
        # else:
        #     time_1d_embed = self.time_step_encoder(torch.zeros_like(time_step), clean_mask)

        # time_embed = torch.cat([time_embed, time_1d_embed], dim=-1)

        if self.psm_config.use_2d_atom_features and "node_attr" in batched_data:
            atom_feature_embedding = self.atom_feature_embed(
                batched_data["node_attr"][:, :, 1:]
            ).sum(
                dim=-2
            )  # B x T x #ATOM_FEATURE x D -> # B x T x D
            atom_feature_embedding = atom_feature_embedding.masked_fill(
                ~molecule_mask.unsqueeze(-1), 0.0
            )

            # # time raito is 0 at time step == 0, time raito is 1 at time step >= 1e-3, linear increase between 0 and 1e-3
            # if time_step is not None:
            #     time_ratio = torch.clamp(time_step / 0.0001, 0.0, 1.0)
            #     condition_embedding += atom_feature_embedding * time_ratio.unsqueeze(-1)
            # else:
            condition_embedding += atom_feature_embedding

        # x = torch.cat([pos_embedding, condition_embedding], dim=-1)

        chain_embed = self.chain_id_proj(chain_id)
        condition_embedding += chain_embed

        # return (
        #     x,
        #     padding_mask,
        #     time_embed.to(x.dtype),
        #     graph_attn_bias,
        #     time_embed.to(x.dtype),
        # )

        return (
            pos_embedding,
            padding_mask,
            time_embed.to(pos_embedding.dtype),
            graph_attn_bias,
            condition_embedding + time_embed.to(pos_embedding.dtype),
        )

        # return (
        #     x + time_embed,
        #     padding_mask,
        #     time_embed,
        #     None,
        #     torch.cat([condition_embedding, atom_feature_embedding], dim=-1).to(x.dtype),
        # )


class PSMLightPEmbedding(PSMMix3dDitEmbedding):
    """
    Class for the embedding layer in the PSM model.
    """

    def __init__(self, psm_config: PSMConfig, use_unified_batch_sampler: bool = False):
        super(PSMLightPEmbedding, self).__init__(psm_config, use_unified_batch_sampler)

        self.embed = nn.Embedding(160, psm_config.encoder_embed_dim // 2)

        self.pos_emb = nn.Linear(3, psm_config.embedding_dim // 2, bias=False)

        self.atom_feature_embed = nn.Embedding(
            psm_config.num_atom_features, psm_config.encoder_embed_dim // 2
        )

        self.time_step_encoder = TimeStepEncoder(
            psm_config.num_timesteps,
            psm_config.embedding_dim,
            psm_config.diffusion_time_step_encoder_type,
        )

    @torch.compiler.disable(recursive=False)
    def _pos_emb(
        self,
        pos: Optional[torch.Tensor],
        expand_pos: torch.Tensor,
        adj: torch.Tensor,
        clean_mask: torch.Tensor,
        molecule_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        batched_data: torch.Tensor,
        pbc_expand_batched: Optional[Dict[str, Tensor]] = None,
    ):
        pos = pos.to(self.pos_emb.weight.dtype)  # / self.psm_config.diffusion_noise_std

        # inf_nan_mask = batched_data["protein_mask"].any(dim=-1)
        padding_mask_tmp = padding_mask | batched_data["token_id"].eq(156)

        if pbc_expand_batched is not None:
            expand_pos = expand_pos.to(
                self.pos_emb.weight.dtype
            )  # / self.psm_config.diffusion_noise_std
            expand_pos = torch.cat([pos, expand_pos], dim=1)
            expand_mask = torch.cat(
                [padding_mask_tmp, pbc_expand_batched["expand_mask"]], dim=-1
            )

            # expand_pos = self.center_pos(expand_pos, expand_mask)
            delta_pos = pos.unsqueeze(2) - expand_pos.unsqueeze(1)
            B, L, expand_L = delta_pos.size()[:3]
            adj = torch.ones(B, L, expand_L, device=adj.device, dtype=torch.bool)
            pbc_expand_batched["local_attention_weight"]
            node_type_edge = pbc_expand_batched["expand_node_type_edge"]
            expand_molecule_mask = torch.cat(
                [
                    molecule_mask,
                    torch.zeros_like(
                        pbc_expand_batched["expand_mask"], dtype=torch.bool
                    ),
                ],
                dim=-1,
            )
            expand_clean_mask = torch.cat(
                [
                    clean_mask,
                    torch.ones_like(
                        pbc_expand_batched["expand_mask"], dtype=torch.bool
                    ),
                ],
                dim=-1,
            )
        else:
            delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
            expand_mask = padding_mask_tmp
            expand_molecule_mask = molecule_mask
            expand_clean_mask = clean_mask
            expand_pos = pos
            node_type_edge = batched_data["node_type_edge"]

        dist = delta_pos.norm(dim=-1)

        adj = adj.masked_fill(~molecule_mask.unsqueeze(-1), True)

        edge_feature = self.gbf(dist, node_type_edge.long())

        edge_bond_feature = self.mol_graph_2d_bond_feat(adj.long())
        edge_bond_feature = edge_bond_feature.masked_fill(
            ~expand_molecule_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )
        edge_bond_feature = edge_bond_feature.masked_fill(
            ~molecule_mask.unsqueeze(2).unsqueeze(-1), 0.0
        )

        if clean_mask is not None:
            adj = adj.masked_fill(clean_mask.unsqueeze(-1), True)
            edge_bond_feature = edge_bond_feature.masked_fill(
                expand_clean_mask.unsqueeze(1).unsqueeze(-1), 0.0
            )
            edge_bond_feature = edge_bond_feature.masked_fill(
                clean_mask.unsqueeze(2).unsqueeze(-1), 0.0
            )

        edge_feature = torch.cat([edge_feature, edge_bond_feature], dim=-1)

        graph_attn_bias = self.gbf_proj(edge_feature)
        graph_attn_bias = graph_attn_bias.masked_fill(
            expand_mask.unsqueeze(1).unsqueeze(-1), float("-inf")
        )

        graph_attn_bias = graph_attn_bias.masked_fill(
            padding_mask_tmp.unsqueeze(-1).unsqueeze(-1), 0.0
        )
        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2)

        return graph_attn_bias

    def forward(
        self,
        batched_data: Dict,
        time_step: Optional[Tensor] = None,
        time_step_1d: Optional[Tensor] = None,
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
        molecule_mask = (
            (token_id <= 129) & (token_id > 1) & (~is_periodic.unsqueeze(-1))
        )

        if aa_mask is not None:
            mask_token_type = token_id.masked_fill(
                aa_mask, 157
            )  # 157 is the mask token
        else:
            mask_token_type = token_id

        if "hot_token_id" not in batched_data:
            batched_data["masked_token_type"] = mask_token_type
            condition_embedding = self.embed(mask_token_type)
        else:
            condition_embedding = torch.matmul(
                batched_data["one_hot_token_id"], self.embed.weight
            )

        pos_embedding = self.pos_emb(
            batched_data["pos"].to(self.pos_emb.weight.dtype)
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # pos_attn_bias = self._pos_emb(
        #     batched_data["pos"],
        #     pbc_expand_batched["expand_pos"]
        #     if pbc_expand_batched is not None
        #     else None,
        #     batched_data["adj"],
        #     clean_mask,
        #     molecule_mask,
        #     padding_mask,
        #     batched_data,
        #     pbc_expand_batched=pbc_expand_batched,
        # )

        if self.psm_config.diffusion_mode == "edm":
            noise_embed_edm = self.noise_cond_embed_edm(
                batched_data["c_noise"].flatten()
            ).to(condition_embedding.dtype)
            time_embed = noise_embed_edm.reshape(
                (pos_embedding.size(0), pos_embedding.size(1), -1)
            )
        elif time_step is not None:
            time_embed = self.time_step_encoder(time_step, clean_mask)

        # if time_step_1d is not None:
        #     time_1d_embed = self.time_step_encoder(time_step_1d, ~aa_mask)
        # else:
        #     time_1d_embed = self.time_step_encoder(torch.zeros_like(time_step), clean_mask)

        # time_embed = torch.cat([time_embed, time_1d_embed], dim=-1)

        if self.psm_config.use_2d_atom_features and "node_attr" in batched_data:
            atom_feature_embedding = self.atom_feature_embed(
                batched_data["node_attr"][:, :, 1:]
            ).sum(
                dim=-2
            )  # B x T x #ATOM_FEATURE x D -> # B x T x D
            atom_feature_embedding = atom_feature_embedding.masked_fill(
                ~molecule_mask.unsqueeze(-1), 0.0
            )

            # # time raito is 0 at time step == 0, time raito is 1 at time step >= 1e-3, linear increase between 0 and 1e-3
            # if time_step is not None:
            #     time_ratio = torch.clamp(time_step / 0.0001, 0.0, 1.0)
            #     condition_embedding += atom_feature_embedding * time_ratio.unsqueeze(-1)
            # else:
            condition_embedding += atom_feature_embedding

        x = torch.cat([pos_embedding, condition_embedding], dim=-1)

        return (
            x + time_embed.to(x.dtype),
            padding_mask,
            time_embed.to(x.dtype),
            None,
            x,
        )


class PSMSeqEmbedding(nn.Module):
    """
    Class for the embedding layer in the PSM model.
    """

    def __init__(self, psm_config: PSMConfig, use_unified_batch_sampler: bool = False):
        super().__init__()

        self.embed = nn.Embedding(160, psm_config.encoder_embed_dim)
        self.atom_feature_embed = nn.Embedding(
            psm_config.num_atom_features, psm_config.encoder_embed_dim
        )

        # maximum 300 chains
        self.chain_id_embed = nn.Embedding(300, psm_config.encoder_embed_dim)

        self.time_step_encoder = TimeStepEncoder(
            psm_config.num_timesteps,
            psm_config.embedding_dim,
            psm_config.diffusion_time_step_encoder_type,
        )

        if psm_config.diffusion_mode == "edm":
            if psm_config.noise_embedding == "positional":
                self.noise_cond_embed_edm = PositionalEmbeddingEDM(
                    num_channels=psm_config.embedding_dim,
                )
            elif psm_config.noise_embedding == "fourier":
                self.noise_cond_embed_edm = FourierEmbeddingAF3(
                    num_channels=psm_config.embedding_dim,
                )

        self.psm_config = psm_config

    def forward(
        self,
        batched_data: Dict,
        time_step: Optional[Tensor] = None,
        time_step_1d: Optional[Tensor] = None,
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
        chain_id = batched_data["chain_ids"]
        padding_mask = token_id.eq(0)  # B x T x 1
        is_periodic = batched_data["is_periodic"]
        molecule_mask = (
            (token_id <= 129) & (token_id > 1) & (~is_periodic.unsqueeze(-1))
        )

        if aa_mask is not None:
            mask_token_type = token_id.masked_fill(
                aa_mask, 157
            )  # 157 is the mask token
        else:
            mask_token_type = token_id

        if "hot_token_id" not in batched_data:
            batched_data["masked_token_type"] = mask_token_type
            x = self.embed(mask_token_type)
        else:
            x = torch.matmul(batched_data["one_hot_token_id"], self.embed.weight)

        if self.psm_config.diffusion_mode == "edm":
            noise_embed_edm = self.noise_cond_embed_edm(
                batched_data["c_noise"].to(x.dtype).flatten()
            ).to(x.dtype)
            time_embed = noise_embed_edm.reshape((x.size(0), x.size(1), -1))
        elif time_step is not None:
            time_embed = self.time_step_encoder(time_step, clean_mask)

        if self.psm_config.use_2d_atom_features and "node_attr" in batched_data:
            atom_feature_embedding = self.atom_feature_embed(
                batched_data["node_attr"][:, :, 1:]
            ).sum(
                dim=-2
            )  # B x T x #ATOM_FEATURE x D -> # B x T x D
            atom_feature_embedding = atom_feature_embedding.masked_fill(
                ~molecule_mask.unsqueeze(-1), 0.0
            )

            x += atom_feature_embedding

        chain_embed = self.chain_id_embed(chain_id)
        x = x + chain_embed

        return (
            x,
            padding_mask,
            time_embed.to(x.dtype),
            None,
            x,
        )


class PSMMixSeqEmbedding(PSMSeqEmbedding):
    """
    Class for the embedding layer in the PSM model.
    """

    def __init__(self, psm_config: PSMConfig, use_unified_batch_sampler: bool = False):
        super().__init__(psm_config, use_unified_batch_sampler)

        self.embed = nn.Embedding(160, psm_config.encoder_embed_dim)
        self.atom_feature_embed = nn.Embedding(
            psm_config.num_atom_features, psm_config.encoder_embed_dim
        )

        # maximum 1000 chains
        self.chain_id_proj = nn.Embedding(1000, psm_config.encoder_embed_dim)

        self.time_step_encoder = TimeStepEncoder(
            psm_config.num_timesteps,
            psm_config.embedding_dim,
            psm_config.diffusion_time_step_encoder_type,
        )

        if psm_config.diffusion_mode == "edm":
            if psm_config.noise_embedding == "positional":
                self.noise_cond_embed_edm = PositionalEmbeddingEDM(
                    num_channels=psm_config.embedding_dim,
                )
            elif psm_config.noise_embedding == "fourier":
                self.noise_cond_embed_edm = FourierEmbeddingAF3(
                    num_channels=psm_config.embedding_dim,
                )

        self.mol_bond_emb = nn.Embedding(
            psm_config.num_edges, psm_config.num_3d_bias_kernel, padding_idx=0
        )

        self.bias_proj = nn.Sequential(
            nn.Linear(
                psm_config.num_3d_bias_kernel,
                psm_config.num_3d_bias_kernel * 4,
                bias=False,
            ),
            nn.LayerNorm(psm_config.num_3d_bias_kernel * 4),  # , bias=False),
            nn.ReLU(),
            nn.Linear(
                psm_config.num_3d_bias_kernel * 4,
                psm_config.num_attention_heads,
                bias=False,
            ),
        )

        self.pair_proj = nn.Sequential(
            nn.LayerNorm(psm_config.embedding_dim),
            nn.Linear(
                psm_config.embedding_dim, psm_config.encoder_pair_embed_dim, bias=False
            ),
            nn.SiLU(),
            nn.Linear(
                psm_config.encoder_pair_embed_dim,
                psm_config.encoder_pair_embed_dim,
                bias=False,
            ),
        )

        self.psm_config = psm_config

    @torch.compiler.disable(recursive=False)
    def _2dedge_emb(
        self,
        adj: torch.Tensor,
        molecule_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        batched_data: torch.Tensor,
        pbc_expand_batched: Optional[Dict[str, Tensor]] = None,
    ):
        # if molecule_mask.any() or batched_data["is_complex"].any():
        if pbc_expand_batched is not None:
            # node_type_edge = pbc_expand_batched["expand_node_type_edge"]
            graph_attn_bias = None
        else:
            node_type_edge = batched_data["node_type_edge"]

            edge_bond_feature = self.mol_bond_emb(node_type_edge.squeeze(-1))

            edge_bond_feature = edge_bond_feature.masked_fill(~adj.unsqueeze(-1), 0.0)

            edge_bond_feature = edge_bond_feature.masked_fill(
                ~molecule_mask.unsqueeze(1).unsqueeze(-1), 0.0
            )
            edge_bond_feature = edge_bond_feature.masked_fill(
                ~molecule_mask.unsqueeze(2).unsqueeze(-1), 0.0
            )

            graph_attn_bias = self.bias_proj(edge_bond_feature)

            graph_attn_bias = graph_attn_bias.masked_fill(
                ~molecule_mask.unsqueeze(1).unsqueeze(-1), 0.0
            )
            graph_attn_bias = graph_attn_bias.masked_fill(
                ~molecule_mask.unsqueeze(2).unsqueeze(-1), 0.0
            )

            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2)
        # else:
        #     graph_attn_bias = None

        return graph_attn_bias

    def forward(
        self,
        batched_data: Dict,
        time_step: Optional[Tensor] = None,
        time_step_1d: Optional[Tensor] = None,
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
        chain_id = batched_data["chain_ids"]
        padding_mask = token_id.eq(0)  # B x T x 1
        is_periodic = batched_data["is_periodic"]
        molecule_mask = (
            (token_id <= 129) & (token_id > 1) & (~is_periodic.unsqueeze(-1))
        )

        if aa_mask is not None:
            mask_token_type = token_id.masked_fill(
                aa_mask, 157
            )  # 157 is the mask token
        else:
            mask_token_type = token_id

        if "hot_token_id" not in batched_data:
            batched_data["masked_token_type"] = mask_token_type
            x = self.embed(mask_token_type)
        else:
            x = torch.matmul(batched_data["one_hot_token_id"], self.embed.weight)

        is_ddpm_for_material_when_edm = (
            self.psm_config.diffusion_mode == "edm"
            and self.psm_config.use_ddpm_for_material
            and batched_data["is_periodic"].all()
        )

        if self.psm_config.diffusion_mode == "edm" and (
            not is_ddpm_for_material_when_edm
        ):
            noise_embed_edm = self.noise_cond_embed_edm(
                batched_data["c_noise"].to(x.dtype).flatten()
            ).to(x.dtype)
            time_embed = noise_embed_edm.reshape((x.size(0), x.size(1), -1))
        elif time_step is not None:
            time_embed = self.time_step_encoder(time_step, clean_mask)

        if self.psm_config.use_2d_atom_features and "node_attr" in batched_data:
            atom_feature_embedding = self.atom_feature_embed(
                batched_data["node_attr"][:, :, 1:]
            ).sum(
                dim=-2
            )  # B x T x #ATOM_FEATURE x D -> # B x T x D
            atom_feature_embedding = atom_feature_embedding.masked_fill(
                ~molecule_mask.unsqueeze(-1), 0.0
            )

            x += atom_feature_embedding

        chain_embed = self.chain_id_proj(chain_id)

        x = x + chain_embed

        graph_attn_bias = self._2dedge_emb(
            batched_data["adj"],
            molecule_mask,
            padding_mask,
            batched_data,
            pbc_expand_batched,
        )

        x_p_i = self.pair_proj(x)

        if pbc_expand_batched is not None:
            outcell_index = pbc_expand_batched["outcell_index"]
            _, _, embed_dim = x_p_i.size()

            outcell_index = outcell_index.unsqueeze(-1).expand(-1, -1, embed_dim)
            expand_x_p_i = torch.gather(x_p_i, dim=1, index=outcell_index)

            x_p_j = torch.cat([x_p_i, expand_x_p_i], dim=1)
        else:
            x_p_j = x_p_i

        x_pair = torch.einsum("blh,bkh->blkh", x_p_i, x_p_j)

        if graph_attn_bias is not None:
            x_pair = x_pair + graph_attn_bias.permute(0, 2, 3, 1).mean(
                dim=-1
            ).unsqueeze(-1)

        x_pair = x_pair.permute(1, 2, 0, 3)

        return (
            x,
            padding_mask,
            time_embed.to(x.dtype),
            graph_attn_bias,
            x,
            x_pair,
        )
