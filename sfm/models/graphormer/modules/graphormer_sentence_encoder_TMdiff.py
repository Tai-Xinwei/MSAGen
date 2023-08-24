# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import numpy as np
import torch

from .graphormer_layers_diff import (
    Graph3DBiasDiff,
    GraphAttnBiasDiff,
    GraphNodeFeatureDiff,
)
from .graphormer_sentence_encoder import GraphormerSentenceEncoder


class GraphormerSentenceEncoderDiff(GraphormerSentenceEncoder):
    def __init__(
        self,
        graphormer_config,
        transformer_m_pretrain: bool = False,
        mode_prob: str = "0.6,0.2,0.2",
        # num_pred_attn_layer: int = 4,
    ) -> None:
        super().__init__(
            graphormer_config,
            init_bias=False,
        )
        args = graphormer_config.args
        self.graph_node_feature = GraphNodeFeatureDiff(
            args,
            num_heads=graphormer_config.num_attention_heads,
            num_atoms=graphormer_config.num_atoms,
            num_in_degree=graphormer_config.num_in_degree,
            num_out_degree=graphormer_config.num_out_degree,
            hidden_dim=graphormer_config.embedding_dim,
            n_layers=graphormer_config.num_encoder_layers,
            no_2d=graphormer_config.no_2d,
            # add_3d=add_3d,
            # args=args,
        )

        self.graph_attn_bias = GraphAttnBiasDiff(
            args,
            num_heads=graphormer_config.num_attention_heads
            * (graphormer_config.num_encoder_layers + 1),
            num_atoms=graphormer_config.num_atoms,
            num_edges=graphormer_config.num_edges,
            num_spatial=graphormer_config.num_spatial,
            num_edge_dis=graphormer_config.num_edge_dis,
            edge_type=graphormer_config.edge_type,
            multi_hop_max_dist=graphormer_config.multi_hop_max_dist,
            hidden_dim=graphormer_config.num_attention_heads,
            n_layers=graphormer_config.num_encoder_layers,
            no_2d=graphormer_config.no_2d,
            # add_3d=add_3d,
            # args=args,
        )

        self.graph_3d_bias = (
            Graph3DBiasDiff(
                args,
                num_heads=graphormer_config.num_attention_heads
                * (graphormer_config.num_encoder_layers + 1),
                num_edges=graphormer_config.num_edges,
                n_layers=graphormer_config.num_encoder_layers,
                embed_dim=graphormer_config.embedding_dim,
                num_kernel=graphormer_config.num_3d_bias_kernel,
                no_share_rpe=False,
                # args=args,
            )
            if graphormer_config.add_3d
            else None
        )

        self.transformer_m_pretrain = transformer_m_pretrain
        self.mode_prob = None
        if transformer_m_pretrain:
            try:
                mode_prob = [float(item) for item in mode_prob.split(",")]
                assert len(mode_prob) == 3
                assert sum(mode_prob) == 1.0
            except:
                mode_prob = [0.2, 0.2, 0.6]
            self.mode_prob = mode_prob

        self.t_timesteps = args.t_timesteps
        assert args.ddpm_schedule in ["linear", "quadratic", "sigmoid", "cosine"]
        (
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
        ) = self._beta_schedule(
            args.t_timesteps,
            args.ddpm_beta_start,
            args.ddpm_beta_end,
            args.ddpm_schedule,
        )

    def _beta_schedule(self, t_timesteps, beta_start, beta_end, schedule_type="linear"):
        if schedule_type == "linear":
            beta_list = torch.linspace(beta_start, beta_end, t_timesteps)
        elif schedule_type == "quadratic":
            beta_list = (
                torch.linspace(beta_start**0.5, beta_end**0.5, t_timesteps) ** 2
            )
        elif schedule_type == "sigmoid":
            betas = torch.linspace(-6, 6, t_timesteps)
            beta_list = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        elif schedule_type == "cosine":
            s = 0.008
            steps = t_timesteps + 1
            x = torch.linspace(0, t_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / t_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            beta_list = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise NotImplementedError

        alphas = 1 - beta_list
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def _noise_sample(self, x_start, t):
        noise = torch.randn_like(x_start) * 1.0

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(
        self,
        batched_data,
        perturb=None,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention

        ori_pos = batched_data["pos"]
        node_mask = batched_data["node_mask"]
        time = torch.randint(
            0, self.t_timesteps, (ori_pos.shape[0],), device=ori_pos.device
        ).long()
        noisy_pos = (
            self._noise_sample(ori_pos, time)
            .masked_fill(~node_mask.bool(), 0.0)
            .to(ori_pos.dtype)
        )
        vis_pos = ori_pos.masked_fill(node_mask.bool(), 0.0).to(ori_pos.dtype)
        pos = noisy_pos + vis_pos
        if self.args.fp16:
            pos = pos.half()

        data_x = batched_data["x"]
        n_graph, n_node = data_x.size()[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)  # B x T x 1
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        mask_dict = {0: [1, 1], 1: [1, 0], 2: [0, 1]}
        mask_tm = (
            torch.zeros(
                n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
            ).float()
            - 1
        )

        mask_2d = None
        mask_3d = None
        if self.transformer_m_pretrain:
            mask_choice = np.random.choice(np.arange(3), n_graph, p=self.mode_prob)
            mask_tm = torch.tensor([mask_dict[i] for i in mask_choice]).to(
                batched_data["pos"]
            )
            mask_2d = mask_tm[:, 0]
            mask_3d = mask_tm[:, 1]
            (mask_tm == torch.tensor([1, 0]).unsqueeze(0).to(mask_tm)).all(dim=-1)

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.graph_node_feature(batched_data, time, mask_2d=mask_2d)

        if self.args.fp16:
            x = x.half()

        if perturb is not None:
            x[:, 1:, :] = x[:, 1:, :] + perturb

        # x: B x T x C
        attn_bias = self.graph_attn_bias(batched_data, mask_2d=mask_2d)

        # @ Roger added: 3D attn bias
        delta_pos = None
        if self.graph_3d_bias is not None and not (batched_data["pos"] == 0).all():
            attn_bias_3d, merged_edge_features, delta_pos = self.graph_3d_bias(
                batched_data, pos
            )
            if mask_3d is not None:
                merged_edge_features, delta_pos = (
                    merged_edge_features * mask_3d[:, None, None],
                    delta_pos * mask_3d[:, None, None, None],
                )
                attn_bias_3d = attn_bias_3d.masked_fill_(
                    (
                        (attn_bias_3d != float("-inf"))
                        * (1 - mask_3d[:, None, None, None])
                    ).bool(),
                    0.0,
                )
            attn_bias[:, :, 1:, 1:] = attn_bias[:, :, 1:, 1:] + attn_bias_3d
            x[:, 1:, :] = x[:, 1:, :] + merged_edge_features * 0.01

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        attn_bias = (
            attn_bias.contiguous()
            .view(n_graph, len(self.layers) + 1, -1, n_node + 1, n_node + 1)
            .contiguous()
        )
        for nl, layer in enumerate(self.layers):
            x, _ = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias[:, nl, :, :, :],
            )
            if not last_state_only:
                inner_states.append(x)

        return x, attn_bias, delta_pos, pos, inner_states, padding_mask
