# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.modules.FairseqDropout import FairseqDropout
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.layer_norm import LayerNorm
from sfm.modules.multihead_attention import MultiheadAttention
from sfm.modules.quant_noise import quant_noise as apply_quant_noise_
from sfm.utils.LayerDropModuleList import LayerDropModuleList

from .graphormer_layers import (
    Graph3DBias,
    GraphAttnBias,
    GraphNodeFeature,
    NodeTaskHead,
)
from .graphormer_sentence_encoder_layer import GraphormerSentenceEncoderLayer


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class GraphormerSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        graphormer_config,
        init_bias: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        # num_pred_attn_layer: int = 4,
    ) -> None:
        super().__init__()
        args = graphormer_config.args
        self.dropout_module = FairseqDropout(
            graphormer_config.dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = graphormer_config.layerdrop
        self.max_seq_len = graphormer_config.max_seq_len
        self.embedding_dim = graphormer_config.embedding_dim
        self.ffn_embedding_dim = graphormer_config.ffn_embedding_dim
        self.num_attention_heads = graphormer_config.num_attention_heads
        self.num_segments = graphormer_config.num_segments
        self.use_position_embeddings = graphormer_config.use_position_embeddings
        self.apply_bert_init = graphormer_config.apply_bert_init
        self.learned_pos_embedding = graphormer_config.learned_pos_embedding

        if init_bias:
            self.graph_node_feature = GraphNodeFeature(
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

            self.graph_attn_bias = GraphAttnBias(
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
                Graph3DBias(
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

        # self.node_proc = NodeTaskHead(embedding_dim, num_attention_heads)

        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        if graphormer_config.encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        droppath_probs = [
            x.item()
            for x in torch.linspace(
                0, graphormer_config.droppath_prob, graphormer_config.num_encoder_layers
            )
        ]

        for nl in range(graphormer_config.num_encoder_layers):
            self.layers.extend(
                [
                    self.build_transformer_sentence_encoder_layer(
                        embedding_dim=graphormer_config.embedding_dim,
                        ffn_embedding_dim=graphormer_config.ffn_embedding_dim,
                        num_attention_heads=graphormer_config.num_attention_heads,
                        dropout=self.dropout_module.p,
                        attention_dropout=graphormer_config.attention_dropout,
                        activation_dropout=graphormer_config.activation_dropout,
                        activation_fn=graphormer_config.activation_fn,
                        export=export,
                        q_noise=q_noise,
                        qn_block_size=qn_block_size,
                        sandwich_ln=graphormer_config.sandwich_ln,
                        droppath_prob=droppath_probs[nl],
                        nl=nl,
                        args=args,
                    )
                ]
            )

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

        # @ shengjie added: initialization from Foundation Transformer
        init_scale = math.sqrt(math.log(graphormer_config.num_encoder_layers))
        for name, p in self.named_parameters():
            if "fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name:
                p.data.mul_(init_scale)

        self.args = args

    def build_transformer_sentence_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        sandwich_ln,
        droppath_prob,
        nl,
        args,
    ):
        return GraphormerSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            sandwich_ln=sandwich_ln,
            droppath_prob=droppath_prob,
            nl=nl,
            args=args,
        )

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
        if not self.args.ft:
            ori_pos = batched_data["pos"]
            node_mask = batched_data["node_mask"]
            noise = (
                torch.randn(ori_pos.shape, device=ori_pos.device)
                * self.args.noise_scale
            )
            noise = noise.masked_fill_(~node_mask.bool(), 0.0)
            batched_data["pos"] = ori_pos + noise

        data_x = batched_data["x"]
        n_graph, n_node = data_x.size()[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)  # B x T x 1
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        mask_2d = mask_3d = None
        # if self.training:
        #     mask_choice = np.random.choice(np.arange(3), n_graph, p=[1.0 / 3, 1.0 / 3, 1.0 / 3])
        #     mask = torch.tensor([mask_dict[i] for i in mask_choice]).to(batched_data['pos'])
        #     mask_2d = mask[:, 0]
        #     mask_3d = mask[:, 1]

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.graph_node_feature(batched_data, mask_2d=mask_2d)

        if perturb is not None:
            x[:, 1:, :] = x[:, 1:, :] + perturb

        # x: B x T x C

        attn_bias = self.graph_attn_bias(batched_data, mask_2d=mask_2d)

        # @ Roger added: 3D attn bias
        delta_pos = None
        if self.graph_3d_bias is not None and not (batched_data["pos"] == 0).all():
            attn_bias_3d, merged_edge_features, delta_pos = self.graph_3d_bias(
                batched_data
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

        return x, attn_bias, delta_pos, inner_states, padding_mask


class GraphormerSentenceEncoderPP(GraphormerSentenceEncoder):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        graphormer_config,
        # num_pred_attn_layer: int = 4,
    ) -> None:
        # inheret from GraphormerSentenceEncoder
        super().__init__(graphormer_config)
        self.graphormer_config = graphormer_config

    @classmethod
    def config(cls):
        return cls.graphormer_config
        # return GraphormerConfig(
        #     hidden_size=cls.embedding_dim,
        #     intermediate_size=cls.ffn_embedding_dim,
        #     num_attention_heads=cls.num_attention_heads,
        #     hidden_act="relu",
        # )

    def forward(self, input_batchdata: Tuple):
        (
            input_ids,
            llm_mask,
            _,
            x,
            in_degree,
            out_degree,
            attn_bias,
            spatial_pos,
            edge_input,
            num_atoms,
        ) = input_batchdata

        # create dict for batched data
        batched_data = {}
        batched_data["attn_bias"] = attn_bias
        batched_data["spatial_pos"] = spatial_pos
        batched_data["in_degree"] = in_degree
        batched_data["out_degree"] = out_degree
        batched_data["x"] = x
        batched_data["edge_input"] = edge_input
        batched_data["attn_edge_type"] = None

        x, _, _, _, padding_mask = super().forward(batched_data)

        return (x, padding_mask, llm_mask, input_ids)


class NodeDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        num_attention_heads: int = 8,
        last_state_only: bool = True,
        args=None,
    ):
        super().__init__()
        if not args.ft:
            self.node_proc = NodeTaskHead(embedding_dim, num_attention_heads)
        self.args = args
        self.last_state_only = last_state_only

    def forward(self, x, attn_bias, delta_pos, inner_states):
        sentence_rep = x[0, :, :]

        node_output = None
        if delta_pos is not None and not self.args.ft:
            node_output = self.node_proc(
                x[1:, :, :], attn_bias[:, -1, :, 1:, 1:], delta_pos
            )

        if self.last_state_only:
            inner_states = [x]

        if not self.last_state_only:
            return torch.stack(inner_states), node_output, sentence_rep
        else:
            return inner_states, node_output, sentence_rep
