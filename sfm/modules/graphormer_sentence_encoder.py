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
from transformers.configuration_utils import PretrainedConfig
from utils.LayerDropModuleList import LayerDropModuleList

from .FairseqDropout import FairseqDropout
from .get_activation_fn import get_activation_fn
from .graphormer_layers import (
    Distance,
    EquivariantLayerNorm,
    EquivariantMultiHeadAttention,
    EquivariantVectorOutput,
    ExpNormalSmearing,
    Graph3DBias,
    GraphAttnBias,
    GraphNodeFeature,
    NodeTaskHead,
    RobertaClassificationHead,
)
from .graphormer_layers_pp import (
    Graph3DBiasPipe,
    GraphAttnBiasPipe,
    GraphNodeFeaturePipe,
    NodeTaskHeadPipe,
)
from .graphormer_sentence_encoder_layer import GraphormerSentenceEncoderLayer
from .layer_norm import LayerNorm
from .multihead_attention import MultiheadAttention
from .quant_noise import quant_noise as apply_quant_noise_


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


class GraphormerConfig(PretrainedConfig):
    model_type = "graphormer"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        molrep_dict_path=None,
        mfm_hidden_size=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.molrep_dict_path = molrep_dict_path
        self.mfm_hidden_size = mfm_hidden_size
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


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
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        sandwich_ln: bool = False,
        droppath_prob: float = 0.0,
        add_3d: bool = False,
        num_3d_bias_kernel: int = 128,
        no_2d: bool = False,
        args=None,
        # num_pred_attn_layer: int = 4,
    ) -> None:
        super().__init__()
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable

        self.graph_node_feature = GraphNodeFeature(
            args,
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            no_2d=no_2d,
            # add_3d=add_3d,
            # args=args,
        )

        self.graph_attn_bias = GraphAttnBias(
            args,
            num_heads=num_attention_heads * (num_encoder_layers + 1),
            num_atoms=num_atoms,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=num_attention_heads,
            n_layers=num_encoder_layers,
            no_2d=no_2d,
            # add_3d=add_3d,
            # args=args,
        )

        self.graph_3d_bias = (
            Graph3DBias(
                args,
                num_heads=num_attention_heads * (num_encoder_layers + 1),
                num_edges=num_edges,
                n_layers=num_encoder_layers,
                embed_dim=embedding_dim,
                num_kernel=num_3d_bias_kernel,
                no_share_rpe=False,
                # args=args,
            )
            if add_3d
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

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        droppath_probs = [
            x.item() for x in torch.linspace(0, droppath_prob, num_encoder_layers)
        ]

        for nl in range(num_encoder_layers):
            self.layers.extend(
                [
                    self.build_transformer_sentence_encoder_layer(
                        embedding_dim=self.embedding_dim,
                        ffn_embedding_dim=ffn_embedding_dim,
                        num_attention_heads=num_attention_heads,
                        dropout=self.dropout_module.p,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                        export=export,
                        q_noise=q_noise,
                        qn_block_size=qn_block_size,
                        sandwich_ln=sandwich_ln,
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
        init_scale = math.sqrt(math.log(num_encoder_layers))
        for name, p in self.named_parameters():
            if "fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name:
                p.data.mul_(init_scale)

        self.args = args

        self.dummy = nn.Linear(1, 1)

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

        x = x.transpose(0, 1)
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
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        sandwich_ln: bool = False,
        droppath_prob: float = 0.0,
        add_3d: bool = False,
        num_3d_bias_kernel: int = 128,
        no_2d: bool = False,
        args=None,
        # num_pred_attn_layer: int = 4,
    ) -> None:
        # inheret from GraphormerSentenceEncoder
        super().__init__(
            num_atoms,
            num_in_degree,
            num_out_degree,
            num_edges,
            num_spatial,
            num_edge_dis,
            edge_type,
            multi_hop_max_dist,
            num_encoder_layers,
            embedding_dim,
            ffn_embedding_dim,
            num_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            layerdrop,
            max_seq_len,
            num_segments,
            use_position_embeddings,
            offset_positions_by_padding,
            encoder_normalize_before,
            apply_bert_init,
            activation_fn,
            learned_pos_embedding,
            embed_scale,
            freeze_embeddings,
            n_trans_layers_to_freeze,
            export,
            traceable,
            q_noise,
            qn_block_size,
            sandwich_ln,
            droppath_prob,
            add_3d,
            num_3d_bias_kernel,
            no_2d,
            args,
        )

    @classmethod
    def config(cls):
        return GraphormerConfig(
            hidden_size=cls.embedding_dim,
            intermediate_size=cls.ffn_embedding_dim,
            num_attention_heads=cls.num_attention_heads,
            hidden_act="relu",
        )

    def forward(self, input_batchdata: Tuple):
        (
            idx,
            attn_bias,
            attn_edge_type,
            spatial_pos,
            in_degree,
            out_degree,
            x,
            edge_input,
            y,
            pos,
            node_type_edge,
            node_mask,
            input_ids,
            llm_mask,
        ) = input_batchdata
        # create dict for batched data

        batched_data = {}
        batched_data["idx"] = idx
        batched_data["attn_bias"] = attn_bias
        batched_data["attn_edge_type"] = attn_edge_type
        batched_data["spatial_pos"] = spatial_pos
        batched_data["in_degree"] = in_degree
        batched_data["out_degree"] = out_degree
        batched_data["x"] = x
        batched_data["edge_input"] = edge_input
        batched_data["y"] = y
        batched_data["pos"] = pos
        batched_data["node_type_edge"] = node_type_edge
        batched_data["node_mask"] = node_mask

        x, _, _, _, padding_mask = super().forward(batched_data)

        del batched_data

        return (x, padding_mask, llm_mask, input_ids)


class Node_decoder(nn.Module):
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
        x = x.transpose(0, 1)

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


class Pre_sentence_encoder_layer(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        sandwich_ln: bool = False,
        droppath_prob: float = 0.0,
        add_3d: bool = False,
        num_3d_bias_kernel: int = 128,
        no_2d: bool = False,
        args=None,
    ):
        super().__init__()

        self.embed_scale = embed_scale
        self.embedding_dim = embedding_dim

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        self.inner_states = None
        self.args = args
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        self.graph_node_feature = GraphNodeFeaturePipe(
            args,
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            no_2d=no_2d,
            # add_3d=add_3d,
            # args=args,
        )

        self.graph_attn_bias = GraphAttnBiasPipe(
            args,
            num_heads=num_attention_heads * (num_encoder_layers + 1),
            num_atoms=num_atoms,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=num_attention_heads,
            n_layers=num_encoder_layers,
            no_2d=no_2d,
            # add_3d=add_3d,
            # args=args,
        )

        self.graph_3d_bias = Graph3DBiasPipe(
            args,
            num_heads=num_attention_heads * (num_encoder_layers + 1),
            num_edges=num_edges,
            n_layers=num_encoder_layers,
            embed_dim=embedding_dim,
            num_kernel=num_3d_bias_kernel,
            no_share_rpe=False,
            # args=args,
        )  # if args.add_3d else None

    def forward(self, input_batchdata: tuple):
        # batched_data, perturb, segment_labels, last_state_only, positions, token_embeddings, token_embeddings, attn_mask = input
        (
            idx,
            attn_bias,
            attn_edge_type,
            spatial_pos,
            in_degree,
            output_degree,
            x_0,
            edge_input,
            y,
            pos,
            node_type_edge,
            node_mask,
            input_ids,
            llm_mask,
        ) = input_batchdata

        assert type(idx) == torch.Tensor
        assert type(attn_bias) == torch.Tensor
        assert type(attn_edge_type) == torch.Tensor
        assert type(spatial_pos) == torch.Tensor
        assert type(in_degree) == torch.Tensor
        assert type(output_degree) == torch.Tensor
        assert type(x_0) == torch.Tensor
        assert type(edge_input) == torch.Tensor
        assert type(y) == torch.Tensor
        assert type(pos) == torch.Tensor
        assert type(node_type_edge) == torch.Tensor
        assert type(node_mask) == torch.Tensor
        # print('input_ids', input_ids, llm_mask)

        if self.args.add_3d and not self.args.ft and not self.args.infer:
            noise = torch.randn(pos.shape, device=pos.device) * self.args.noise_scale
            noise = noise.masked_fill_(~node_mask.bool(), 0.0)
            pos = pos + noise

        token_embeddings = None
        perturb = None
        last_state_only = False

        n_graph, n_node = x_0.size()[:2]
        padding_mask = (x_0[:, :, 0]).eq(0)  # B x T x 1
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        mask_2d = None
        mask_3d = None

        if token_embeddings is not None:
            x = token_embeddings
        else:
            input_tuple = (x_0, in_degree, output_degree, node_mask, mask_2d)
            x = self.graph_node_feature(input_tuple)

        if perturb is not None:
            x[:, 1:, :] = x[:, 1:, :] + perturb

        # x: B x T x C
        input_tuple2 = (
            attn_bias,
            spatial_pos,
            x_0,
            edge_input,
            attn_edge_type,
            node_mask,
            mask_2d,
        )
        attn_bias = self.graph_attn_bias(input_tuple2)

        # @ Roger added: 3D attn bias
        delta_pos = None
        if self.graph_3d_bias is not None and not (pos == 0).all() and self.args.add_3d:
            input_tuple3 = (pos, x_0, node_type_edge, node_mask)
            attn_bias_3d, merged_edge_features, delta_pos = self.graph_3d_bias(
                input_tuple3
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

        if not last_state_only:
            self.inner_states = x[None, ...]

        attn_bias = (
            attn_bias.contiguous()
            .view(n_graph, self.args.encoder_layers + 1, -1, n_node + 1, n_node + 1)
            .contiguous()
        )

        if self.args.infer:
            return x, padding_mask, attn_bias, input_ids, llm_mask.bool()
        else:
            return x, padding_mask, attn_bias, delta_pos.bool(), pos

        # output, shape_tensor = self.tensors_encode(x, attn_bias, delta_pos)
        # return (output, padding_mask, shape_tensor)

    def tensors_encode(self, x, self_attn_bias, delta_pos):
        shape_tensor = torch.cat(
            [
                torch.tensor(x.shape),
                torch.tensor(self_attn_bias.shape),
                torch.tensor(delta_pos.shape),
            ],
            dim=-1,
        )
        output = torch.cat(
            [
                x.contiguous().view(-1),
                self_attn_bias.contiguous().view(-1),
                delta_pos.contiguous().view(-1),
            ],
            dim=-1,
        )

        return output, shape_tensor.to(x.device)  # , torch.tensor(shape_list)


class Post_sentence_encoder_layer(nn.Module):
    def __init__(
        self,
        args,
        embedding_dim,
        num_attention_heads,
        num_pred_attn_layer,
        num_3d_bias_kernel,
    ):
        super().__init__()

        self.last_state_only = False
        self.embedding_dim = embedding_dim
        # self.node_proc = NodeTaskHeadPipe(embedding_dim, num_attention_heads)

        # self.output_model_noise = EquivariantVectorOutput(embedding_dim, d_tilde=args.d_tilde)
        # self.out_norm_vec = EquivariantLayerNorm(embedding_dim)

        # self.distance = Distance()
        # self.distance_expansion = ExpNormalSmearing(num_rbf=num_3d_bias_kernel)

        # self.out_norm = nn.LayerNorm(embedding_dim)

        # self.attention_layers = nn.ModuleList()
        # for _ in range(num_pred_attn_layer):
        #     layer = EquivariantMultiHeadAttention(
        #         hidden_channels=embedding_dim,
        #         num_rbf=num_3d_bias_kernel,
        #         num_heads=num_attention_heads,
        #         d_tilde=args.d_tilde,
        #     )
        #     # layer = EquivariantMultiHeadAttention()
        #     self.attention_layers.append(layer)

        self.args = args

    def tensors_encode(self, x, node_output):
        shape_tensor = torch.cat(
            [torch.tensor(x.shape), torch.tensor(node_output.shape)], dim=-1
        )
        output = torch.cat(
            [x.contiguous().view(-1), node_output.contiguous().view(-1)], dim=-1
        )

        return output, shape_tensor.to(x.device)  # , torch.tensor(shape_list)

    def tensors_decode(self, output, shape_tensor):
        x_len = shape_tensor[0] * shape_tensor[1] * shape_tensor[2]
        self_attn_bias_len = (
            shape_tensor[3]
            * shape_tensor[4]
            * shape_tensor[5]
            * shape_tensor[6]
            * shape_tensor[7]
        )

        x = output[:x_len].view(shape_tensor[0], shape_tensor[1], shape_tensor[2])
        self_attn_bias = output[x_len : x_len + self_attn_bias_len].view(
            shape_tensor[3],
            shape_tensor[4],
            shape_tensor[5],
            shape_tensor[6],
            shape_tensor[7],
        )
        delta_pos = output[x_len + self_attn_bias_len :].view(
            shape_tensor[8], shape_tensor[9], shape_tensor[10], shape_tensor[11]
        )

        return x, self_attn_bias, delta_pos

    def forward(self, input_tuple: tuple):
        # print("post1")
        # x, attn_bias, padding_mask, delta_pos, inner_states = input
        if not self.args.infer:
            x, padding_mask, attn_bias, delta_pos, pos = input_tuple
        else:
            x, padding_mask, attn_bias, input_ids, llm_mask = input_tuple

        n_graph, _, _, _, n_node = attn_bias.size()
        n_node = n_node - 1
        # output, self_attn_padding_mask, shape_tensor = input_tuple
        # x, attn_bias, delta_pos = self.tensors_decode(output, shape_tensor)
        # x = x.to(torch.float16)

        assert type(x) == torch.Tensor
        assert type(attn_bias) == torch.Tensor
        assert type(padding_mask) == torch.Tensor
        # assert type(delta_pos) == torch.Tensor

        sentence_rep = x[0, :, :]

        # node_output = None
        # if delta_pos is not None:
        #     input_tuple = (x[1:, :, :], attn_bias[:, -1, :, 1:, 1:], delta_pos)
        #     node_output = self.node_proc(input_tuple)

        if self.args.infer or self.args.ft:
            # return (x, sentence_rep, padding_mask, input_ids)
            return (x, padding_mask, input_ids, llm_mask)

        node_output = None

        # if delta_pos is not None:
        #     # pos = batched_data['pos']
        #     is_not_pad = ~(padding_mask[:, 1:].reshape(-1))
        #     pos = pos.reshape(-1, 3)[is_not_pad]
        #     cnt = (~padding_mask[:, 1:]).sum(-1)
        #     cnt_cumsum = cnt.cumsum(0)
        #     batch = torch.zeros(pos.shape[0]).to(cnt)
        #     batch[cnt_cumsum[:-1]] = 1
        #     batch = batch.cumsum(0)

        #     edge_index, edge_weight, edge_vec = self.distance(pos.to(torch.float32), batch)
        #     assert (
        #             edge_vec is not None
        #     ), "Distance module did not return directional information"

        #     edge_attr = self.distance_expansion(edge_weight)
        #     edge_mask = edge_index[0] != edge_index[1]
        #     edge_vec[edge_mask] = edge_vec[edge_mask] / (torch.norm(edge_vec[edge_mask], dim=1).unsqueeze(1) + 1e-3)

        #     x_feat = x.contiguous().transpose(0, 1)[:, 1:, :].reshape(-1, self.embedding_dim)[is_not_pad]
        #     vec_feat = torch.zeros(x_feat.size(0), 3, x_feat.size(1)).to(x_feat)
        #     edge_weight, edge_vec, edge_attr = \
        #         edge_weight.to(x_feat), edge_vec.to(x_feat), edge_attr.to(x_feat)

        #     for attn in self.attention_layers:
        #         dx, dvec = attn(x_feat, vec_feat, edge_index, edge_weight, edge_attr, edge_vec)
        #         x_feat = x_feat + dx
        #         vec_feat = vec_feat + dvec
        #     x_feat = self.out_norm(x_feat)
        #     if self.out_norm_vec is not None:
        #         vec_feat = self.out_norm_vec(vec_feat)

        #     new_atom_output = self.output_model_noise(x_feat, vec_feat)

        #     node_output = torch.zeros(n_graph, n_node, 3).to(new_atom_output)
        #     total = 0
        #     for atom_idx in range(n_graph):
        #         cur_valid_atoms = int((~padding_mask[atom_idx, 1:]).sum())
        #         node_output[atom_idx, :cur_valid_atoms, :] = new_atom_output[total:total + cur_valid_atoms, :]
        #         total += cur_valid_atoms

        # if self.last_state_only:
        #     self.inner_states = x[None, ...]

        # x [T, Bs, emb], node_output [Bs, T-1, 3],
        return (x, node_output, sentence_rep)

        # value_tensor, shape_tensor = self.tensors_encode(x, node_output)
        # return (value_tensor, shape_tensor)
