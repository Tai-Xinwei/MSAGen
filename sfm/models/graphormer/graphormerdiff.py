# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.layer_norm import LayerNorm
from sfm.modules.quant_noise import quant_noise

from .modules.graphormer_layers import RobertaClassificationHead
from .modules.graphormer_sentence_encoder_TMdiff import GraphormerSentenceEncoderDiff
from .modules.UnifiedDecoder import UnifiedDecoder

logger = logging.getLogger(__name__)


class GraphormerDiffModel(nn.Module):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(self, args):
        super().__init__()

        graphormer_config = GraphormerConfig(args)
        self.args = graphormer_config.args
        logger.info(self.args)

        self.sentence_encoder = GraphormerSentenceEncoderDiff(
            graphormer_config,
            transformer_m_pretrain=args.transformer_m_pretrain,
            mode_prob=args.mode_prob,
        )

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.sentence_projection_layer = None
        self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None
        self.proj_out = None
        self.args = args

        # Remove head is set to true during fine-tuning
        self.load_softmax = not args.ft  # getattr(args, "remove_head", False)
        print("if finetune:", args.ft)

        # decoder is not used in finetune downstream tasks, so we don't need to load it
        if self.load_softmax:
            self.decoder = UnifiedDecoder(
                args,
                num_pred_attn_layer=args.num_pred_attn_layer,
                embedding_dim=args.encoder_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.act_dropout,
                num_3d_bias_kernel=args.num_3d_bias_kernel,
                num_edges=args.num_edges,
                num_atoms=args.num_atoms,
            )

        # linear head for sentence prediction or ft
        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.pooler_activation = get_activation_fn(args.pooler_activation_fn)

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None

        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim, args.num_classes, bias=False
                )

            if args.sent_loss:
                self.sentence_projection_layer = nn.Linear(
                    args.encoder_embed_dim, self.sentence_out_dim, bias=False
                )
        else:
            if isinstance(args.num_classes, int):
                self.proj_out = nn.Linear(
                    args.encoder_embed_dim, args.num_classes, bias=True
                )
            else:
                raise NotImplementedError

        self.graph_embed_out = RobertaClassificationHead(
            args.encoder_embed_dim, args.encoder_embed_dim, 1, args.activation_fn
        )

    def forward(
        self,
        batched_data,
        perturb=None,
        segment_labels=None,
        masked_tokens=None,
        **unused,
    ):
        """
        Forward pass for Masked LM encoder. This first computes the token
        embedding using the token embedding matrix, position embeddings (if
        specified) and segment embeddings (if specified).

        Here we assume that the sentence representation corresponds to the
        output of the classification_token (see bert_task or cross_lingual_lm
        task for more details).
        Args:
            - src_tokens: B x T matrix representing sentences
            - segment_labels: B x T matrix representing segment label for tokens
        Returns:
            - a tuple of the following:
                - logits for predictions in format B x T x C to be used in
                  softmax afterwards
                - a dictionary of additional data, where 'pooled_output' contains
                  the representation for classification_token and 'inner_states'
                  is a list of internal model states used to compute the
                  predictions (similar in ELMO). 'sentence_logits'
                  is the prediction logit for NSP task and is only computed if
                  this is specified in the input arguments.
        """
        # transfer batch from tuple to dict
        # idxs, attn_bias, attn_edge_type, spatial_pos, in_degree, in_degree, x, edge_input, y, pos, node_type_edge, node_mask = batched_data

        # batched_data = {"x": x, "pos": pos, "node_type_edge": node_type_edge, "node_mask": node_mask, "attn_bias": attn_bias, "spatial_pos": spatial_pos, "edge_input": edge_input, "attn_edge_type": attn_edge_type}

        x, attn_bias, delta_pos, inner_states, padding_mask = self.sentence_encoder(
            batched_data,
            segment_labels=segment_labels,
            perturb=perturb,
        )

        node_output = self.decoder(batched_data, x, delta_pos, padding_mask)

        x = x.transpose(0, 1)

        y_pred = None
        if self.graph_embed_out is not None:
            y_pred = self.graph_embed_out(x)

        # project masked tokens only
        if masked_tokens is not None:
            x = x[masked_tokens, :]

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
            self.sentence_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.sentence_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)

        if self.lm_output_learned_bias is not None and self.load_softmax:
            x = x + self.lm_output_learned_bias

        # finetuning
        if self.proj_out is not None:
            x = self.proj_out(x)

        return x, node_output, y_pred

    def init_state_dict_weight(self, weight, bias):
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
