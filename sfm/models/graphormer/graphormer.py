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
from modules.get_activation_fn import get_activation_fn
from modules.graphormer_sentence_encoder import (
    GraphormerSentenceEncoder,
    Node_decoder,
    init_bert_params,
)
from modules.layer_norm import LayerNorm
from modules.quant_noise import quant_noise

logger = logging.getLogger(__name__)


class GraphormerModel(nn.Module):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_embed_dim = args.encoder_embed_dim

        # add architecture parameters
        # base_architecture(args)
        graphormer_base_architecture(args)

        if not hasattr(args, "max_positions"):
            try:
                args.max_positions = args.tokens_per_sample
            except:
                args.max_positions = args.max_nodes

        logger.info(args)

        self.encoder = GraphormerEncoder(args)

    def max_positions(self):
        return self.encoder.max_positions

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)


class GraphormerEncoder(nn.Module):
    """
    Encoder for Masked Language Modelling.
    """

    def __init__(self, args):
        super().__init__()
        self.max_positions = args.max_positions

        self.sentence_encoder = GraphormerSentenceEncoder(
            # < for graphormer
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            # >
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=self.max_positions,
            num_segments=args.num_segment,
            use_position_embeddings=not args.no_token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_bert_init=args.apply_bert_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
            sandwich_ln=args.sandwich_ln,
            droppath_prob=args.droppath_prob,
            add_3d=args.add_3d,
            num_3d_bias_kernel=args.num_3d_bias_kernel,
            no_2d=args.no_2d,
            args=args,
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
        self.decoder = Node_decoder(
            args.encoder_embed_dim, args.encoder_attention_heads, args=args
        )

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

        x, attn_bias, delta_pos, inner_states, _ = self.sentence_encoder(
            batched_data,
            segment_labels=segment_labels,
            perturb=perturb,
        )

        inner_states, node_output, sentence_rep = self.decoder(
            x, attn_bias, delta_pos, inner_states
        )

        x = inner_states[-1].transpose(0, 1)

        # FIXME: not compatible with batched_data

        # project masked tokens only
        if masked_tokens is not None:
            x = x[masked_tokens, :]

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        pooled_output = self.pooler_activation(self.masked_lm_pooler(sentence_rep))

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

        if self.sentence_projection_layer:
            self.sentence_projection_layer(pooled_output)

        return x, node_output
        # return x, node_output, {
        #     "inner_states": inner_states,
        #     "pooled_output": pooled_output,
        #     "sentence_logits": sentence_logits,
        # }

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        tmp_dict = {}
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if (
                    "embed_out.weight" in k
                    or "sentence_projection_layer.weight" in k
                    or "lm_output_learned_bias" in k
                    or "regression_lm_head_list" in k
                    or "regression_ln_list" in k
                    or "regression_embed_out_list" in k
                    or "classification_lm_head_list" in k
                    or "classification_ln_list" in k
                    or "classification_embed_out_list" in k
                ):
                    print("Removing", k, "(because load_softmax is False)")
                    tmp_dict[k] = state_dict[k]
                    del state_dict[k]
            proj_weight = torch.rand(self.proj_out.weight.shape)
            proj_bias = torch.rand(self.proj_out.bias.shape)

            # lm_head_transform_weight_weight = torch.rand(self.lm_head_transform_weight.weight.shape)
            # lm_head_transform_weight_bias = torch.rand(self.lm_head_transform_weight.bias.shape)
            lm_head_transform_weight_weight = tmp_dict.get(
                "encoder.regression_lm_head_list.0.weight", None
            )
            lm_head_transform_weight_bias = tmp_dict.get(
                "encoder.regression_lm_head_list.0.bias", None
            )
            ln_weight = tmp_dict.get("encoder.regression_ln_list.0.weight", None)
            ln_bias = tmp_dict.get("encoder.regression_ln_list.0.bias", None)

            self.init_state_dict_weight(proj_weight, proj_bias)
            # self.init_state_dict_weight(lm_head_transform_weight_weight, lm_head_transform_weight_bias)

            state_dict["encoder.proj_out.weight"] = state_dict.get(
                "encoder.proj_out.weight", proj_weight
            )
            state_dict["encoder.proj_out.bias"] = state_dict.get(
                "encoder.proj_out.bias", proj_bias
            )
            state_dict["encoder.lm_head_transform_weight.weight"] = state_dict.get(
                "encoder.lm_head_transform_weight.weight",
                lm_head_transform_weight_weight,
            )
            state_dict["encoder.lm_head_transform_weight.bias"] = state_dict.get(
                "encoder.lm_head_transform_weight.bias", lm_head_transform_weight_bias
            )
            state_dict["encoder.layer_norm.weight"] = state_dict.get(
                "encoder.layer_norm.weight", ln_weight
            )
            state_dict["encoder.layer_norm.bias"] = state_dict.get(
                "encoder.layer_norm.bias", ln_bias
            )
        return state_dict

    def init_state_dict_weight(self, weight, bias):
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)


def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.0)

    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.num_segment = getattr(args, "num_segment", 2)

    args.sentence_class_num = getattr(args, "sentence_class_num", 2)
    args.sent_loss = getattr(args, "sent_loss", False)

    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)

    args.sandwich_ln = getattr(args, "sandwich_ln", False)
    args.droppath_prob = getattr(args, "droppath_prob", 0.0)

    # add
    args.atom_loss_coeff = getattr(args, "atom_loss_coeff", 1.0)
    args.pos_loss_coeff = getattr(args, "pos_loss_coeff", 1.0)

    args.max_positions = getattr(args, "max_positions", 512)
    args.num_atoms = getattr(args, "num_atoms", 512 * 9)
    args.num_edges = getattr(args, "num_edges", 512 * 3)
    args.num_in_degree = getattr(args, "num_in_degree", 512)
    args.num_out_degree = getattr(args, "num_out_degree", 512)
    args.num_spatial = getattr(args, "num_spatial", 512)
    args.num_edge_dis = getattr(args, "num_edge_dis", 128)
    args.multi_hop_max_dist = getattr(args, "multi_hop_max_dist", 5)
    args.edge_type = getattr(args, "edge_type", "multi_hop")


def bert_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.num_segment = getattr(args, "num_segment", 2)

    args.encoder_layers = getattr(args, "encoder_layers", 12)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)

    args.sentence_class_num = getattr(args, "sentence_class_num", 2)
    args.sent_loss = getattr(args, "sent_loss", False)

    args.apply_bert_init = getattr(args, "apply_bert_init", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.sandwich_ln = getattr(args, "sandwich_ln", False)
    args.droppath_prob = getattr(args, "droppath_prob", 0.0)

    args.add_3d = getattr(args, "add_3d", False)
    args.num_3d_bias_kernel = getattr(args, "num_3d_bias_kernel", 128)
    args.no_2d = getattr(args, "no_2d", False)
    base_architecture(args)


def graphormer_base_architecture(args):
    # if args.pretrained_model_name == "pcqm4mv1_graphormer_base" or \
    #    args.pretrained_model_name == "pcqm4mv2_graphormer_base" or \
    #    args.pretrained_model_name == "pcqm4mv1_graphormer_base_for_molhiv":

    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.1)
    # else:
    #     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    #     args.encoder_layers = getattr(args, "encoder_layers", 12)
    #     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    #     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
    #     args.dropout = getattr(args, "dropout", 0.0)
    #     args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    #     args.act_dropout = getattr(args, "act_dropout", 0.1)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")

    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", True)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.pre_layernorm = getattr(args, "pre_layernorm", False)
    base_architecture(args)
