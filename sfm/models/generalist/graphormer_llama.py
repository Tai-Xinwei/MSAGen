# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch

from sfm.models.transformers.modeling_utils import PreTrainedModel
from sfm.models.transformers.models.llama.configuration_llama import LlamaConfig
from sfm.models.transformers.models.llama.modeling_llama import LlamaForCausalLM
from sfm.modules import GraphormerSentenceEncoder, init_bert_params
from sfm.modules.hybrid_emb import AdaptorConfig, Hybrid_emb
from sfm.sfmlogging.loggers import sfm_logger as logger


class GraphormerLlamaModel(torch.nn.Module):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(
        self,
        args,
        checkpointpth: Optional[list] = None,
        load_ckp: Optional[bool] = False,
    ):
        super().__init__()
        self.args = args

        # if specified then apply bert initialization on the model. We need
        # to explictly call this to make sure that the output embeddings
        # and projection layers are also correctly initialized
        if getattr(args, "apply_bert_init", False):
            self.apply(init_bert_params)
        self.encoder_embed_dim = args.encoder_embed_dim

        graphormer_base_architecture(args)

        if not hasattr(args, "max_positions"):
            try:
                args.max_positions = args.tokens_per_sample
            except:
                args.max_positions = args.max_nodes

        logger.info(args)

        self.max_positions = args.max_positions

        self.graphormer_encoder = GraphormerSentenceEncoder(
            # < for graphormer
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
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

        self.decoder = LlamaForCausalLM.from_pretrained(args.llm_model_name_or_path)
        Llama_config = LlamaConfig.from_pretrained(args.llm_model_name_or_path)

        adaptorconfig = self.init_adaptor_param(args, Llama_config)
        self.adaptor = Hybrid_emb(adaptorconfig)

    def load_state_dict(self, graphormer_ckppth, llama_ckppth=None, strict=True):
        pass

    def forward(
        self,
        batched_data: Dict,
        perturb=None,
        segment_labels=None,
    ) -> torch.Tensor:
        # generate mol_emb
        mol_emb, _, _, _, mol_padding_mask = self.graphormer_encoder(
            batched_data,
            segment_labels=segment_labels,
            perturb=perturb,
        )

        # generate text_emb
        text_embeds = self.decoder.get_input_embeddings()(batched_data["input_ids"])
        text_paddding_mask = batched_data["attention_mask"]

        # mix embeddings
        inputs_embeds, attention_mask, position_ids = self.adaptor(
            mol_emb,
            mol_padding_mask,
            text_embeds,
            text_paddding_mask,
            batched_data["input_ids"],
        )

        # decode
        logits = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]

        return logits

    def init_adaptor_param(self, args, Llama_config):
        config = AdaptorConfig(
            hidden_size=Llama_config.hidden_size,
            intermediate_size=Llama_config.intermediate_size,
            num_attention_heads=Llama_config.num_attention_heads,
            hidden_act=Llama_config.hidden_act,
            rms_norm_eps=Llama_config.rms_norm_eps,
            mfm_hidden_size=args.encoder_embed_dim,
            pool_mode=args.pool_mode,
            btn_adaptor=args.btn_adaptor,
            hidden_dropout_prob=args.dropout,
            embedding_length=args.embedding_length,
        )

        return config


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
