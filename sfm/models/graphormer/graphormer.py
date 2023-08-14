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

from sfm.logging import logger
from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.layer_norm import LayerNorm
from sfm.modules.quant_noise import quant_noise
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model

from .modules.graphormer_sentence_encoder import GraphormerSentenceEncoder, NodeDecoder


class GraphormerModel(Model):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(self, args, loss_fn=None, data_mean=0.0, data_std=1.0, not_init=False):
        super().__init__()
        if not_init:
            return
        graphormer_config = GraphormerConfig(args)
        self.args = graphormer_config.args
        if args.rank == 0:
            logger.info(self.args)

        self.L1loss = loss_fn(reduction="mean", data_mean=data_mean, data_std=data_std)

        self.net = Graphormer(args, graphormer_config)
        self.load_pretrained_weights(args, checkpoint_path=args.loadcheck_path)

    def load_pretrained_weights(self, args, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.
        """
        if args.ifresume or args.ft or args.infer:
            checkpoints_state = torch.load(checkpoint_path, map_location="cpu")
            if "model" in checkpoints_state:
                checkpoints_state = checkpoints_state["model"]
            elif "module" in checkpoints_state:
                checkpoints_state = checkpoints_state["module"]

            IncompatibleKeys = self.net.load_state_dict(checkpoints_state, strict=False)
            IncompatibleKeys = IncompatibleKeys._asdict()

            missing_keys = []
            for keys in IncompatibleKeys["missing_keys"]:
                if keys.find("dummy") == -1:
                    missing_keys.append(keys)

            unexpected_keys = []
            for keys in IncompatibleKeys["unexpected_keys"]:
                if keys.find("dummy") == -1:
                    unexpected_keys.append(keys)

            if len(missing_keys) > 0:
                logger.info(
                    "Missing keys in {}: {}".format(
                        checkpoint_path,
                        missing_keys,
                    )
                )

            if len(unexpected_keys) > 0:
                logger.info(
                    "Unexpected keys {}: {}".format(
                        checkpoint_path,
                        unexpected_keys,
                    )
                )

    def max_positions(self):
        return self.net.max_positions

    def forward(self, batched_data, **kwargs):
        return self.net(batched_data, **kwargs)

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        logits = model_output[0]
        bs = logits.shape[0]
        loss = self.L1loss(batch_data, logits)
        return ModelOutput(loss=loss, log_output={}, num_examples=bs)

    def config_optimizer(self):
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            tuple[Optimizer, LRScheduler]:
        """
        pass


class Graphormer(nn.Module):
    """
    Encoder for Masked Language Modelling.
    """

    def __init__(self, args, graphormer_config):
        super().__init__()
        self.max_positions = args.max_positions

        self.sentence_encoder = GraphormerSentenceEncoder(graphormer_config)

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.sentence_projection_layer = None
        self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None
        self.proj_out = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not args.ft  # getattr(args, "remove_head", False)
        print("if finetune:", args.ft)
        self.decoder = NodeDecoder(
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
        # logger.info("encoder x: {}".format(x))

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
