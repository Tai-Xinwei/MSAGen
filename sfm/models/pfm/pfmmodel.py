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
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.layer_norm import LayerNorm
from sfm.modules.quant_noise import quant_noise
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model

from .modules.pfm_encoder import NodeDecoder, PFMEncoder
from .modules.UnifiedDecoder import UnifiedDecoder


class PFMModel(Model):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(self, args, loss_fn=None, data_mean=0.0, data_std=1.0, not_init=False):
        super().__init__()
        if not_init:
            return
        pfm_config = PFMConfig(args)
        self.args = pfm_config.args
        if args.rank == 0:
            logger.info(self.args)

        self.loss = loss_fn(args)

        self.net = PFM(args, pfm_config)

        self.load_pretrained_weights(args, checkpoint_path=args.loadcheck_path)

    def load_pretrained_weights(self, args, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.
        """
        if args.ft or args.infer:
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
        seq_aa = batch_data["x"]
        logits = model_output[0]
        node_output = model_output[1]
        mask_pos = model_output[2]
        mask_aa = model_output[3]

        bs = seq_aa.shape[0]
        output = self.loss(batch_data, logits, node_output, mask_pos, mask_aa)
        loss = output[0]
        if len(output) > 1:
            log_loss = output[1]
        return ModelOutput(loss=loss, log_output=log_loss, num_examples=bs)

    def config_optimizer(self):
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            tuple[Optimizer, LRScheduler]:
        """
        pass


class PFM(nn.Module):
    """
    Encoder for Masked Language Modelling.
    """

    def __init__(self, args, pfm_config):
        super().__init__()
        self.max_positions = args.max_positions

        self.sentence_encoder = PFMEncoder(pfm_config)

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
        # self.decoder = NodeDecoder(
        #     args.encoder_embed_dim, args.encoder_attention_heads, args=args
        # )
        # self.decoder = UnifiedDecoder(
        #     args,
        #     num_pred_attn_layer=args.num_pred_attn_layer,
        #     embedding_dim=args.encoder_embed_dim,
        #     num_attention_heads=args.encoder_attention_heads,
        #     ffn_embedding_dim=args.encoder_ffn_embed_dim,
        #     dropout=args.dropout,
        #     attention_dropout=args.attention_dropout,
        #     activation_dropout=args.act_dropout,
        #     num_3d_bias_kernel=args.num_3d_bias_kernel,
        #     num_edges=args.num_edges,
        #     num_atoms=args.num_atoms,
        # )

        self.lm_output_learned_bias = None

        self.fc_pmlm_q = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim, bias=False
        )
        self.fc_pmlm_k = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim, bias=False
        )

        if self.load_softmax:
            # self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

            if not self.share_input_output_embed:
                # self.embed_out = nn.Linear(
                #     args.encoder_embed_dim, args.num_residues, bias=False
                # )
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim,
                    args.num_residues * args.num_residues,
                    bias=False,
                )

            if args.sent_loss:
                self.sentence_projection_layer = nn.Linear(
                    args.encoder_embed_dim, self.sentence_out_dim, bias=False
                )
        else:
            if isinstance(args.num_residues, int):
                self.proj_out = nn.Linear(
                    args.encoder_embed_dim, args.num_residues, bias=True
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
        (
            x,
            _,
            _,
            pos,
            inner_states,
            padding_mask,
            mask_pos,
            mask_aa,
        ) = self.sentence_encoder(
            batched_data,
            segment_labels=segment_labels,
            perturb=perturb,
        )
        # logger.info("encoder x: {}".format(x))

        # inner_states, node_output, sentence_rep = self.decoder(
        #     x, attn_bias, delta_pos, inner_states
        # )

        # node_output = self.decoder(
        #     batched_data, x, pos, padding_mask, mask_pos=mask_pos
        # )

        diag_mask = None
        x = x.transpose(0, 1)  # [B, L, H]

        # q = self.fc_pmlm_q(x)
        # k = self.fc_pmlm_k(x)
        # x = torch.einsum("zic,zjc->zijc", q, k)  # [B, L, L, H]

        # # memory efficient implementation
        # # mask_aa is a boolean mask of shape [B, L, 1]

        B, _, H = x.shape
        masked_indices = torch.where(mask_aa.squeeze(-1).bool())
        x = x[masked_indices[0], masked_indices[1]]  # [num_masked, H]

        # Compute q and k only for the selected positions
        q_masked = self.fc_pmlm_q(x)  # [num_masked, H]
        k_masked = self.fc_pmlm_k(x)  # [num_masked, H]

        masked_per_batch = mask_aa.squeeze(-1).sum(dim=1)

        q_split = torch.split(q_masked, masked_per_batch.tolist())
        k_split = torch.split(k_masked, masked_per_batch.tolist())

        result_list = []
        mask_list = []
        for i in range(B):
            x_i = torch.einsum(
                "ih,jh->ijh", q_split[i], k_split[i]
            )  # [mask_i_len, mask_i_len, H]
            result_list.append(x_i.view(-1, H))
            diag_mask = torch.eye(x_i.size(0), dtype=torch.bool, device=x_i.device)
            mask_list.append(diag_mask.view(-1).bool())

        x = torch.cat(result_list, dim=0)
        diag_mask = torch.cat(mask_list, dim=0)

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
            self.sentence_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.sentence_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)  # [B, L, L, vocab^2]

        if self.lm_output_learned_bias is not None and self.load_softmax:
            x = x + self.lm_output_learned_bias

        # finetuning
        if self.proj_out is not None:
            x = self.proj_out(x)

        return (x, None, diag_mask, mask_aa)

    def init_state_dict_weight(self, weight, bias):
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
