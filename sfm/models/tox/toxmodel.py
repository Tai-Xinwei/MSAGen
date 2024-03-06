# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sfm.logging import logger
from sfm.models.tox.tox_config import TOXConfig
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.layer_norm import LayerNorm
from sfm.modules.quant_noise import quant_noise
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model

from .modules import torus as ts
from .modules.physics import MixtureGaussian, set_time_step
from .modules.timestep_encoder import DiffNoise
from .modules.torchMD import TorchMD_HEAD
from .modules.tox_encoder import NodeDecoder, TOXEncoder
from .modules.toxmixencoder import TOXMixEncoder
from .modules.UnifiedDecoder import UnifiedDecoder


class TOXModel(Model):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(
        self,
        args,
        loss_fn=None,
        data_mean=0.0,
        data_std=1.0,
        not_init=False,
        load_ckpt=False,
    ):
        super().__init__()
        if not_init:
            return
        pfm_config = TOXConfig(args)
        self.args = pfm_config.args
        if args.rank == 0:
            logger.info(self.args)

        self.loss = loss_fn(args)

        self.net = TOX(args, pfm_config)

        if load_ckpt:
            self.load_pretrained_weights(args, checkpoint_path=args.loadcheck_path)
        else:
            logger.info("No checkpoint is loaded")

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

            IncompatibleKeys = self.load_state_dict(checkpoints_state, strict=False)
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

            logger.info(f"checkpoint: {checkpoint_path} is loaded")
        else:
            logger.info("No checkpoint is loaded")

    def max_positions(self):
        return self.net.max_positions

    def forward(self, batched_data, **kwargs):
        return self.net(batched_data, **kwargs)

    def ft_forward(self, batched_data, **kwargs):
        return self.net.ft_forward(batched_data, **kwargs)

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        logits = model_output[0]
        node_output = model_output[1]
        angle_output = model_output[2]
        mask_pos = model_output[3]
        mask_aa = model_output[4]
        ang_score = model_output[5]
        ang_score_norm = model_output[6]
        padding_mask = model_output[7]
        pair_mask_aa = model_output[8]

        bs = logits.shape[0]
        output = self.loss(
            batch_data,
            logits,
            node_output,
            pair_mask_aa,
            angle_output,
            mask_pos,
            mask_aa,
            ang_score,
            ang_score_norm,
            padding_mask,
        )
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


class TOXPDEModel(TOXModel):
    """
    Support the PDE loss for the TOXModel
    """

    # FIXME: here we reuse ang_score for the score_based model output, i.e., ang_score = (angle_output -xt)/sigma_t^2 for VE

    def __init__(
        self,
        args,
        loss_fn=None,
        data_mean=0.0,
        data_std=1.0,
        not_init=False,
        load_ckpt=False,
    ):
        super().__init__(
            args,
            loss_fn,
            data_mean,
            data_std,
            not_init,
            load_ckpt,
        )
        self.mixture_gaussian = MixtureGaussian()

    # PDE loss related only using the batch data for the gaussian mixture center points
    def forward(self, batched_data, **kwargs):
        # Forward the score model [out:angle_output] for score matching
        (
            x,
            x_pair,
            angle_output,
            mask_pos,
            mask_aa,
            ang_score,
            ang_score_norm,
            padding_mask,
            pair_mask_aa,
        ) = self.net(batched_data, **kwargs)

        # Using a class object's forward function to generate the q_point, phi, nabla_phi, laplace_phi
        # genrate q_point, phi, nabla_phi, laplace_phi

        # Retrieve the time_pos from the TOX model instance
        time_pos = self.net.score_time
        # time_pos[:] = time_pos[0].item()
        time_pos = torch.zeros_like(time_pos)
        self.net.score_time = time_pos

        # Retrieve the angle from the batched_data
        ori_angle = batched_data["ang"]
        # Retrieve the noised angle from the TOX model instance
        noised_angle = self.net.noised_angle
        angle_mask = batched_data["ang_mask"].bool()
        ori_angle = ori_angle.masked_fill(~angle_mask, 100.0)
        noised_angle = noised_angle.masked_fill(~angle_mask, 100.0)

        # The input x is changed to the noised data
        # q_point, nabla_phi_term, laplace_phi_term = self.mixture_gaussian(noised_angle)
        q_point, q_point_0, nabla_phi_term, laplace_phi_term = self.mixture_gaussian(
            noised_angle, ori_angle
        )

        # setting delta_tq
        delta_tq = 0  # using self.score_time + delta_tq as the time_pos
        # Forward the score model [out:q_output] for PDE loss
        (
            _,
            _,
            q_output,
            _,
            _,
            q_score,
            q_score_norm,
            _,
            _,
        ) = self.net(
            batched_data, q=q_point, q_0=q_point_0, delta_tq=delta_tq, **kwargs
        )

        # TODO: make it more general
        # input time_pos in the setDeltaTq function to make time_pos - hm > 0
        hp, hm = set_time_step(time_pos)

        delta_tq = hp
        # Forward the score model [out:q_output] for PDE loss
        (
            _,
            _,
            q_output_ptq,
            _,
            _,
            q_score,
            q_score_norm,
            _,
        ) = self.net(
            batched_data, q=q_point, q_0=q_point_0, delta_tq=delta_tq, **kwargs
        )

        delta_tq = -hm
        # Forward the score model [out:q_output] for PDE loss
        (
            _,
            _,
            q_output_mtq,
            _,
            _,
            q_score,
            q_score_norm,
            _,
            _,
        ) = self.net(
            batched_data, q=q_point, q_0=q_point_0, delta_tq=delta_tq, **kwargs
        )

        return (
            x,
            x_pair,
            angle_output,
            mask_pos,
            mask_aa,
            ang_score,
            ang_score_norm,
            q_output,
            q_output_mtq,
            q_output_ptq,
            q_score,
            q_score_norm,
            padding_mask,
            time_pos,
            q_point,  # for PDE loss
            nabla_phi_term,
            laplace_phi_term,
            hp,
            hm,
        )

    # def ft_forward(self, batched_data, **kwargs):
    #     return self.net.ft_forward(batched_data, **kwargs)

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        logits = model_output[0]
        node_output = model_output[1]
        angle_output = model_output[2]
        mask_pos = model_output[3]
        mask_aa = model_output[4]
        ang_score = model_output[5]
        ang_score_norm = model_output[6]
        q_output = model_output[7]  # used for PDE loss
        q_output_mtq = model_output[8]  # used for PDE loss
        q_output_ptq = model_output[9]  # used for PDE loss
        q_score = model_output[10]
        q_score_norm = model_output[11]
        padding_mask = model_output[12]
        time_pos = model_output[13]  # used for PDE loss
        q_point = model_output[14]  # used for PDE loss
        nabla_phi_term = model_output[15]  # used for PDE loss
        laplace_phi_term = model_output[16]  # used for PDE loss
        hp = model_output[17]  # used for PDE loss
        hm = model_output[18]  # used for PDE loss

        bs = logits.shape[0]
        output = self.loss(
            batch_data,
            logits,
            node_output,
            angle_output,
            mask_pos,
            mask_aa,
            ang_score,
            ang_score_norm,
            q_output,
            q_output_mtq,
            q_output_ptq,
            q_score,
            q_score_norm,
            padding_mask,
            time_pos,
            q_point,
            nabla_phi_term,
            laplace_phi_term,
            hp,
            hm,
        )
        loss = output[0]
        if len(output) > 1:
            log_loss = output[1]
        return ModelOutput(loss=loss, log_output=log_loss, num_examples=bs)


class TOX(nn.Module):
    """
    Encoder for Masked Language Modelling.
    """

    def __init__(self, args, pfm_config):
        super().__init__()
        self.max_positions = args.max_positions

        self.sentence_encoder = TOXMixEncoder(pfm_config)

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.sentence_projection_layer = None
        self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None
        self.proj_out = None
        self.pfm_config = pfm_config
        self.args = args

        # Remove head is set to true during fine-tuning
        self.load_softmax = not args.ft  # getattr(args, "remove_head", False)
        print("if finetune:", args.ft)

        self.pi = torch.from_numpy(np.array(np.pi))

        # add score_time for PDE score loss
        self.score_time = None
        self.noised_angle = None
        self.ang_sigma = None
        self.ang_score = None
        self.angle_score_norm = None

        # self.uni_decoder = UnifiedDecoder(
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

        self.angle_decoder = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
            nn.GELU(),
            nn.Linear(args.encoder_embed_dim, 3),
        )

        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.pooler_activation = get_activation_fn(args.pooler_activation_fn)

        self.activation_fn = get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None

        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

        self.fc_pmlm_q = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim // 4, bias=False
        )
        self.fc_pmlm_k = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim // 4, bias=False
        )
        self.pair_layer_norm = nn.LayerNorm(args.encoder_embed_dim // 4)
        # self.dist_head = nn.Linear(args.encoder_embed_dim // 4, 1, bias=False)
        self.pair_head = nn.Linear(
            args.encoder_embed_dim // 4,
            args.num_residues * args.num_residues,
            bias=False,
        )

        if self.load_softmax:
            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim, args.num_residues, bias=False
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

        self.t_timesteps = args.t_timesteps

        self.diffnoise = DiffNoise(pfm_config)

        try:
            mode_prob = [float(item) for item in pfm_config.mode_prob.split(",")]
            assert len(mode_prob) == 4
            assert sum(mode_prob) == 1.0
        except:
            mode_prob = [0.0, 1.0, 0.0, 0.0]
        self.mode_prob = mode_prob
        logger.info(f"mode prob: {mode_prob}")

    def _set_noise(
        self,
        ori_pos,
        ori_angle,
        mask_pos,
        mask_angle,
        mode_mask=None,
        time_step=None,
        infer=False,
    ):
        if self.pfm_config.noise_mode == "mae":
            return ori_pos, None, None, None
        elif self.pfm_config.noise_mode == "const":
            noise = (
                torch.randn(ori_pos.shape, device=ori_pos.device)
                * self.args.noise_scale
            )
            noise = noise.masked_fill_(~mask_pos.bool(), 0.0)
            pos = ori_pos + noise
            return pos, None, None, None
        elif self.pfm_config.noise_mode == "diff":  # diff means diffusion
            # Here we modify the time_pos to be sampled from 0 to 1.
            if infer:  # give the same final time step (end time) to the batch
                if time_step is None:
                    time_step = self.t_timesteps - 1

                time_pos = torch.ones((ori_pos.shape[0],), device=ori_pos.device)

                time_ang = time_pos

                ori_pos = torch.zeros_like(ori_pos)
                ori_angle = torch.zeros_like(ori_angle)
            else:  # give random time point to each one in the batch
                time_pos = torch.rand((ori_pos.shape[0],), device=ori_pos.device)
                time_pos = torch.where((mode_mask == 2), 1, time_pos)

                time_ang = time_pos

            pos_scale_coeff = 1.0
            noisy_pos = (
                self.diffnoise._noise_sample(
                    ori_pos / pos_scale_coeff, time_pos, unit_noise_scale=1.0
                )
                * pos_scale_coeff
            )
            noisy_pos = noisy_pos.masked_fill(~mask_pos.bool(), 0.0).to(ori_pos.dtype)

            noisy_ang, ang_noise, ang_sigma = self.diffnoise._angle_noise_sample(
                ori_angle, time_ang
            )

            # FIXME: ang_score is hard to identify if it is correct or not
            ang_score = ts.score(
                ang_noise.float().cpu().numpy(), ang_sigma.float().cpu().numpy()
            )
            ang_score = torch.tensor(ang_score, device=noisy_ang.device)
            ang_score_norm = ts.score_norm(ang_sigma.float().cpu().numpy())
            ang_score_norm = torch.tensor(ang_score_norm, device=noisy_ang.device)
            noisy_ang = noisy_ang.masked_fill(~mask_angle.bool(), 0.0).to(
                ori_angle.dtype
            )

            vis_pos = ori_pos.masked_fill(mask_pos.bool(), 0.0)
            pos = noisy_pos + vis_pos

            vis_ang = ori_angle.masked_fill(mask_angle.bool(), 0.0)
            angle = noisy_ang + vis_ang

            return (
                pos,
                angle,
                time_pos,
                time_ang,
                ang_score,
                ang_score_norm,
                ang_noise,
                ang_sigma,
            )
        else:
            raise Exception(
                f"noise mode {self.pfm_config.noise_mode} not implemented, please choose from ['const', 'diff']"
            )

    def _set_mask(self, mask_aa, mask_pos, residue_seq):
        n_graph, n_node = residue_seq.size()[:2]

        # 1 is pad token, 2 is eos token
        padding_mask = (residue_seq[:, :]).eq(1)  # B x T x 1
        eos_mask = (residue_seq[:, :]).eq(2)

        mask_choice = np.random.choice(np.arange(4), n_graph, p=self.mode_prob)
        mask_choice = torch.tensor([i for i in mask_choice]).to(residue_seq.device)
        mask = (
            mask_choice.unsqueeze(1).unsqueeze(-1).repeat(1, n_node, 1)
        )  # [ngraph, nnode+1]

        # # 0:  mask_aa and mask_pos are the same, 2d3d -> 2d3d, both 2d3d masked with mask_ratio
        mask_pos = torch.where(mask == 0, mask_aa, mask_pos)

        # # 1 or 2:  mask_pos is full, 2d -> 2d3d with mask_ratio for 2d, 3d all noising
        mask_pos = torch.where((mask == 1) | (mask == 2), True, mask_pos)
        mask_aa = torch.where(mask == 1, False, mask_aa)

        # # 3:  mask_aa is full and no mask_pos, 3d -> 2d with 2d all mask
        mask_aa = torch.where(mask == 3, True, mask_aa)
        mask_pos = torch.where(mask == 3, False, mask_pos)

        # # cls token should not be masked
        mask_aa[:, 0, :] = False
        mask_pos[:, 0, :] = False
        mask_aa = mask_aa.masked_fill(padding_mask.bool().unsqueeze(-1), False)
        mask_aa = mask_aa.masked_fill(eos_mask.bool().unsqueeze(-1), False)
        mask_pos = mask_pos.masked_fill(padding_mask.bool().unsqueeze(-1), False)
        mask_pos = mask_pos.masked_fill(eos_mask.bool().unsqueeze(-1), False)

        mask_angle = mask_pos
        mask_pos = mask_pos.unsqueeze(-1)

        # mask_angle = torch.where(mask == 2, True, mask_angle)

        # mask_angle = mask_aa | mask_pos

        return mask_aa, mask_pos, padding_mask, eos_mask, mask_choice, mask_angle

    def _pos_map(self, x, mask_aa, residue_seq):
        B, L, H = x.shape
        masked_per_batch = []
        pair_mask_aa = torch.zeros((B, L, L, 1), device=x.device, dtype=torch.int8)
        result_list = []
        for i in range(B):
            masked_per_batch.append(mask_aa[i, :].sum().item())
            pair_mask_aa[i, :, :, :] = 1
            masked_indices = torch.where(mask_aa[i, :, 0].bool())
            masked_x = x[i, masked_indices[0], :]
            q_i = self.fc_pmlm_q(masked_x)
            k_i = self.fc_pmlm_k(masked_x)
            # diag_seq_list.append(residue_seq[i, masked_indices[0]])

            x_i = torch.einsum("ih,jh->ijh", q_i, k_i)
            H_i = x_i.shape[-1]
            result_list.append(x_i.view(-1, H_i))

        x = torch.cat(result_list, dim=0)
        x_pair = self.pair_head(x)
        # diag_seq = torch.cat(diag_seq_list, dim=0)

        return x_pair, pair_mask_aa

    def forward(
        self,
        batched_data,
        perturb=None,
        time_step=None,
        q=None,  # for computing the score model on the q
        q_0=None,
        delta_tq=None,  # for computing the score model on the q at time_pos + delta_tq
        mask_aa=None,
        mask_pos=None,
        mask_angle=None,
        padding_mask=None,
        mode_mask=None,
        time_pos=None,
        time_aa=None,
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
        residue_seq = batched_data["x"]
        if mask_aa is None:
            mask_aa = batched_data["masked_aa"]
        if mask_pos is None:
            mask_pos = batched_data["mask_pos"]

        ori_pos = batched_data["pos"]
        ori_angle = batched_data["ang"]

        angle_mask = batched_data["ang_mask"].bool()
        ori_angle = ori_angle.masked_fill(~angle_mask, 100.0).to(ori_pos.dtype)
        pos_mask = batched_data["pos_mask"].bool().unsqueeze(-1)
        ori_pos = ori_pos.masked_fill(~pos_mask, 0.0)

        (
            mask_aa,
            mask_pos,
            padding_mask,
            eos_mask,
            mode_mask,
            mask_angle,
        ) = self._set_mask(mask_aa, mask_pos, residue_seq)

        if q is None:
            (
                pos,
                angle,
                time_pos,
                time_aa,
                ang_score,
                ang_score_norm,
                ang_noise,
                ang_sigma,
            ) = self._set_noise(ori_pos, ori_angle, mask_pos, mask_angle, mode_mask)
            # angle = torch.remainder(angle, 2 * self.pi) - self.pi
            # TODO: 1000 should be given in the config file
            # give score_time for PDE score loss when q is not None
            self.score_time = time_pos
            self.noised_angle = angle
            self.ang_sigma = ang_sigma
            self.ang_score = ang_score
            self.angle_score_norm = ang_score_norm
        else:
            # actually we do not need q_score and q_score_norm
            angle = q
            angle_mask = batched_data["ang_mask"].bool()
            angle = angle.masked_fill(~angle_mask, 100.0).to(ori_pos.dtype)
            # angle = torch.remainder(angle, 2 * self.pi) - self.pi

            # if delta_tq is not None, means we are doing s(q, t + delta_tq)
            # if delta_tq is given, then time_pos will be updated to the real time_pos= time_pos + delta_tq
            if delta_tq is not None:
                time_pos = self.score_time + delta_tq

            pos = None
            ang_score, ang_score_norm = None, None

        (
            x,
            _,
            _,
            pos,
            inner_states,
            padding_mask,
            _,
            mask_aa,
            time_pos,
            time_aa,
        ) = self.sentence_encoder(
            batched_data,
            pos=pos,
            angle=angle,
            mask_aa=mask_aa,
            mask_pos=mask_pos,
            mask_angle=mask_angle,
            time_pos=time_pos,
            time_aa=time_aa,
            mode_mask=mode_mask,
            padding_mask=padding_mask,
            segment_labels=segment_labels,
            perturb=perturb,
        )

        padding_mask = padding_mask | eos_mask
        padding_mask[:, 0] = True

        # dist_mask = padding_mask.unsqueeze(-1) | padding_mask.unsqueeze(-2)

        # node_output, _, x_dec = self.uni_decoder(
        #     batched_data,
        #     x,
        #     pos,
        #     angle,
        #     padding_mask,
        #     mask_aa=mask_aa,
        #     mask_pos=mask_pos,
        #     time_pos=time_pos,
        # )

        x = x.transpose(0, 1)
        x = self.layer_norm(x)

        angle_output = self.angle_decoder(x)

        # apply ∆τ to ˆ C; predict δτ = sθ,G( ˆ C, t); update θ ← θ − α∇θ‖δτ − ∇∆τ pt|0(∆τ | 0)‖2; given in Algorithm 2 of
        # Training procedure of Torsional Diffusion for Molecular Conformer Generation
        # Hence here, angle_output gives the delta tau, which is the angle difference between the predicted and the ground truth
        # FIXME: check if it is okay to use the angle_output as score approximation directly, Done. VE is okay, but VP is not okay. see (31, 33) in the paper
        # SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS

        # TODO: check if it is okay to mask the paddings for q
        angle_output = angle_output.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if q is None:
            x_pair, pair_mask_aa = self._pos_map(x, mask_aa, residue_seq)
        else:
            x_pair = None
            pair_mask_aa = None

        # project masked tokens only
        if masked_tokens is not None:
            x = x[masked_tokens, :]

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

        # if q is None:
        #     pass
        # elif q_0 is not None:
        #     angle_output = (angle_output - q) #/ self.ang_sigma**2
        # else:
        #     raise NotImplementedError

        return (
            x,
            x_pair,
            angle_output,  # if q is not None, this is q_output
            mask_pos,
            mask_aa,
            ang_score,
            ang_score_norm,
            padding_mask,
            pair_mask_aa,
        )
        # return (x, node_output, angle_output, mask_pos, mask_aa, angle, pos, padding_mask)

    @torch.no_grad()
    def sample(
        self,
        batched_data,
        perturb=None,
        time_step=None,
        mask_aa=None,
        mask_pos=None,
        mask_angle=None,
        padding_mask=None,
        mode_mask=None,
        time_pos=None,
        time_aa=None,
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
        residue_seq = batched_data["x"]
        ori_pos = batched_data["pos"]
        ori_angle = batched_data["ang"]

        # angle_mask = ori_angle == float("inf")
        # ori_angle = ori_angle.masked_fill(angle_mask, 0.0).to(ori_pos.dtype)

        pos_mask = ori_pos == float("inf")
        ori_pos = ori_pos.masked_fill(pos_mask, 0.0)

        pos = ori_pos
        angle = ori_angle
        angle = torch.remainder(angle, 2 * self.pi) - self.pi
        # ang_score, ang_score_norm = None, None

        time_pos = (
            torch.ones((ori_pos.shape[0],), device=ori_pos.device, dtype=torch.long)
            * time_step
        )
        (
            torch.ones((ori_angle.shape[0],), device=ori_pos.device, dtype=torch.long)
            * time_step
        )

        (
            x,
            _,
            _,
            pos,
            inner_states,
            padding_mask,
            _,
            mask_aa,
            time_pos,
            time_aa,
        ) = self.sentence_encoder(
            batched_data,
            pos=pos,
            angle=angle,
            mask_aa=mask_aa,
            mask_pos=mask_pos,
            mask_angle=mask_angle,
            time_pos=time_pos,
            time_aa=time_aa,
            mode_mask=mode_mask,
            padding_mask=padding_mask,
            segment_labels=segment_labels,
            perturb=perturb,
        )

        eos_mask = (residue_seq[:, :]).eq(2)
        padding_mask = padding_mask | eos_mask
        padding_mask[:, 0] = True

        node_output, _, x_dec = self.uni_decoder(
            batched_data,
            x,
            pos,
            angle,
            padding_mask,
            mask_aa=mask_aa,
            mask_pos=mask_pos,
            time_pos=time_pos,
        )

        x = inner_states[-1].transpose(0, 1)

        angle_output = self.angle_decoder(x)
        angle_output = angle_output.masked_fill(padding_mask.unsqueeze(-1), 0.0)

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

        # return (x, node_output, angle_output, mask_pos, mask_aa, ang_score, ang_score_norm)
        return (x, node_output, angle_output, mask_pos, mask_aa, angle, pos)

    def ft_forward(
        self,
        batched_data,
        mode="T_noise",
        perturb=None,
        time_step=None,
        mask_aa=None,
        mask_pos=None,
        mask_angle=None,
        padding_mask=None,
        mode_mask=None,
        time_pos=None,
        time_aa=None,
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
        with torch.no_grad():
            residue_seq = batched_data["x"]
            B, L = residue_seq.shape
            if "pos" in batched_data:
                pos = batched_data["pos"]
            else:
                pos = None

            if "ang" in batched_data:
                angle = batched_data["ang"]
            else:
                angle = None

            padding_mask = (residue_seq[:, :]).eq(1)  # B x T x 1
            eos_mask = (residue_seq[:, :]).eq(2)
            angle_mask = batched_data["ang_mask"].bool()

        if mode == "T_noise":
            mask_pos = torch.ones_like(residue_seq).bool().unsqueeze(-1)
            mask_aa = torch.zeros_like(residue_seq).bool().unsqueeze(-1)

            angle = torch.zeros((B, L, 9), device=residue_seq.device)
            angle = torch.rand_like(angle) * torch.pi

            time_pos = torch.ones(B, device=residue_seq.device, dtype=torch.long)
        elif mode == "Diff_noise":
            mask_pos = torch.ones_like(residue_seq).bool().unsqueeze(-1)
            mask_aa = torch.zeros_like(residue_seq).bool().unsqueeze(-1)

            (
                mask_aa,
                mask_pos,
                padding_mask,
                eos_mask,
                mode_mask,
                mask_angle,
            ) = self._set_mask(mask_aa, mask_pos, residue_seq)
            (
                pos,
                angle,
                time_pos,
                time_aa,
                ang_score,
                ang_score_norm,
                ang_noise,
                ang_sigma,
            ) = self._set_noise(pos, angle, mask_pos, mask_angle, mode_mask)

        elif mode == "ori_angle":
            mask_pos = torch.zeros_like(residue_seq).bool().unsqueeze(-1)
            mask_aa = torch.zeros_like(residue_seq).bool().unsqueeze(-1)

            time_pos = torch.zeros(B, device=residue_seq.device, dtype=torch.long)
        else:
            assert f"mode {mode} not implemented, please choose from ['T_noise', 'Diff_noise', 'ori_angle']"

        angle = angle.masked_fill(~angle_mask, 100.0).to(residue_seq.dtype)

        # angle = torch.remainder(angle, 2 * torch.pi) - torch.pi

        (
            x,
            _,
            _,
            pos,
            inner_states,
            padding_mask,
            _,
            mask_aa,
            time_pos,
            time_aa,
        ) = self.sentence_encoder(
            batched_data,
            pos=pos,
            angle=angle,
            mask_aa=mask_aa,
            mask_pos=mask_pos,
            mask_angle=mask_angle,
            time_pos=time_pos,
            time_aa=time_aa,
            mode_mask=mode_mask,
            padding_mask=padding_mask,
            segment_labels=segment_labels,
            perturb=perturb,
        )

        padding_mask = padding_mask | eos_mask
        padding_mask[:, 0] = True

        x = inner_states[-1].transpose(0, 1)

        return x

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
