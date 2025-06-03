# -*- coding: utf-8 -*-
# Copyright (c) Mircrosoft.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sfm.data.psm_data.utils import MSAVOCAB, VOCAB, plot_probability_heatmaps
from sfm.logging import logger
from sfm.models.psm.equivariant.e2former import E2former
from sfm.models.psm.equivariant.equiformer.graph_attention_transformer import Equiformer
from sfm.models.psm.equivariant.equiformer_series import Equiformerv2SO2
from sfm.models.psm.equivariant.equivariant import EquivariantDecoder
from sfm.models.psm.equivariant.geomformer import EquivariantVectorOutput
from sfm.models.psm.equivariant.nodetaskhead import (
    AADiffusionModule,
    AAVectorProjOutput,
    ConditionVectorGatedOutput,
    DiffusionModule,
    DiffusionModule2,
    DiffusionModule3,
    ForceGatedOutput,
    ForceVecOutput,
    InvariantDiffusionModule,
    MSADiffusionModule,
    NodeTaskHead,
    ScalarGatedOutput,
    VectorGatedOutput,
    VectorOutput,
    VectorProjOutput,
)
from sfm.models.psm.equivariant.vectorVT import VectorVanillaTransformer
from sfm.models.psm.invariant.dit_encoder import PSMDiTEncoder
from sfm.models.psm.invariant.ditp_encoder import PSMPDiTPairEncoder
from sfm.models.psm.invariant.invariant_encoder import PSMEncoder
from sfm.models.psm.invariant.plain_encoder import (
    MSAGenEncoder,
    PSMPairPlainEncoder,
    PSMPlainEncoder,
)
from sfm.models.psm.modules.diffusion import DiffNoise, Diffsuion_LM, TimeStepSampler
from sfm.models.psm.modules.embedding import PSMMixEmbedding
from sfm.models.psm.modules.mixembedding import (
    MSAGenSeqEmbedding,
    ProteaEmbedding,
    PSMLightEmbedding,
    PSMLightPEmbedding,
    PSMMix3dDitEmbedding,
    PSMMix3dEmbedding,
    PSMMixSeqEmbedding,
    PSMSeqEmbedding,
)
from sfm.models.psm.modules.mixembedding_equiv import PSMMix3DEquivEmbedding
from sfm.models.psm.modules.pbc import CellExpander
from sfm.models.psm.psm_config import ForceHeadType, GaussianFeatureNodeType, PSMConfig
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model

from .modules.autograd import GradientHead
from .modules.confidence_model import lddt
from .modules.dataaug import uniform_random_rotation
from .modules.diffusion import DIFFUSION_PROCESS_REGISTER
from .modules.sampled_structure_converter import SampledStructureConverter


class MSAGenModel(Model):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(
        self,
        args,
        loss_fn=None,
        not_init=False,
        psm_finetune_head: nn.Module = None,
        molecule_energy_per_atom_std=1.0,
        periodic_energy_per_atom_std=1.0,
        molecule_force_std=1.0,
        periodic_force_std=1.0,
        periodic_stress_mean=0.0,
        periodic_stress_std=1.0,
        reload_checkpoint=True,
    ):
        """
        Initialize the PSMModel class.

        Args:
            args: Command line arguments.
            loss_fn: The loss function to use.
            data_mean: The mean of the label. For label normalization.
            data_std: The standard deviation of the label. For label normalization.
            not_init: If True, the model will not be initialized. Default is False.
            psm_finetune_head: head used to finetune psm
        """

        super().__init__()
        if not_init:
            return
        self.psm_config = PSMConfig(args)
        self.cut_off = self.psm_config.cutoff
        self.args = self.psm_config.args
        if args.rank == 0:
            logger.info(self.args)

        self.net = MSAGen(
            args,
            self.psm_config,
        )
        self.psm_finetune_head = psm_finetune_head
        if self.psm_config.diffusion_mode == "epsilon":
            self.noise_loss = nn.L1Loss(reduction="mean")

        elif self.psm_config.diffusion_mode in ["diff-lm", "OADM"]:
            self.noise_loss = nn.CrossEntropyLoss(reduction="none")
            self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        self.diffnoise = DiffNoise(self.psm_config)

        self.diffusion_process = DIFFUSION_PROCESS_REGISTER[
            self.psm_config.diffusion_sampling
        ](self.diffnoise.alphas_cumprod, self.psm_config)
        self.time_step_sampler = TimeStepSampler(self.psm_config.num_timesteps)
        if reload_checkpoint:
            self.checkpoint_loaded = self.reload_checkpoint()
        self.aa_mlm_loss = nn.CrossEntropyLoss(reduction="mean")
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        # self.loss_fn = loss_fn(args)

    def reload_checkpoint(self):
        if self.psm_config.psm_finetune_mode or self.psm_config.psm_validation_mode:
            if os.path.exists(self.args.loadcheck_path):
                self.load_pretrained_weights(
                    self.args, checkpoint_path=self.args.loadcheck_path
                )
                loaded = True
                logger.info(f"checkpoint: {self.args.loadcheck_path} is loaded")
            else:
                logger.warning(
                    "Finetune or validation mode, but no checkpoint is loaded"
                )
                loaded = False
        else:
            logger.info("No checkpoint is loaded")
            loaded = False
        if self.psm_config.psm_finetune_mode:
            settings = dict(
                psm_finetune_reset_head=self.psm_config.psm_finetune_reset_head,
                psm_finetune_head=(
                    self.psm_finetune_head.__class__ if self.psm_finetune_head else None
                ),
                psm_finetune_noise_mode=self.psm_config.psm_finetune_noise_mode,
            )
            logger.info(f"Finetune settings: {settings}")
            if self.psm_config.psm_finetune_reset_head:
                self.net.reset_head_for_finetune()
        else:
            assert not self.psm_finetune_head
            self.psm_finetune_head = None

        return loaded

    def load_pretrained_weights(self, args, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.

        Args:
            args: Command line arguments.
            checkpoint_path: Path to the pretrained weights.
        """
        checkpoints_state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if "model" in checkpoints_state:
            checkpoints_state = checkpoints_state["model"]
        elif "module" in checkpoints_state:
            checkpoints_state = checkpoints_state["module"]

        for key in list(checkpoints_state.keys()):
            if key.startswith("base."):
                checkpoints_state[key[5:]] = checkpoints_state.pop(key)

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

    def max_positions(self):
        """
        Returns the maximum positions of the net.
        """
        return self.net.max_positions

    @torch.no_grad()
    def sample(self, batched_data):
        """
        Sapmle method for diffusion model
        """
        self.net.eval()
        B, L = batched_data["token_type"].size()
        device = batched_data["token_type"].device
        token_id = batched_data["token_type"]
        padding_mask = token_id.eq(0)
        batched_data["padding_mask"] = padding_mask
        batched_data["aa_mask"] = torch.zeros_like(
            token_id, dtype=torch.bool, device=device
        )
        if self.psm_config.mode == 0:
            # probs = torch.tensor(
            #     [0.2, 0.2, 0.2, 0.2, 0.1, 0.1], dtype=torch.float, device="cpu"
            # )
            # probs = torch.tensor(
            #     [0.25, 0.25, 0.25, 0.25, 0.0, 0.0], dtype=torch.float, device="cpu"
            # )
            probs = torch.tensor(
                [1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0],
                dtype=torch.float,
                device="cpu",
            )
            idx = torch.multinomial(probs, num_samples=1).item()
            mode = idx + 1
        else:
            mode = self.psm_config.mode
        # mode = 1
        batched_data["mode"] = mode
        for i in range(1, 2):
            mode = i
            # print(mode)
            # MSAGen has 4 mode
            if mode == 1:
                self.psm_config.keep_clean_num = 1  # mode1: 1->1
                self.cut_off = 2
            elif mode == 2:
                self.psm_config.keep_clean_num = 2  # mode2: 2->2
                self.cut_off = 4
            elif mode == 3:
                self.psm_config.keep_clean_num = 4  # mode3: 4->4
                self.cut_off = 8
            elif mode == 4:
                self.psm_config.keep_clean_num = 8  # mode4: 8->8
                self.cut_off = 16
            elif mode == 5:
                self.psm_config.keep_clean_num = 16  # mode4: 16->16
                self.cut_off = 32
            elif mode == 6:
                self.psm_config.keep_clean_num = 6  # mode4: 32->32
                self.cut_off = 7
            elif mode == 7:
                self.psm_config.keep_clean_num = 7  # mode4: 32->32
                self.cut_off = 8
            else:
                self.psm_config.keep_clean_num = mode
                self.cut_off = mode + 1

            self.psm_config.keep_clean_num = mode
            self.cut_off = mode + 1
            if i == 1:
                batched_data["ori_128_msa_token_type"] = batched_data["msa_token_type"][
                    :, : self.cut_off, :
                ].clone()
            else:
                batched_data["ori_128_msa_token_type"] = batched_data[
                    "128_msa_token_type"
                ].clone()
                padnum = self.cut_off - batched_data["ori_128_msa_token_type"].shape[1]
                if padnum > 0:
                    pad = torch.full((B, padnum, L), 0, device=device)
                    batched_data["ori_128_msa_token_type"] = torch.cat(
                        (batched_data["ori_128_msa_token_type"], pad), dim=1
                    )
            batched_data["128_msa_token_type"] = batched_data["msa_token_type"][
                :, : self.cut_off, :
            ]
            ori_128_msa_one_hot = (
                F.one_hot(
                    batched_data["128_msa_token_type"].long(), num_classes=27
                ).float()
                * 2
                - 1
            )
            for i in range(self.cut_off):
                ## AR generate
                self.keep_clean = i + 1
                self.cut_off = self.keep_clean + 1
                for sample_time_index in range(self.psm_config.num_sampling_time):
                    batched_data["init_128_msa_one_hot"] = torch.zeros(
                        B, self.cut_off, L, 27, device=device
                    ).float()
                    if self.args.fp16:
                        batched_data["init_128_msa_one_hot"] = batched_data[
                            "init_128_msa_one_hot"
                        ].to(torch.float16)
                        ori_128_msa_one_hot = ori_128_msa_one_hot.to(torch.float16)
                    elif self.args.bf16:
                        batched_data["init_128_msa_one_hot"] = batched_data[
                            "init_128_msa_one_hot"
                        ].to(torch.bfloat16)
                        ori_128_msa_one_hot = ori_128_msa_one_hot.to(torch.bfloat16)
                    else:
                        pass
                    batched_data["128_msa_one_hot"] = self.diffnoise.get_sampling_start(
                        batched_data["init_128_msa_one_hot"]
                    )
                    padding_mask_2D = (
                        batched_data["token_type"]
                        .eq(0)
                        .unsqueeze(1)
                        .repeat(1, self.cut_off, 1)
                    )
                    clean_mask = torch.zeros(
                        B, self.cut_off, L, dtype=torch.bool, device=device
                    )
                    min_D = min(self.cut_off, ori_128_msa_one_hot.shape[1])
                    clean_mask = clean_mask.masked_fill(padding_mask_2D, True)
                    # set first to clean
                    if self.psm_config.keep_clean_num > 0:
                        clean_mask[:, : self.psm_config.keep_clean_num, :] = True
                    if clean_mask is not None:
                        batched_data["128_msa_one_hot"][:, :min_D, :, :] = torch.where(
                            clean_mask[:, :min_D, :].unsqueeze(-1),
                            ori_128_msa_one_hot,
                            batched_data["128_msa_one_hot"][:, :min_D, :, :],
                        )
                    batched_data["clean_mask"] = clean_mask
                    # T = torch.full((B,), self.T - 1, device=device)
                    # x_T = self.diffusion.q_sample(
                    #     batched_data["128_msa_one_hot"], T, clean_mask, device
                    # )
                    batched_data["128_2D_padding_mask"] = padding_mask_2D
                    # batched_data["128_msa_one_hot"] = x_T
                    # batched_data["time_step"] = T
                    # true_prob = self.calculate_prob(batched_data["msa_token_type"])
                    if self.psm_config.diffusion_mode == "epsilon":
                        for t in range(
                            self.psm_config.num_timesteps - 1,
                            -1,
                            self.psm_config.num_timesteps_stepsize,
                        ):
                            # forward
                            time_step = self.time_step_sampler.get_continuous_time_step(
                                t,
                                B,
                                device=device,
                                dtype=batched_data["128_msa_one_hot"].dtype,
                            )
                            time_step = (
                                time_step.unsqueeze(-1)
                                .unsqueeze(-1)
                                .repeat(1, self.cut_off, L)
                            )
                            if clean_mask is not None:
                                time_step = time_step.masked_fill(clean_mask, 0.0)
                            x_t = batched_data["128_msa_one_hot"].clone()
                            # batched_data[
                            #     "sqrt_one_minus_alphas_cumprod_t"
                            # ] = self.diffnoise._extract(
                            #     self.diffnoise.sqrt_one_minus_alphas_cumprod,
                            #     (time_step * self.psm_config.num_timesteps).long(),
                            #     batched_data["128_msa_one_hot"].shape,
                            # )
                            batched_data["time_step"] = time_step
                            net_result = self.net(batched_data)
                            predicted_noise = net_result["noise_pred"]
                            epsilon = self.diffnoise.get_noise(
                                batched_data["128_msa_one_hot"]
                            )
                            batched_data[
                                "128_msa_one_hot"
                            ] = self.diffusion_process.sample_step(
                                x_t,
                                batched_data["init_128_msa_one_hot"],
                                predicted_noise,
                                epsilon,
                                t,
                                stepsize=-self.psm_config.num_timesteps_stepsize,
                            )
                            if clean_mask is not None:
                                batched_data["128_msa_one_hot"] = torch.where(
                                    clean_mask.unsqueeze(-1),
                                    ori_128_msa_one_hot,
                                    batched_data["128_msa_one_hot"],
                                )
                            batched_data["128_msa_one_hot"] = batched_data[
                                "128_msa_one_hot"
                            ].detach()
                    elif self.psm_config.diffusion_mode == "diff-lm":
                        # diff-lm
                        for t in range(
                            self.psm_config.num_timesteps - 1,
                            -1,
                            self.psm_config.num_timesteps_stepsize,
                        ):
                            # forward
                            time_step = self.time_step_sampler.get_continuous_time_step(
                                t,
                                B,
                                device=device,
                                dtype=batched_data["128_msa_one_hot"].dtype,
                            )
                            time_step = (
                                time_step.unsqueeze(-1)
                                .unsqueeze(-1)
                                .repeat(1, self.cut_off, L)
                            )
                            if clean_mask is not None:
                                time_step = time_step.masked_fill(clean_mask, 0.0)
                            x_t = batched_data["128_msa_one_hot"].clone()
                            # batched_data[
                            #     "sqrt_one_minus_alphas_cumprod_t"
                            # ] = self.diffnoise._extract(
                            #     self.diffnoise.sqrt_one_minus_alphas_cumprod,
                            #     (time_step * self.psm_config.num_timesteps).long(),
                            #     batched_data["128_msa_one_hot"].shape,
                            # )
                            batched_data["time_step"] = time_step
                            net_result = self.net(batched_data)
                            x0_pred = net_result["noise_pred"]
                            if t == 0:
                                batched_data["128_msa_one_hot"] = x0_pred
                                if clean_mask is not None:
                                    batched_data["128_msa_one_hot"][
                                        :, :min_D, :, :
                                    ] = torch.where(
                                        clean_mask[:, :min_D, :].unsqueeze(-1),
                                        ori_128_msa_one_hot,
                                        batched_data["128_msa_one_hot"][
                                            :, :min_D, :, :
                                        ],
                                    )
                                continue
                            else:
                                time_step_pre = (
                                    self.time_step_sampler.get_continuous_time_step(
                                        t - 1,
                                        B,
                                        device=device,
                                        dtype=batched_data["128_msa_one_hot"].dtype,
                                    )
                                )

                                time_step_pre = (
                                    time_step_pre.unsqueeze(-1)
                                    .unsqueeze(-1)
                                    .repeat(1, self.cut_off, L)
                                )
                                (
                                    noise_msa,
                                    noise,
                                    sqrt_one_minus_alphas_cumprod_t,
                                    sqrt_alphas_cumprod_t,
                                ) = self.diffnoise.noise_sample(
                                    x_start=x0_pred,
                                    t=time_step_pre,
                                    clean_mask=clean_mask,
                                )
                                # epsilon = self.diffnoise.get_noise(batched_data["128_msa_one_hot"])
                                # batched_data[
                                #     "128_msa_one_hot"
                                # ] = self.diffusion_process.sample_step(
                                #     x_t,
                                #     batched_data["init_128_msa_one_hot"],
                                #     predicted_noise,
                                #     epsilon,
                                #     t,
                                #     stepsize=-self.psm_config.num_timesteps_stepsize,
                                # )
                                batched_data["128_msa_one_hot"] = noise_msa
                                if clean_mask is not None:
                                    batched_data["128_msa_one_hot"][
                                        :, :min_D, :, :
                                    ] = torch.where(
                                        clean_mask[:, :min_D, :].unsqueeze(-1),
                                        ori_128_msa_one_hot,
                                        batched_data["128_msa_one_hot"][
                                            :, :min_D, :, :
                                        ],
                                    )
                                batched_data["128_msa_one_hot"] = batched_data[
                                    "128_msa_one_hot"
                                ].detach()
                        pred_msa = (
                            batched_data["128_msa_one_hot"].clone().argmax(dim=-1)
                        )

                        # kl_loss=self.kl(x_t.argmax(dim=-1),batched_data["msa_token_type"])
                        # pred_prob = self.calculate_prob(pred_msa.argmax(dim=-1))
                        pred_seq = self.convert(pred_msa)
                    else:
                        # OADM
                        if self.psm_config.OADM_row_random:
                            sigma = torch.stack(
                                [
                                    torch.stack(
                                        [
                                            torch.randperm(L, device=device)
                                            for _ in range(self.cut_off)
                                        ]
                                    )
                                    for _ in range(B)
                                ]
                            )
                        else:
                            sigma = (
                                torch.randperm(L, device=device)
                                .unsqueeze(0)
                                .unsqueeze(0)
                                .repeat(B, self.cut_off, 1)
                            )
                        batched_data["128_msa_token_type"] = torch.full(
                            (B, self.cut_off, L), 27, device=device
                        )  # B,D,L 27means mask
                        if clean_mask is not None:
                            batched_data["128_msa_token_type"][
                                :, :min_D, :
                            ] = torch.where(
                                clean_mask[:, :min_D, :],
                                batched_data["ori_128_msa_token_type"],
                                batched_data["128_msa_token_type"][:, :min_D, :],
                            )
                        for t in range(1, L + 1):
                            # m = (sigma < t).unsqueeze(0).unsqueeze(0).repeat(B,self.cut_off,1)
                            n = sigma + 1 == t
                            time_step = torch.full(
                                (B, self.cut_off, L), L - t + 1, device=device
                            )
                            if clean_mask is not None:
                                time_step = time_step.masked_fill(clean_mask, 0)
                            batched_data["time_step"] = time_step
                            net_result = self.net(batched_data)
                            logits = net_result["noise_pred"]  # B D L 27

                            # according the biggest prob to denoise
                            # if True:
                            #     is_mask = batched_data["128_msa_token_type"] == 27
                            #     logits_max_perL = F.softmax(logits, dim=-1).max(dim=-1).values
                            #     logits_max_perL = logits_max_perL.masked_fill(
                            #         ~is_mask, -float("inf")
                            #     )
                            #     # col consistency
                            #     # logits_max_perL = logits_max_perL.sum(dim=1).unsqueeze(1).repeat(1,logits.size(1),1) #B D L
                            #     n = (
                            #         F.one_hot(
                            #             logits_max_perL.argmax(dim=-1),
                            #             num_classes=logits.size(2),
                            #         ).bool()
                            #         & is_mask
                            #     )
                            logits = F.softmax(logits, dim=-1)
                            logits[:, :, :, 26] /= 1
                            B, D, L, V = logits.shape
                            sample_probs = logits.permute(0, 1, 2, 3).reshape(
                                -1, V
                            )  # [B*D*L, 27]
                            sample = torch.multinomial(
                                sample_probs, num_samples=1
                            )  # [B*D*L, 1]
                            sample = sample.view(B, D, L)
                            # sample = logits.argmax(dim=-1)  # B D L
                            batched_data["128_msa_token_type"] = torch.where(
                                n, sample, batched_data["128_msa_token_type"]
                            )
                            if clean_mask is not None:
                                batched_data["128_msa_token_type"][
                                    :, :min_D, :
                                ] = torch.where(
                                    clean_mask[:, :min_D, :],
                                    batched_data["ori_128_msa_token_type"],
                                    batched_data["128_msa_token_type"][:, :min_D, :],
                                )
                        pred_msa = batched_data["128_msa_token_type"]
                        pred_seq = self.convert(batched_data["128_msa_token_type"])

        gt_seq = self.convert(batched_data["msa_token_type"][:, : self.cut_off, :])
        # samples.append(self.convert(pred_msa.argmax(dim=-1)))
        results = self.calculate_acc(
            pred_msa[:, :min_D, :],
            batched_data,
        )
        true_prob = self.calculate_prob(batched_data["msa_token_type"])
        pred_prob = self.calculate_prob(pred_msa)
        diff_prob = abs(true_prob - pred_prob).sum()

        return (
            results,
            pred_seq,
            gt_seq,
            diff_prob,
        )

        # self.net.train()
        # plot_probability_heatmaps(true_prob, pred_prob, padding_mask, batched_data)
        # return torch.stack(samples, dim=0)

    def calculate_acc(self, generated, batched_data):
        """
        Calculate mutation prediction performance per sample.

        Returns:
            - avg_precision
            - avg_recall
            - avg_f1
            - avg_mutation_accuracy
        """
        B, D, L = generated.shape

        if self.psm_config.keep_clean_num > 0:
            clean_msa_num = self.psm_config.keep_clean_num
        else:
            clean_msa_num = 1
        ground_truth = batched_data["msa_token_type"][:, : self.cut_off, :]
        ref_gt = ground_truth[:, 0, :]  # (B, L)
        msa_gt = ground_truth[:, clean_msa_num:, :]  # (B, D-1, L)
        ref_pred = ground_truth[:, 0, :]  # (B, L)
        msa_pred = generated[:, clean_msa_num:, :]  # (B, D-1, L)
        msa_exclude_ori = batched_data["msa_token_type"][:, 1:65, :]  # B 64 L
        results = {
            "precision": [],
            "recall": [],
            "f1": [],
            "mutation_accuracy": [],
            "mutation_num": [],
            "union_mutation_num": [],
            "precision_u": [],
            "recall_u": [],
            "f1_u": [],
        }

        for b in range(B):
            gt = msa_gt[b]  # (D-1, L)
            pred = msa_pred[b]
            ref_gt_b = ref_gt[b]
            ref_pred_b = ref_pred[b]
            msa_exclude_ori_b = msa_exclude_ori[b]

            mutation_gt = gt != ref_gt_b.unsqueeze(0)  # (D-1, L)
            union_mutation_gt = msa_exclude_ori_b != ref_gt_b.unsqueeze(0)

            mutation_pred = pred != ref_pred_b.unsqueeze(0)
            union_mutation = union_mutation_gt.any(dim=0, keepdim=True)  # 1 L
            tp = (
                (mutation_gt & mutation_pred).sum().float()
            )  # pred mutation and gt mutation
            fp = (
                (~mutation_gt & mutation_pred).sum().float()
            )  # pred mutation but gt not mutation
            fn = (
                (mutation_gt & ~mutation_pred).sum().float()
            )  # gt mutation but pred not mutation

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            tp_u = (
                (union_mutation & mutation_pred).sum().float()
            )  # pred mutation and gt mutation
            fp_u = (
                (~union_mutation & mutation_pred).sum().float()
            )  # pred mutation but gt not mutation
            fn_u = (
                (union_mutation & ~mutation_pred).sum().float()
            )  # gt mutation but pred not mutation

            precision_u = tp_u / (tp_u + fp_u + 1e-8)
            recall_u = tp_u / (tp_u + fn_u + 1e-8)
            f1_u = 2 * precision_u * recall_u / (precision_u + recall_u + 1e-8)

            # here, using ground_truth[:, 1:, :][b] rather than mutation_gt for all mutaion num except for original
            mutation_num = (ground_truth[:, 1:, :][b] != ref_gt_b.unsqueeze(0)).sum(
                dim=-1
            )
            union_mutation_num = union_mutation.sum(dim=-1)
            # Calculate mutation accuracy
            correct_mutated_aa = (
                gt[mutation_gt & mutation_pred] == pred[mutation_gt & mutation_pred]
            ).sum()
            mutation_acc = correct_mutated_aa.float() / (tp + 1e-8)

            # Append to result
            results["precision"].append(precision)
            results["recall"].append(recall)
            results["f1"].append(f1)
            results["mutation_accuracy"].append(mutation_acc)
            results["mutation_num"].append(mutation_num)
            results["union_mutation_num"].append(union_mutation_num)
            results["precision_u"].append(precision_u)
            results["recall_u"].append(recall_u)
            results["f1_u"].append(f1_u)

        for key in results:
            if len(results[key]) == 0:
                results[key] = torch.tensor([0.0], device=generated.device)
            else:
                results[key] = torch.stack(results[key])

        return results

    def calculate_prob(self, x):
        # calculate true prob
        B, D, L = x.size()
        msa_token_type_t = x.transpose(1, 2)  # B L D

        counts = torch.zeros(B, L, 26, device=x.device, dtype=torch.int32)
        indices = (msa_token_type_t - 1).clamp(
            min=0
        )  # B L D minus 1 so that 0 means indicates=0 which indicates the first aa
        valid_mask = msa_token_type_t.ne(0)  # B L D
        # count num of valid according indices
        counts.scatter_add_(2, indices.long(), valid_mask.int())
        true_prob = counts / valid_mask.int().sum(dim=-1, keepdim=True).clamp(min=1)
        true_prob = (true_prob + 1e-5) / true_prob.sum(dim=-1, keepdim=True)
        return true_prob

    def convert(self, x):
        inv_vocab = {v: k for k, v in MSAVOCAB.items()}
        # indices = x.argmax(dim=-1)
        indices = x
        B, D, L = indices.shape
        sequences = []
        for b in range(B):
            sample_seqs = []
            for d in range(D):
                token_ids = indices[b, d].tolist()
                tokens = []
                for idx in token_ids:
                    if idx == 0:
                        print("error")
                        continue
                    tokens.append(inv_vocab.get(idx))
                sample_seqs.append("".join(tokens))
            sequences.append(sample_seqs)
        return sequences

    def _set_noise(self, batched_data):
        B, D, max_L = batched_data["msa_token_type"].shape
        min_D = min(D, batched_data["cut_off"])
        device = batched_data["msa_token_type"].device
        batched_data["ori_128_msa_one_hot"] = batched_data["128_msa_one_hot"].clone()
        if self.psm_config.diffusion_mode in ["epsilon", "diff-lm"]:
            # time sample
            time_step, clean_mask = self.time_step_sampler.sample(
                B,
                device,
                batched_data["128_msa_one_hot"].dtype,
                self.psm_config.clean_sample_ratio,
            )
            clean_mask = clean_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, min_D, max_L)
            time_step = time_step.unsqueeze(-1).unsqueeze(-1).repeat(1, min_D, max_L)

            # t = torch.randint(0, 1000, (B,), device=device)

            # set padding to clean
            clean_mask = clean_mask.masked_fill(
                batched_data["128_2D_padding_mask"], True
            )

            # set keepclean to clean
            if self.psm_config.keep_clean_num > 0:
                clean_mask[:, : self.psm_config.keep_clean_num, :] = True

            time_step = time_step.masked_fill(clean_mask, 0.0)

            (
                noise_msa,
                noise,
                sqrt_one_minus_alphas_cumprod_t,
                sqrt_alphas_cumprod_t,
            ) = self.diffnoise.noise_sample(
                x_start=batched_data["128_msa_one_hot"],
                t=time_step,
                clean_mask=clean_mask,
            )
            # x_t = self.diffusion.q_sample(
            #     batched_data["128_msa_one_hot"],s
            #     t,
            #     clean_mask,
            #     device,
            # )
            batched_data["noise"] = noise
            batched_data[
                "sqrt_one_minus_alphas_cumprod_t"
            ] = sqrt_one_minus_alphas_cumprod_t
            batched_data["sqrt_alphas_cumprod_t"] = sqrt_alphas_cumprod_t
            batched_data["128_msa_one_hot"] = noise_msa.to(dtype=time_step.dtype)
            batched_data["time_step"] = time_step
            batched_data["clean_mask"] = clean_mask
        else:
            # OADM
            time_step_list = []
            clean_mask_list = []
            noise_msa_list = []

            for b in range(B):
                L = (~batched_data["padding_mask"][b]).sum().item()
                if L <= 1:
                    t = 1
                else:
                    t = np.random.randint(1, L)
                num_mask = L - t + 1
                # Append time_step
                time_step_list.append(torch.tensor(num_mask, device=device))
                # Generate mask
                if self.psm_config.OADM_row_random:
                    mask = torch.zeros((min_D, max_L), dtype=torch.bool, device=device)
                    for i in range(min_D):
                        rand_indices = torch.randperm(L, device=device)[:num_mask]
                        mask[i, rand_indices] = True
                else:
                    mask_aa = np.random.choice(L, num_mask, replace=False)
                    index_aa = np.arange(0, max_L)
                    mask = np.isin(index_aa, mask_aa, invert=False)
                    # Mask inputs
                    mask = (
                        torch.tensor(mask, dtype=torch.bool, device=device)
                        .unsqueeze(0)
                        .repeat(min_D, 1)
                    )
                clean_mask_list.append(~mask)
                x_t = batched_data["128_msa_token_type"][b].masked_fill(
                    mask, 27
                )  # 27 means <mask>
                noise_msa_list.append(x_t)
            time_step = (
                torch.stack(time_step_list, dim=0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, min_D, max_L)
            )
            clean_mask = torch.stack(clean_mask_list, dim=0)
            noise_msa = torch.stack(noise_msa_list, dim=0)
            # set padding to clean
            clean_mask = clean_mask.masked_fill(
                batched_data["128_2D_padding_mask"], True
            )
            # set keepclean to clean
            if self.psm_config.keep_clean_num > 0:
                clean_mask[:, : self.psm_config.keep_clean_num, :] = True
            time_step = time_step.masked_fill(clean_mask, 0)
            noise_msa = torch.where(
                clean_mask, batched_data["ori_128_msa_token_type"], noise_msa
            )
            batched_data["time_step"] = time_step
            batched_data["clean_mask"] = clean_mask
            batched_data["128_msa_token_type"] = noise_msa
        # return x_t,t

    def _pre_forward_operation(
        self,
        batched_data,
    ):
        """
        pre forward operation
        """
        # if self.psm_config.mode == 0:
        #     # probs = torch.tensor(
        #     #     [0.2, 0.2, 0.2, 0.2, 0.1, 0.1], dtype=torch.float, device="cpu"
        #     # )
        #     # probs = torch.tensor(
        #     #     [0.25, 0.25, 0.25, 0.25, 0.0, 0.0], dtype=torch.float, device="cpu"
        #     # )
        #     probs = torch.tensor(
        #         [1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7],
        #         dtype=torch.float,
        #         device="cpu",
        #     )
        #     idx = torch.multinomial(probs, num_samples=1).item()
        #     mode = idx + 1
        # else:
        #     mode = self.psm_config.mode
        # # mode = 1
        # batched_data["mode"] = mode
        # # MSAGen has 4 mode
        # if mode == 1:
        #     self.psm_config.keep_clean_num = 1  # mode1: 1->1
        #     self.cut_off = 2
        # elif mode == 2:
        #     self.psm_config.keep_clean_num = 2  # mode2: 2->2
        #     self.cut_off = 3
        # elif mode == 3:
        #     self.psm_config.keep_clean_num = 3  # mode3: 4->4
        #     self.cut_off = 4
        # elif mode == 4:
        #     self.psm_config.keep_clean_num = 4  # mode4: 8->8
        #     self.cut_off = 5
        # elif mode == 5:
        #     self.psm_config.keep_clean_num = 1  # mode4: 16->16
        #     self.cut_off = 10
        # elif mode == 6:
        #     self.psm_config.keep_clean_num = 6  # mode4: 32->32
        #     self.cut_off = 7
        # elif mode == 7:
        #     self.psm_config.keep_clean_num = 7  # mode4: 32->32
        #     self.cut_off = 8
        # if self.psm_config.mode == 0:
        #     # 从 1 到 63 均匀采样一个 mode
        #     probs = torch.ones(63, dtype=torch.float, device="cpu") / 63
        #     idx = torch.multinomial(probs, num_samples=1).item()
        #     mode = idx + 1  # idx ∈ [0,62] → mode ∈ [1,63]
        # else:
        #     mode = self.psm_config.mode
        # mode = 1
        # batched_data["mode"] = mode
        self.psm_config.keep_clean_num = 1
        self.cut_off = self.psm_config.cutoff
        cut_off = self.cut_off
        batched_data["cut_off"] = cut_off
        token_id = batched_data["token_type"]
        padding_mask = token_id.eq(0)  # B x T x 1
        B, D, L = batched_data["msa_token_type"].shape
        msa_token_type = batched_data["msa_token_type"]
        batched_data["padding_mask"] = padding_mask
        batched_data["row_padding_mask"] = (msa_token_type == 0).all(dim=-1)
        batched_data["col_padding_mask"] = (msa_token_type == 0).all(dim=1)
        batched_data["2D_padding_mask"] = msa_token_type == 0
        batched_data["128_msa_token_type"] = batched_data["msa_token_type"][
            :, :cut_off, :
        ]
        batched_data["128_msa_one_hot"] = (
            F.one_hot(batched_data["128_msa_token_type"].long(), num_classes=27).float()
            * 2
            - 1
        )  # 26 plus <pad>
        batched_data["ori_128_msa_token_type"] = batched_data[
            "128_msa_token_type"
        ].clone()
        if self.args.fp16:
            batched_data["128_msa_one_hot"] = batched_data["128_msa_one_hot"].to(
                torch.float16
            )
        elif self.args.bf16:
            batched_data["128_msa_one_hot"] = batched_data["128_msa_one_hot"].to(
                torch.bfloat16
            )
        else:
            pass

        batched_data["128_row_padding_mask"] = batched_data["row_padding_mask"][
            :, :cut_off
        ]
        batched_data["128_col_padding_mask"] = batched_data["col_padding_mask"][
            :, :cut_off
        ]
        batched_data["128_2D_padding_mask"] = batched_data["2D_padding_mask"][
            :, :cut_off, :
        ]
        # set aa_mask
        mask_ratio = self.psm_config.mask_ratio
        aa_mask = torch.rand_like(token_id, dtype=torch.float) < mask_ratio
        aa_mask = aa_mask & ~padding_mask
        batched_data["aa_mask"] = aa_mask

        # calculate true prob
        msa_token_type_t = batched_data["msa_token_type"].transpose(1, 2)  # B L D

        counts = torch.zeros(
            B, L, 26, device=batched_data["msa_token_type"].device, dtype=torch.int32
        )
        indices = (msa_token_type_t - 1).clamp(
            min=0
        )  # B L D minus 1 so that 0 means indicates=0 which indicates the first aa
        valid_mask = msa_token_type_t.ne(0)  # B L D
        # count num of valid according indices
        counts.scatter_add_(2, indices.long(), valid_mask.int())
        true_prob = counts / valid_mask.int().sum(dim=-1, keepdim=True).clamp(min=1)
        batched_data["true_prob"] = (true_prob + 1e-5) / true_prob.sum(
            dim=-1, keepdim=True
        )
        self._set_noise(batched_data)

    def _KL_reconstruction_loss(self, batched_data, x0_pred, x0, filter_mask):
        t = batched_data["time_step"]
        time = (t.float() * (self.psm_config.num_timesteps - 1)).long()  # B D L
        if torch.all(time == 0):
            return torch.tensor(0.0, requires_grad=True, device=time.device)

        x_t_1_pred = self.diffnoise.get_x_t_1(x0_pred, t)[filter_mask]
        x_t_1_pred = F.log_softmax(x_t_1_pred, dim=-1)

        x_t_1 = self.diffnoise.get_x_t_1(x0, t)[filter_mask]
        x_t_1 = F.softmax(x_t_1, dim=-1)
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        return kl_loss(x_t_1_pred, x_t_1)

    def _forward_net(self, batched_data, skip_sample=False, **kwargs):
        """
        Forward pass of the model.

        Args:
            batched_data: Input data for the forward pass.
            skip_sample: Skip the sampling step.
            **kwargs: Additional keyword
        """

        if self.psm_config.psm_sample_structure_in_finetune:
            self.net.eval()

        # context = torch.no_grad() if self.psm_config.freeze_backbone else nullcontext()
        # with context:
        result_dict = self.net(
            batched_data,
            **kwargs,
        )

        result_dict["data_name"] = (
            batched_data["data_name"] if "data_name" in batched_data else None
        )

        result_dict["position_ids"] = batched_data["position_ids"]

        return result_dict

    def forward(self, batched_data, skip_sample=False, **kwargs):
        """
        Forward pass of the model.

        Args:
            batched_data: Input data for the forward pass.
            **kwargs: Additional keyword arguments.
        """

        if (
            self.psm_config.sample_in_validation
            # and not self.training
            # and not skip_sample
        ):
            (
                results,
                pred_seqs,
                gt_seqs,
                _,
            ) = self.sample(batched_data)
            B = batched_data["msa_token_type"].shape[0]
            for i in range(B):
                pdbid = batched_data["unique_ids"][i]
                precision = results["precision"][i]
                recall = results["recall"][i]
                f1 = results["f1"][i]
                mutation_accuracy_i = results["mutation_accuracy"][i]
                pred_seq = pred_seqs[i]
                gt_seq = gt_seqs[i]
                mutation_num = results["mutation_num"][i].tolist()
                union_mutaion_num = results["union_mutation_num"][i].tolist()
                union_precision = results["precision_u"][i]
                union_recall = results["recall_u"][i]
                union_f1 = results["f1_u"][i]

                save_path = os.path.join(self.args.save_dir, f"{pdbid}.json")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                record = f"pdbid: {pdbid},precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, mutation_accuracy: {mutation_accuracy_i:.3f}"
                record_u = f"precision_u: {union_precision:.3f}, recall: {union_recall:.3f}, f1: {union_f1:.3f}"
                with open(save_path, "w") as f:
                    json.dump(
                        {
                            "record": record,
                            "record_u": record_u,
                            "mutation_nums": mutation_num,
                            "union_mutaion_num": union_mutaion_num,
                            "pred_seq": pred_seq,
                            "ground_truth": gt_seq,
                        },
                        f,
                        indent=4,
                    )
                logger.info(record)
            # print(1)

        self._pre_forward_operation(batched_data)
        result_dict = self._forward_net(batched_data, skip_sample, **kwargs)

        # if (
        #     self.psm_config.sample_in_validation
        #     and not self.training
        #     and not skip_sample
        # ):
        #     result_dict.update(results)

        return result_dict

    def compute_loss(self, model_output, batched_data) -> ModelOutput:
        """
        Compute loss for the model.

        Args:
            model_output: The output from the model.
            batched_data: The batch data.

        Returns:
            ModelOutput: The model output which includes loss, log_output, num_examples.
        """
        noise_pred = model_output["noise_pred"]
        mutation_pred = model_output["mutation_pred"]
        filter_mask = ~batched_data["clean_mask"]  # B D L
        B, D, L = batched_data["128_2D_padding_mask"].size()
        is_gap = batched_data["ori_128_msa_token_type"] == 26
        if self.psm_config.diffusion_mode == "epsilon":
            noise_label = batched_data["noise"]
            diffusion_loss = self.compute_diff_loss(
                noise_label[filter_mask],
                noise_pred[filter_mask],
                is_gap[filter_mask],
                1.0,
                "L2",
                batched_data,
                filter_mask
                # batched_data["ori_128_msa_one_hot"].argmax(dim=-1).unsqueeze(-1).view(B,D*L,-1),
            )
        elif self.psm_config.diffusion_mode in ["diff-lm", "OADM"]:
            noise_label = batched_data["ori_128_msa_token_type"]
            (
                diffusion_loss,
                diff_celoss,
                recons_loss,
                diff_bceloss,
            ) = self.compute_diff_loss(
                noise_label,
                noise_pred,
                mutation_pred,
                is_gap,
                1.0,
                "L2",
                batched_data,
                filter_mask,
                batched_data["time_step"],
                batched_data["128_2D_padding_mask"]
                # batched_data["ori_128_msa_one_hot"].argmax(dim=-1).unsqueeze(-1).view(B,D*L,-1),
            )
            # pred_prob = self.calculate_prob(noise_pred.argmax(dim=-1))
            # diffusion_kl_loss = F.kl_div(
            #     pred_prob[filter_mask[:, 0, :]].log(),
            #     batched_data["true_prob"][filter_mask[:, 0, :]],
            #     reduction="batchmean",
            # )
            # diffusion_loss += diffusion_kl_loss
            # print("diff_loss",diffusion_loss)
            # noise_pred means x0_pred

        batched_data["init_128_msa_one_hot"] = torch.zeros(
            B, D, L, 27, device=batched_data["msa_token_type"].device
        ).float()

        loss = diffusion_loss  # + diff_bceloss  # + kl_loss
        # loss += cross_entropy_loss
        logging_output = {
            "total_loss": float(loss.detach()),
            "diffusion_loss": float(diffusion_loss.detach()),
            "diffusion_bce_loss": float(diff_bceloss.detach()),
            # "diffusion_kl_loss": float(diffusion_kl_loss.detach()),
            "diffusion_ce_loss": float(diff_celoss.detach()),
            "recons_loss": float(recons_loss.detach()),
        }

        return ModelOutput(
            loss=loss,
            num_examples=model_output["padding_mask"].shape[0],
            log_output=logging_output,
        )

    def compute_diff_loss(
        self,
        label,
        pred,
        mutation_pred,
        is_gap,
        gap_weight,
        loss_type="L1",
        batched_data=None,
        filter_mask=None,
        time_step=None,
        padding_mask=None,
    ):
        if self.psm_config.diffusion_mode == "epsilon":
            if loss_type == "L1":
                loss = torch.abs(label - pred).mean(dim=-1)
            elif loss_type == "L2":
                loss = ((label - pred) ** 2).mean(dim=-1)
            weights = torch.where(
                is_gap,
                torch.tensor(gap_weight, device=is_gap.device, dtype=loss.dtype),
                torch.tensor(1.0, device=is_gap.device, dtype=loss.dtype),
            )

            loss_weighted = loss * weights

            total_loss = loss_weighted.sum() / loss_weighted.numel()
            return total_loss
        elif self.psm_config.diffusion_mode == "diff-lm":
            # diff-lm
            ce_loss = self.noise_loss(pred[filter_mask], label[filter_mask].long())
            differ_mask = ~(
                batched_data["ori_128_msa_token_type"]
                == batched_data["token_type"].unsqueeze(1)
            )[
                filter_mask
            ]  # true means differ
            # if differ, enlarge the loss, except for gap
            differ_mask = differ_mask & ~is_gap[filter_mask]
            ce_loss = ce_loss * (1 + 9.0 * differ_mask.float())
            # 0.2 for gap
            # ce_loss = ce_loss * (1 - 0.8 * is_gap[filter_mask].float())
            kl_loss = self._KL_reconstruction_loss(
                batched_data, pred, batched_data["ori_128_msa_one_hot"], filter_mask
            )
            loss = ce_loss.mean() + kl_loss
            return loss, ce_loss.mean(), kl_loss.mean()
        else:
            # OADM
            # shift
            pred = pred[:, :-1, :, :]
            label = label[:, 1:, :]
            padding_mask = (padding_mask[:, :-1, :]) | (
                padding_mask[:, 1:, :]
            )  # to avoid predict padding
            is_gap = is_gap[:, 1:, :]
            filter_mask = filter_mask[:, :-1, :]
            filter_mask = filter_mask & ~padding_mask  # update filter_mask
            time_step = time_step[:, :-1, :]

            ce_loss = self.noise_loss(pred.permute(0, 3, 1, 2), label.long())
            # print(ce_loss)
            differ_mask = ~(
                label == batched_data["token_type"].unsqueeze(1)
            )  # true means differ
            # if differ, enlarge the loss, except for gap
            differ_mask = differ_mask & ~is_gap
            same_mask = (~differ_mask) & ~is_gap
            # ce_loss = ce_loss * (1 + 1.0 * differ_mask.float())
            bce_loss = self.bce_loss(
                mutation_pred.squeeze(-1)[:, 1:, :], differ_mask.float()
            )

            # reweight
            non_pad_counts = (~padding_mask).sum(dim=-1, keepdim=True)
            non_pad_counts = non_pad_counts.expand_as(padding_mask).float()
            weights = non_pad_counts * (1.0 / (time_step + 1e-5))
            total_loss = ce_loss  # + bce_loss
            reweight_loss = total_loss * weights  # B D L
            # print(reweight_loss)
            reweight_loss = reweight_loss * filter_mask.float()
            # print(filter_mask)
            # first sample-internal mean and then cross-sample mean and according mask to mean
            counts_diff = (
                (differ_mask & filter_mask).sum(dim=2).clamp(min=1).float()
            )  # (B,D)
            counts_same = (
                (same_mask & filter_mask).sum(dim=2).clamp(min=1).float()
            )  # (B,D)
            counts_gap = (is_gap & filter_mask).sum(dim=2).clamp(min=1).float()  # (B,D)

            sum_diff = (reweight_loss * (differ_mask & filter_mask).float()).sum(
                dim=2
            )  # (B,D)
            sum_same = (reweight_loss * (same_mask & filter_mask).float()).sum(
                dim=2
            )  # (B,D)
            sum_gap = (reweight_loss * (is_gap & filter_mask).float()).sum(
                dim=2
            )  # (B,D)

            mean_diff = sum_diff / counts_diff  # (B,D)
            mean_same = sum_same / counts_same  # (B,D)
            mean_gap = sum_gap / counts_gap  # (B,D)

            per_row_loss = torch.stack([mean_diff, mean_same, mean_gap], dim=-1).mean(
                dim=-1
            )  # (B,D)
            D = per_row_loss.shape[1]
            row_weight = torch.arange(D, 0, -1, device=per_row_loss.device).float() / D
            # print(row_weight)
            per_row_loss = per_row_loss * row_weight.unsqueeze(0)  # (B,D)
            # valid_counts = filter_mask.sum(dim=(1, 2)).clamp(min=1).float()  # (B)
            per_sample_loss = (
                per_row_loss.sum(dim=-1)
                / filter_mask.any(dim=2).sum(dim=1).clamp(min=1).float()
            )  # B
            # print(per_sample_loss)
            # per_sample_loss = reweight_loss.sum(dim=(1, 2)) / valid_counts
            loss = per_sample_loss.mean()
            mean_ce = ce_loss[filter_mask].mean()
            mean_bce_loss = bce_loss[filter_mask].mean()
            if torch.isnan(mean_ce):
                mean_ce = torch.tensor(
                    0.0, device=mean_ce.device, requires_grad=True
                )  # in case of filter_mask all false
            if torch.isnan(mean_bce_loss):
                mean_bce_loss = torch.tensor(
                    0.0, device=mean_ce.device, requires_grad=True
                )  # in case of filter_mask all false
            # print(ce_loss)
            # print(ce_loss[filter_mask])
            kl_loss = torch.tensor(0.0, requires_grad=True)
            return loss, mean_ce, kl_loss, mean_bce_loss
            # differ_mask = differ_mask & ~is_gap[filter_mask]

    def compute_cross_entropy_loss(self, logits, target, filter_mask):
        """
        compute cross entropy loss
        """

        # log_prob = F.log_softmax(logits.float(), dim=-1)  # B,D,L,num_classes
        # loss = -(target * log_prob).sum(dim=-1)
        # loss = loss[filter_mask]
        # B, D, L, C = logits.size()
        # logits = logits.view(B, D * L, C).float().permute(0, 2, 1)
        # target = target.view(B, D * L)
        loss = self.aa_mlm_loss(logits, target)
        return loss

    # def compute_l1_loss(self, logits, target):
    #     """
    #     compute L1 loss
    #     """
    #     return F.l1_loss(logits, target, reduction="mean")

    def config_optimizer(self, model: Optional[nn.Module]):
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            tuple[Optimizer, LRScheduler]:
        """
        return (None, None)


class MSAGen(nn.Module):
    """
    Class for training Physics science module
    """

    def __init__(
        self,
        args,
        psm_config: PSMConfig,
    ):
        super().__init__()
        self.max_positions = args.max_positions
        self.args = args
        self.backbone = args.backbone

        self.psm_config = psm_config

        self.embedding = MSAGenSeqEmbedding(psm_config)

        # self.encoder = MSAGenEncoder(args, psm_config)

        # self.x_proj = nn.Sequential(
        #     nn.Linear(
        #         psm_config.embedding_dim, psm_config.embedding_dim // 2, bias=False
        #     ),
        #     nn.SiLU(),
        #     nn.Linear(psm_config.embedding_dim // 2, 26, bias=False),
        # )
        # self.aa_mask_head = nn.Sequential(
        #     nn.Linear(
        #         psm_config.embedding_dim, psm_config.embedding_dim // 2, bias=False
        #     ),
        #     nn.SiLU(),
        #     nn.Linear(psm_config.embedding_dim // 2, 30, bias=False),
        # )

        self.decoder = MSADiffusionModule(args, psm_config)

    def forward(
        self,
        batched_data,
        **unused,
    ):
        """
        Forward pass for PSM. This first computes the token

        Args:
            - batched_data: keys need to be defined in the data module
        Returns:
            - need to be defined
        """
        # token_embedding = self.embedding(
        #     batched_data["token_type"],
        #     batched_data["aa_mask"],
        #     batched_data["padding_mask"],
        # )

        # encoder_x = self.encoder(
        #     token_embedding.transpose(0, 1),
        #     batched_data["padding_mask"],
        #     batched_data,
        # )
        if self.psm_config.diffusion_mode == "OADM":
            msa_embedding = self.embedding(batched_data["128_msa_token_type"])
        # decoder_x = self.x_proj(encoder_x)
        else:
            msa_embedding = batched_data["128_msa_one_hot"].clone()

        noise_pred, mutation_pred = self.decoder(
            batched_data,
            msa_embedding,
            # encoder_x.transpose(0, 1)
            # .unsqueeze(1)
            # .repeat(1, msa_embedding.shape[1], 1, 1),
            batched_data["128_2D_padding_mask"],
            # batched_data["padding_mask"],
        )
        # print(decoder_x.shape)
        # x0_pred = (
        #     decoder_x.transpose(0, 1)
        #     .unsqueeze(1)
        #     .repeat(1, msa_embedding.shape[1], 1, 1)
        # )
        # noise_pred = F.softmax(noise_pred, dim=-1)
        # model_prob = F.softmax(decoder_x.transpose(0, 1), dim=-1)
        # model_log_prob = F.log_softmax(
        #     decoder_x.transpose(0, 1), dim=-1
        # )  # calculate kl loss which needs log softmax first
        # aa_logits = self.aa_mask_head(encoder_x)
        result_dict = {
            "noise_pred": noise_pred,
            "mutation_pred": mutation_pred,
            # "aa_logits": aa_logits.transpose(0, 1),
            # # "true_prob": batched_data["true_prob"],
            # "model_prob": model_prob,
            # "model_log_prob": model_log_prob,
            # "decoder_x": decoder_x.transpose(0, 1),
            "padding_mask": batched_data["padding_mask"],
        }
        return result_dict

    def init_state_dict_weight(self, weight, bias):
        """
        Initialize the state dict weight.
        """
        pass
