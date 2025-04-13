# -*- coding: utf-8 -*-
# Copyright (c) Mircrosoft.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from contextlib import nullcontext
from typing import Optional

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
        self.cut_off = 64
        self.psm_config = PSMConfig(args)
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

        elif self.psm_config.diffusion_mode == "diff-lm":
            self.noise_loss = nn.CrossEntropyLoss(reduction="none")

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
        checkpoints_state = torch.load(checkpoint_path, map_location="cpu")
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
        batched_data["128_msa_token_type"] = batched_data["msa_token_type"][
            :, : self.cut_off, :
        ]
        ori_128_msa_one_hot = (
            F.one_hot(batched_data["128_msa_token_type"].long(), num_classes=27).float()
            * 2
            - 1
        )
        samples = []
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
                batched_data["token_type"].eq(0).unsqueeze(1).repeat(1, self.cut_off, 1)
            )
            print(padding_mask.shape)
            clean_mask = torch.zeros(
                B, self.cut_off, L, dtype=torch.bool, device=device
            )
            print(clean_mask.shape)
            clean_mask = clean_mask.masked_fill(padding_mask_2D, True)
            # set first to clean
            # clean_mask[:, 0, :] = True
            if clean_mask is not None:
                batched_data["128_msa_one_hot"] = torch.where(
                    clean_mask.unsqueeze(-1),
                    ori_128_msa_one_hot,
                    batched_data["128_msa_one_hot"],
                )
            batched_data["clean_mask"] = clean_mask
            # T = torch.full((B,), self.T - 1, device=device)
            # x_T = self.diffusion.q_sample(
            #     batched_data["128_msa_one_hot"], T, clean_mask, device
            # )
            batched_data["128_2D_padding_mask"] = padding_mask_2D
            # batched_data["128_msa_one_hot"] = x_T
            # batched_data["time_step"] = T
            true_prob = self.calculate_prob(batched_data["msa_token_type"])
            if self.psm_config.diffusion_mode == "epsilon":
                for t in range(
                    self.psm_config.num_timesteps - 1,
                    -1,
                    self.psm_config.num_timesteps_stepsize,
                ):
                    # forward
                    time_step = self.time_step_sampler.get_continuous_time_step(
                        t, B, device=device, dtype=batched_data["128_msa_one_hot"].dtype
                    )
                    time_step = (
                        time_step.unsqueeze(-1).unsqueeze(-1).repeat(1, self.cut_off, L)
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
                    epsilon = self.diffnoise.get_noise(batched_data["128_msa_one_hot"])
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
            else:
                # diff-lm
                for t in range(
                    self.psm_config.num_timesteps - 1,
                    -1,
                    self.psm_config.num_timesteps_stepsize,
                ):
                    # forward
                    time_step = self.time_step_sampler.get_continuous_time_step(
                        t, B, device=device, dtype=batched_data["128_msa_one_hot"].dtype
                    )
                    time_step = (
                        time_step.unsqueeze(-1).unsqueeze(-1).repeat(1, self.cut_off, L)
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
                        continue
                    else:
                        time_step_pre = self.time_step_sampler.get_continuous_time_step(
                            t - 1,
                            B,
                            device=device,
                            dtype=batched_data["128_msa_one_hot"].dtype,
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
                            batched_data["128_msa_one_hot"] = torch.where(
                                clean_mask.unsqueeze(-1),
                                ori_128_msa_one_hot,
                                batched_data["128_msa_one_hot"],
                            )
                        batched_data["128_msa_one_hot"] = batched_data[
                            "128_msa_one_hot"
                        ].detach()

            pred_msa = batched_data["128_msa_one_hot"].clone()

            # kl_loss=self.kl(x_t.argmax(dim=-1),batched_data["msa_token_type"])
            pred_prob = self.calculate_prob(pred_msa.argmax(dim=-1))
            samples.append(self.convert(pred_msa.argmax(dim=-1)))
        # self.net.train()
        plot_probability_heatmaps(true_prob, pred_prob, padding_mask, batched_data)
        # return torch.stack(samples, dim=0)

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
        return F.normalize(true_prob + 1e-5, dim=-1)

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
        B, D, L = batched_data["msa_token_type"].shape
        min_D = min(D, batched_data["cut_off"])
        device = batched_data["msa_token_type"].device
        batched_data["ori_128_msa_one_hot"] = batched_data["128_msa_one_hot"].clone()
        # time sample
        time_step, clean_mask = self.time_step_sampler.sample(
            B,
            device,
            batched_data["128_msa_one_hot"].dtype,
            self.psm_config.clean_sample_ratio,
        )
        clean_mask = clean_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, min_D, L)
        time_step = time_step.unsqueeze(-1).unsqueeze(-1).repeat(1, min_D, L)

        # t = torch.randint(0, 1000, (B,), device=device)

        # set padding to clean
        clean_mask = clean_mask.masked_fill(batched_data["128_2D_padding_mask"], True)

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
        # return x_t,t

    def _pre_forward_operation(
        self,
        batched_data,
    ):
        """
        pre forward operation
        """
        # set padding_mask
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
        mask_ratio = 0.15
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
        batched_data["true_prob"] = F.normalize(true_prob + 1e-5, dim=-1)
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
            self.sample(batched_data)

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

        kl_loss = F.kl_div(
            model_output["model_log_prob"], batched_data["true_prob"], reduction="none"
        )
        kl_loss = kl_loss.sum(dim=-1)
        mask = ~model_output["padding_mask"]
        kl_loss = (kl_loss * mask).sum() / mask.sum()
        # if batched_data["aa_mask"].any():
        aa_mask = batched_data["aa_mask"]
        logits = model_output["aa_logits"][aa_mask]
        aa_mlm_loss = self.aa_mlm_loss(
            logits,
            batched_data["token_type"][aa_mask].long(),
        )
        # print("aa_mlm_loss",aa_mlm_loss)
        noise_pred = model_output["noise_pred"]
        filter_mask = ~batched_data["clean_mask"]  # B D L
        B, D, L = batched_data["128_2D_padding_mask"].size()
        is_gap = batched_data["128_msa_token_type"] == 26
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
        elif self.psm_config.diffusion_mode == "diff-lm":
            noise_label = batched_data["128_msa_token_type"]
            diffusion_loss, diff_celoss, recons_loss = self.compute_diff_loss(
                noise_label,
                noise_pred,
                is_gap,
                1.0,
                "L2",
                batched_data,
                filter_mask,
                # batched_data["ori_128_msa_one_hot"].argmax(dim=-1).unsqueeze(-1).view(B,D*L,-1),
            )
            # print("diff_loss",diffusion_loss)
            # noise_pred means x0_pred

        batched_data["init_128_msa_one_hot"] = torch.zeros(
            B, D, L, 27, device=batched_data["msa_token_type"].device
        ).float()
        # batched_data["msa_token_type"].device
        # epsilon = self.diffnoise.get_noise(batched_data["128_msa_one_hot"])
        # t = (
        #     (batched_data["time_step"][0].float() * (self.psm_config.num_timesteps - 1))
        #     .long()
        #     .cpu()
        # )
        # pred_msa = self.diffusion_process.get_x0(
        #     batched_data["128_msa_one_hot"],
        #     batched_data["init_128_msa_one_hot"],
        #     noise_pred,
        #     epsilon,
        #     t,
        #     stepsize=-self.psm_config.num_timesteps_stepsize,
        # )
        # pred_msa = model_output["noise_pred"]
        # pred_ori = pred_msa[:, 0, :, :]
        # ori_ce_loss = self.ce_loss(
        #     pred_msa.contiguous().view(B*D,L,27).permute(0, 2, 1),
        #     batched_data["128_msa_token_type"].contiguous().view(B*D,L).long(),
        # )
        # ori_ce_loss = self.ce_loss(
        #     pred_msa[batched_data["128_2D_padding_mask"]],
        #     batched_data["128_msa_token_type"][batched_data["128_2D_padding_mask"]].long(),
        # )
        # print(pred_ori.shape,batched_data["token_type"].shape)
        # l1_loss = self.compute_l1_loss(
        #     model_output["x0_pred"], batched_data["ori_128_msa_one_hot"]
        # )
        loss = aa_mlm_loss + diffusion_loss + kl_loss
        # loss = ori_ce_loss
        # loss += cross_entropy_loss
        logging_output = {
            "total_loss": float(loss.detach()),
            "diffusion_loss": float(diffusion_loss.detach()),
            "diffusion_ce_loss": float(diff_celoss.detach()),
            "recons_loss": float(recons_loss.detach()),
            "KL_loss": float(kl_loss.detach()),
            "aa_mlm_loss": float(aa_mlm_loss.detach()),
            # "ori_ce_loss": float(ori_ce_loss.detach()),
        }

        return ModelOutput(
            loss=loss,
            num_examples=model_output["model_prob"].shape[0],
            log_output=logging_output,
        )

    def compute_diff_loss(
        self,
        label,
        pred,
        is_gap,
        gap_weight,
        loss_type="L1",
        batched_data=None,
        filter_mask=None,
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
        else:
            # diff-lm
            ce_loss = self.noise_loss(pred[filter_mask], label[filter_mask].long())
            differ_mask = ~(
                batched_data["128_msa_token_type"]
                == batched_data["token_type"].unsqueeze(1)
            )[
                filter_mask
            ]  # true means differ
            # if differ, enlarge the loss, except for gap
            differ_mask = differ_mask & ~is_gap[filter_mask]
            ce_loss = ce_loss * (1 + 4.0 * differ_mask.float())
            # 0.2 for gap
            ce_loss = ce_loss * (1 - 0.8 * is_gap[filter_mask].float())
            kl_loss = self._KL_reconstruction_loss(
                batched_data, pred, batched_data["ori_128_msa_one_hot"], filter_mask
            )
            loss = ce_loss.mean() + kl_loss
            return loss, ce_loss.mean(), kl_loss.mean()

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

        self.encoder = MSAGenEncoder(args, psm_config)

        self.x_proj = nn.Sequential(
            nn.Linear(
                psm_config.embedding_dim, psm_config.embedding_dim // 2, bias=False
            ),
            nn.SiLU(),
            nn.Linear(psm_config.embedding_dim // 2, 26, bias=False),
        )
        self.aa_mask_head = nn.Sequential(
            nn.Linear(
                psm_config.embedding_dim, psm_config.embedding_dim // 2, bias=False
            ),
            nn.SiLU(),
            nn.Linear(psm_config.embedding_dim // 2, 30, bias=False),
        )

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
        token_embedding = self.embedding(
            batched_data["token_type"],
            batched_data["aa_mask"],
            batched_data["padding_mask"],
        )

        encoder_x = self.encoder(
            token_embedding.transpose(0, 1),
            batched_data["padding_mask"],
            batched_data,
        )

        # msa_embedding = self.embedding(batched_data["128_msa_token_type"])
        decoder_x = self.x_proj(encoder_x)
        msa_embedding = batched_data["128_msa_one_hot"].clone()

        noise_pred = self.decoder(
            batched_data,
            msa_embedding,
            encoder_x.transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, msa_embedding.shape[1], 1, 1),
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
        model_prob = F.softmax(decoder_x.transpose(0, 1), dim=-1)
        model_log_prob = F.log_softmax(
            decoder_x.transpose(0, 1), dim=-1
        )  # calculate kl loss which needs log softmax first
        aa_logits = self.aa_mask_head(encoder_x)
        result_dict = {
            "noise_pred": noise_pred,
            "aa_logits": aa_logits.transpose(0, 1),
            # "true_prob": batched_data["true_prob"],
            "model_prob": model_prob,
            "model_log_prob": model_log_prob,
            "decoder_x": decoder_x.transpose(0, 1),
            "padding_mask": batched_data["padding_mask"],
        }
        return result_dict

    def init_state_dict_weight(self, weight, bias):
        """
        Initialize the state dict weight.
        """
        pass
