# -*- coding: utf-8 -*-
# Copyright (c) Mircrosoft.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from sfm.logging import logger
from sfm.models.psm.equivariant.equiformer_series import Equiformerv2SO2
from sfm.models.psm.equivariant.equivariant import EquivariantDecoder
from sfm.models.psm.equivariant.geomformer import EquivariantVectorOutput
from sfm.models.psm.equivariant.nodetaskhead import NodeTaskHead, VectorOutput
from sfm.models.psm.invariant.invariant_encoder import PSMEncoder
from sfm.models.psm.invariant.plain_encoder import PSMPlainEncoder
from sfm.models.psm.modules.embedding import PSMMixEmbedding
from sfm.models.psm.modules.mixembedding import PSMMix3dEmbedding
from sfm.models.psm.modules.mixembedding_equiv import PSMMix3DEquivEmbedding
from sfm.models.psm.modules.pbc import CellExpander
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psmmodel import PSMModel, center_pos, complete_cell
from sfm.modules.layer_norm import AdaNorm
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model

from .modules.autograd import GradientHead
from .modules.dataaug import uniform_random_rotation
from .modules.diffusion import DIFFUSION_PROCESS_REGISTER
from .modules.sampled_structure_converter import SampledStructureConverter
from .modules.timestep_encoder import DiffNoise, TimeStepSampler


class PSMModel_RL(PSMModel):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(
        self,
        args,
        loss_fn=None,
        reward_model=None,
        not_init=False,
        psm_finetune_head: nn.Module = None,
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

        super().__init__(args, loss_fn, not_init, psm_finetune_head)

        self.net_old = copy.deepcopy(self.net)
        for param in self.net_old.parameters():
            param.requires_grad = False
        self.net_old.eval()
        self.loss_fn = loss_fn(args)
        self.reward_fn = reward_model(args, reward_model=self.psm_config.reward_model)
        self.alternate_counter = 0

        self.diffusion_process2 = DIFFUSION_PROCESS_REGISTER[
            self.psm_config.diffusion_sampling_rl
        ](self.diffnoise.alphas_cumprod, self.psm_config)

        if (
            self.psm_config.reward_model == "plddt"
            and self.psm_config.finetune_module == "plddt_confidence_head"
        ):
            for param in self.psm_finetune_head.parameters():
                param.requires_grad = False
            self.psm_finetune_head.eval()

        self.value_head = (
            nn.Sequential(
                AdaNorm(self.psm_config.embedding_dim),
                nn.Linear(
                    self.psm_config.embedding_dim,
                    self.psm_config.embedding_dim,
                    bias=True,
                ),
                nn.SiLU(),
                nn.Linear(self.psm_config.embedding_dim, 1, bias=True),
            )
            if args.backbone in ["vanillatransformer", "vanillatransformer_equiv"]
            else nn.Sequential(
                nn.Linear(
                    self.psm_config.embedding_dim,
                    self.psm_config.embedding_dim,
                    bias=True,
                ),
                nn.SiLU(),
                nn.Linear(self.psm_config.embedding_dim, 1, bias=True),
            )
        )

    def compute_plddt_reward(self, batched_data):
        time_step = self.time_step_sampler.get_continuous_time_step(
            t=0,
            n_graph=batched_data["pos"].shape[0],
            device=batched_data["pos"].device,
            dtype=batched_data["pos"].dtype,
        )
        time_step = time_step.unsqueeze(-1).repeat(1, batched_data["pos"].shape[1])
        result_dict = self.net_old(batched_data, time_step=time_step)
        result_dict = self.psm_finetune_head(result_dict)
        return result_dict["mean_plddt"]

    def logprob(
        self,
        pos,
        next_pos,
        pred_noise,
        pred_noise_old,
        t,
        protein_mask,
        padding_mask,
        step=1,
    ):
        # p(x_{t-1} | x_t) ~ N[1 / sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - hat{alpha_t}) * pred_noise), beta_title]
        alpha_cummlative_product = self.diffusion_process.alpha_cummlative_product
        alpha_cummlative_product_t_1 = (
            self.diffusion_process.alpha_cummlative_product_t_1
        )

        hat_alpha_t = self.diffusion_process._extract(
            alpha_cummlative_product, t, pos.shape
        )
        hat_alpha_t_1 = self.diffusion_process._extract(
            alpha_cummlative_product_t_1, t, pos.shape
        )
        # hat_alpha_t_1 = torch.where(
        #     t == 0,
        #     torch.tensor(1.0).to(t.device),
        #     self.diffusion_process._extract(alpha_cummlative_product, t - 1, pos.shape),
        # )

        alpha_t = hat_alpha_t / hat_alpha_t_1
        beta_t = 1 - alpha_t

        beta_tilde_t = torch.where(
            t == 0,
            torch.tensor(0.0).to(t.device),
            ((1.0 - hat_alpha_t_1) / (1.0 - hat_alpha_t) * beta_t).sqrt(),
        )

        if self.psm_config.diffusion_sampling_rl == "ddpm":
            temp = beta_t / torch.sqrt(1 - hat_alpha_t)
            mean = (
                1
                / torch.sqrt(alpha_t).unsqueeze(-1).unsqueeze(-1)
                * (pos - temp.unsqueeze(-1).unsqueeze(-1) * pred_noise)
            )
            mean_old = (
                1
                / torch.sqrt(alpha_t).unsqueeze(-1).unsqueeze(-1)
                * (pos - temp.unsqueeze(-1).unsqueeze(-1) * pred_noise_old)
            )
            var = beta_tilde_t**2
        elif self.psm_config.diffusion_sampling_rl == "sde":
            beta_t = beta_t * step
            score = -pred_noise / (1.0 - hat_alpha_t).sqrt()
            score_old = -pred_noise_old / (1.0 - hat_alpha_t).sqrt()
            mean = (2 - (1.0 - beta_t).sqrt()) * pos + beta_t * (score)
            mean_old = (2 - (1.0 - beta_t).sqrt()) * pos + beta_t * (score_old)
            var = beta_t
        else:
            raise ValueError(
                f"diffusion_sampling_rl: {self.psm_config.diffusion_sampling_rl} not supported"
            )

        log_exp = -((next_pos - mean) ** 2) / (2 * var)
        log_exp_old = -((next_pos - mean_old) ** 2) / (2 * var)

        # TODO: remove log_constant
        log_prob = log_exp
        old_log_prob = log_exp_old
        kl = (pred_noise - pred_noise_old) ** 2

        # protein_mask = protein_mask.any(dim=-1).unsqueeze(-1)
        loss_mask = (protein_mask.any(dim=-1) | padding_mask).unsqueeze(-1)
        selected_count = (~loss_mask).sum(dim=-1).sum(dim=-1)

        log_prob = log_prob.masked_fill(loss_mask, 0.0)
        log_prob = log_prob.sum(dim=-1).sum(dim=-1)
        log_prob = log_prob / selected_count.float()
        log_prob[selected_count == 0] = 0.0

        old_log_prob = old_log_prob.masked_fill(loss_mask, 0.0)
        old_log_prob = old_log_prob.sum(dim=-1).sum(dim=-1)
        old_log_prob = old_log_prob / selected_count.float()
        old_log_prob[selected_count == 0] = 0.0

        kl = kl.masked_fill(loss_mask, 0.0)
        kl = kl.sum(dim=-1).sum(dim=-1)
        kl = kl / selected_count.float()
        kl[selected_count == 0] = 0.0

        # log_prob = log_prob.mean(dim=-1).mean(dim=-1)
        # old_log_prob = old_log_prob.mean(dim=-1).mean(dim=-1)
        # kl = kl.mean(dim=-1).mean(dim=-1)

        return log_prob, old_log_prob, kl

    def forward(self, batched_data, **kwargs):
        """
        Forward pass of the model.

        Args:
            batched_data: Input data for the forward pass.
            **kwargs: Additional keyword arguments.
        """
        self.alternate_counter += 1
        self.alternate_counter = self.alternate_counter % (
            self.psm_config.psm_value_step + 1
        )

        # 0. turn on all gradients
        for param in self.net.parameters():
            param.requires_grad = True
        for param in self.value_head.parameters():
            param.requires_grad = True

        # 1. sample x0
        batched_data["pos"] = self.sample(batched_data)["pred_pos"]
        batched_data["pred_pos"] = batched_data["pos"]
        # compute plddt reward if needed
        if (
            self.psm_config.reward_model == "plddt"
            and self.psm_config.finetune_module == "plddt_confidence_head"
        ):
            batched_data_copy = copy.deepcopy(batched_data)
            batched_data["plddt_reward"] = self.compute_plddt_reward(batched_data_copy)

        # 2. collect xt from x0
        # repeat ''perturbation_each_traj'' times
        for key, value in batched_data.items():
            if isinstance(value, torch.Tensor):
                batched_data[key] = torch.cat(
                    [value] * self.psm_config.perturbation_each_traj, dim=0
                )

        (
            clean_mask,
            aa_mask,
            time_step,
            noise,
            padding_mask,
        ) = self._pre_forward_operation(batched_data)

        # time_step[:, :] = time_step[0, 0]

        # 3. denoise from xt to xt-1
        # t = int(time_step[0, 0].cpu() * self.psm_config.num_timesteps)
        # if t > self.psm_config.num_timesteps_stepsize_rl:
        #     step = self.psm_config.num_timesteps_stepsize_rl
        # else:
        #     step = 1

        # batched_data["sqrt_one_minus_alphas_cumprod_t"] = self.diffnoise._extract(
        #     self.diffnoise.sqrt_one_minus_alphas_cumprod,
        #     (time_step * self.psm_config.num_timesteps).long(),
        #     batched_data["pos"].shape,
        # )

        t = (time_step[:, 0] * self.psm_config.num_timesteps).long()
        step = 1

        pred_noise_old = self.net_old(
            batched_data, time_step=time_step, padding_mask=padding_mask
        )["noise_pred"]

        next_pos, pred_noise, _, decoder_x_output, result_dict = self.sample_each_t(
            t,
            time_step,
            batched_data,
            n_graphs=batched_data["pos"].shape[0],
            padding_mask=padding_mask,
            step=step,
        )

        # 4. compute log prob of (xt, xt-1)
        log_prob, old_log_prob, kl = self.logprob(
            batched_data["pos"],
            next_pos,
            pred_noise,
            pred_noise_old,
            t,
            result_dict["protein_mask"],
            padding_mask,
            step=step,
        )

        # 5. compute reward
        result_dict["padding_mask"] = padding_mask
        if self.psm_config.reward_model == "plddt":
            reward_batch = batched_data["plddt_reward"]
        elif self.psm_config.reward_model in ["rmsd", "lddt"]:
            reward_batch = self.reward_fn(result_dict, batched_data)
        else:
            raise ValueError(
                f"reward_model: {self.psm_config.reward_model} not supported"
            )
        # decoder_x_output: [B, L, embedding_dim]
        result_dict["value_per_atom"] = (
            self.value_head(decoder_x_output).mean(dim=1).squeeze(-1).squeeze(-1)
        )
        if self.alternate_counter == 0:
            result_dict["value_per_atom"] = result_dict["value_per_atom"].detach()

        value_loss = torch.mean((result_dict["value_per_atom"] - reward_batch) ** 2)

        result_dict["log_prob"] = log_prob
        result_dict["old_log_prob"] = old_log_prob
        result_dict["kl"] = kl
        result_dict["value_loss"] = value_loss
        result_dict["reward"] = reward_batch
        result_dict["train_value_step"] = self.psm_config.psm_value_step

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

        model_output["alternate_counter"] = self.alternate_counter

        output = super().compute_loss(model_output, batched_data)
        if self.alternate_counter == 0:
            for param in self.value_head.parameters():
                param.requires_grad = False
        else:
            for param in self.net.parameters():
                param.requires_grad = False

        return output

    def config_optimizer(self):
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            tuple[Optimizer, LRScheduler]:
        """
        return (None, None)

    def sample_each_t(self, t, time_step, batched_data, n_graphs, padding_mask, step=1):
        """
        Sample method for diffussion model
        """

        model_output = self.net(
            batched_data, time_step=time_step, padding_mask=padding_mask
        )
        predicted_noise = model_output["noise_pred"]
        decoder_x_output = model_output["decoder_x_output"]
        epsilon = self.diffnoise.get_noise(
            batched_data["pos"],
            batched_data["non_atom_mask"],
            batched_data["is_periodic"],
        )

        batched_data["pos"] = self.diffusion_process2.sample_step_multi_t(
            batched_data["pos"],
            batched_data["init_pos"],
            predicted_noise,
            epsilon,
            t,
            stepsize=step,
        )
        batched_data["pos"] = complete_cell(batched_data["pos"], batched_data)
        # batched_data["pos"] = center_pos(
        #     batched_data, padding_mask
        # )  # centering to remove noise translation
        batched_data["pos"] = batched_data["pos"].detach()

        return (
            batched_data["pos"],
            predicted_noise,
            time_step,
            decoder_x_output,
            model_output,
        )
