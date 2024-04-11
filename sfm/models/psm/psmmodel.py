# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import pickle as pkl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sfm.logging import logger
from sfm.models.psm.equivariant.equivariant import EquivariantDecoder
from sfm.models.psm.invariant.invariant_encoder import PSMEncoder
from sfm.models.psm.modules.embedding import PSMMixEmbedding
from sfm.models.psm.psm_config import PSMConfig
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.layer_norm import LayerNorm
from sfm.modules.quant_noise import quant_noise
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model

from .modules.timestep_encoder import DiffNoise, TimeStepEncoder, TimeStepSampler


class PSMModel(Model):
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
        """
        Initialize the TOXModel class.

        Args:
            args: Command line arguments.
            loss_fn: The loss function to use.
            data_mean: The mean of the data.
            data_std: The standard deviation of the data.
            not_init: If True, the model will not be initialized. Default is False.
            load_ckpt: If True, the model will load a checkpoint. Default is False.
        """

        super().__init__()
        if not_init:
            return
        self.psm_config = PSMConfig(args)
        self.args = self.psm_config.args
        if args.rank == 0:
            logger.info(self.args)

        # self.loss = loss_fn(args)

        self.net = PSM(args, self.psm_config)

        if load_ckpt:
            self.load_pretrained_weights(args, checkpoint_path=args.loadcheck_path)
        else:
            logger.info("No checkpoint is loaded")

        # Implement the Diffusion noise
        self.diffnoise = DiffNoise(self.psm_config)

        self.time_step_sampler = TimeStepSampler(self.psm_config.num_timesteps)

        self.energy_loss = nn.L1Loss(reduction="mean")
        self.force_loss = nn.L1Loss(reduction="none")
        self.noise_loss = nn.L1Loss(reduction="none")

    def _set_noise(
        self,
        ori_pos,
        padding_mask,
        batched_data,
        ori_angle=None,
        mask_pos=None,
        mask_angle=None,
        mode_mask=None,
        time_step=None,
        clean_mask=None,
        infer=False,
    ):
        """
        set diffusion noise here
        """

        n_graphs = ori_pos.size()[0]
        center_pos = get_center_pos(batched_data)
        ori_pos -= center_pos
        ori_pos = ori_pos.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        init_cell_pos = torch.zeros_like(ori_pos)
        init_cell_pos_input = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ]
            ],
            dtype=ori_pos.dtype,
            device=ori_pos.device,
        ).repeat([n_graphs, 1, 1]) * self.psm_config.diff_init_lattice_size - (
            self.psm_config.diff_init_lattice_size / 2.0
        )  # centering
        scatter_index = torch.arange(8, device=ori_pos.device).unsqueeze(0).unsqueeze(
            -1
        ).repeat([n_graphs, 1, 3]) + batched_data["num_atoms_in_cell"].unsqueeze(
            -1
        ).unsqueeze(
            -1
        )
        init_cell_pos = init_cell_pos.scatter(1, scatter_index, init_cell_pos_input)
        batched_data["init_pos"] = init_cell_pos.clone()

        noise_pos, noise, _ = self.diffnoise.noise_sample(
            ori_pos, time_step, x_init=batched_data["init_pos"], clean_mask=clean_mask
        )
        noise_pos = complete_cell(noise_pos, batched_data)

        return noise_pos, noise

    def load_pretrained_weights(self, args, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.

        Args:
            args: Command line arguments.
            checkpoint_path: Path to the pretrained weights.
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
        """
        Returns the maximum positions of the net.
        """
        return self.net.max_positions

    def forward(self, batched_data, **kwargs):
        """
        Forward pass of the model.

        Args:
            batched_data: Input data for the forward pass.
            **kwargs: Additional keyword arguments.
        """

        pos = batched_data["pos"]
        n_graphs = pos.size(0)
        time_step, clean_mask = self.time_step_sampler.sample(
            n_graphs, pos.device, pos.dtype, self.psm_config.clean_sample_ratio
        )

        token_id = batched_data["token_id"]
        padding_mask = token_id.eq(0)  # B x T x 1

        pos, noise = self._set_noise(
            ori_pos=pos,
            padding_mask=padding_mask,
            batched_data=batched_data,
            time_step=time_step,
            clean_mask=clean_mask,
        )
        batched_data["pos"] = pos
        result_dict = self.net(batched_data, time_step=time_step, **kwargs)
        result_dict["noise"] = noise
        result_dict["clean_mask"] = clean_mask

        return result_dict

    def ft_forward(self, batched_data, **kwargs):
        """
        Forward pass of the model during fine-tuning.

        Args:
            batched_data: Input data for the forward pass.
            **kwargs: Additional keyword arguments.
        """
        return self.net.ft_forward(batched_data, **kwargs)

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        """
        Compute loss for the model.

        Args:
            model_output: The output from the model.
            batch_data: The batch data.

        Returns:
            ModelOutput: The model output which includes loss, log_output, num_examples.
        """
        force_label = batch_data["forces"]
        energy_label = batch_data["y"]
        noise_label = model_output["noise"]
        force_pred = model_output["forces"]
        energy_pred = model_output["energy"]
        noise_pred = model_output["noise_pred"]
        atomic_numbers = batch_data["token_id"]
        atom_mask = model_output["atom_mask"]
        clean_mask = model_output["clean_mask"]

        n_graphs = energy_label.size()[0]
        if clean_mask is None:
            clean_mask = torch.zeros(
                n_graphs, dtype=torch.bool, device=energy_label.device
            )
        n_clean_graphs = torch.sum(clean_mask.to(dtype=torch.long))
        n_corrupted_graphs = n_graphs - n_clean_graphs
        padding_mask = atomic_numbers.eq(0)
        if n_clean_graphs > 0:
            energy_loss = self.energy_loss(
                energy_pred[clean_mask], energy_label[clean_mask]
            )
            force_loss = (
                self.force_loss(force_pred, force_label)
                .masked_fill(atom_mask.unsqueeze(-1), 0.0)
                .sum(dim=[1, 2])
                / (batch_data["num_atoms_in_cell"] * 3)
            )[clean_mask].mean()
        else:
            energy_loss = 0.0
            force_loss = 0.0

        if n_corrupted_graphs > 0:
            noise_loss = (
                self.noise_loss(noise_pred, noise_label)
                .masked_fill(padding_mask.unsqueeze(-1), 0.0)
                .sum(dim=[1, 2])
                / (
                    torch.sum(
                        (~padding_mask).to(dtype=energy_loss.dtype),
                        dim=-1,
                        keepdim=True,
                    )
                    * 3
                )
            )[~clean_mask].mean()
        else:
            noise_loss = 0.0
        loss = energy_loss + force_loss + noise_loss
        logging_output = {
            "loss": loss,
            "energy_loss": energy_loss,
            "force_loss": force_loss,
            "noise_loss": noise_loss,
        }
        return ModelOutput(
            loss=loss, num_examples=energy_label.size()[0], log_output=logging_output
        )

    def config_optimizer(self):
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            tuple[Optimizer, LRScheduler]:
        """
        pass

    @torch.no_grad()
    def sample(
        self,
        batch_data,
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
        Sample method for diffussion model
        """

        device = batch_data["x"].device

        n_graphs = batch_data["x"].shape[0]

        token_id = batch_data["token_id"]
        padding_mask = token_id.eq(0)  # B x T x 1

        center_pos = get_center_pos(batch_data)
        batch_data["pos"] -= center_pos
        batch_data["pos"] = batch_data["pos"].masked_fill(
            ~padding_mask.unsqueeze(-1), 0.0
        )
        orig_pos = batch_data["pos"].clone()

        init_cell_pos = torch.zeros_like(batch_data["pos"])
        init_cell_pos_input = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ]
            ],
            dtype=batch_data["pos"].dtype,
            device=device,
        ).repeat([n_graphs, 1, 1]) * self.psm_config.lattice_size - (
            self.psm_config.lattice_size / 2.0
        )  # centering
        scatter_index = torch.arange(8, device=batch_data["pos"].device).unsqueeze(
            0
        ).unsqueeze(-1).repeat([n_graphs, 1, 3]) + batch_data[
            "num_atoms_in_cell"
        ].unsqueeze(
            -1
        ).unsqueeze(
            -1
        )
        init_cell_pos = init_cell_pos.scatter(1, scatter_index, init_cell_pos_input)
        batch_data["init_pos"] = init_cell_pos.clone()

        pos_noise = torch.zeros(size=orig_pos.size(), device=device)
        pos_noise = pos_noise.normal_() * self.psm_config.diffusion_noise_std

        batch_data["pos"] = pos_noise + init_cell_pos
        batch_data["pos"] = batch_data["pos"].masked_fill(
            ~padding_mask.unsqueeze(-1), 0.0
        )
        batch_data["pos"] = complete_cell(batch_data["pos"], batch_data)

        if self.psm_config.diffusion_sampling == "ddpm":
            # Sampling from Step T-1 to Step 0
            for t in tqdm(range(self.psm_config.num_timesteps - 1, -1, -1)):
                hat_alpha_t = self.diffnoise.alphas_cumprod[t]
                hat_alpha_t_1 = 1.0 if t == 0 else self.diffnoise.alphas_cumprod[t - 1]
                alpha_t = hat_alpha_t / hat_alpha_t_1
                beta_t = 1 - alpha_t
                sigma_t = (
                    0.0
                    if t == 0
                    else ((1.0 - hat_alpha_t_1) / (1.0 - hat_alpha_t) * beta_t).sqrt()
                )

                # forward
                time_step = self.time_step_sampler.get_continuous_time_step(
                    t, n_graphs, device=device, dtype=batch_data["pos"].dtype
                )
                noise = self.net(batch_data, time_step=time_step)["noise_pred"]
                noise = noise.detach()

                epsilon = (
                    torch.zeros_like(batch_data["pos"]).normal_()
                    * self.psm_config.diffusion_noise_std
                )

                ext_pos = (
                    batch_data["pos"]
                    - init_cell_pos
                    - (1 - alpha_t) / (1 - hat_alpha_t).sqrt() * noise
                ) / alpha_t.sqrt() + sigma_t * epsilon

                batch_data["pos"] = ext_pos + init_cell_pos

                batch_data["pos"] = complete_cell(batch_data["pos"], batch_data)
                batch_data["pos"] = batch_data["pos"].detach()
                batch_data["pos"] = batch_data["pos"].masked_fill(
                    ~padding_mask.unsqueeze(-1), 0.0
                )
        elif self.psm_config.diffusion_sampling == "ddim":
            sampled_steps, _ = torch.sort(
                (
                    torch.randperm(
                        self.psm_config.num_timesteps - 2,
                        dtype=torch.long,
                        device=device,
                    )
                    + 1
                )[: self.psm_config.ddim_steps - 1]
            )
            sampled_steps = torch.cat(
                [
                    sampled_steps,
                    torch.tensor(
                        [self.psm_config.num_timesteps - 1], device=device
                    ).long(),
                ]
            )
            for i in tqdm(range(sampled_steps.shape[0] - 1, 0, -1)):
                t = sampled_steps[i]
                t_1 = sampled_steps[i - 1]
                hat_alpha_t = self.diffnoise.alphas_cumprod[t]
                hat_alpha_t_1 = self.diffnoise.alphas_cumprod[t_1]
                alpha_t = hat_alpha_t / hat_alpha_t_1
                beta_t = 1.0 - alpha_t
                sigma_t = (
                    self.psm_config.ddim_eta
                    * ((1.0 - hat_alpha_t_1) / (1.0 - hat_alpha_t) * beta_t).sqrt()
                )

                # forward
                time_step = self.time_step_sampler.get_continuous_time_step(
                    t, n_graphs, device=device, dtype=batch_data["pos"].dtype
                )
                noise = self.net(batch_data, time_step=time_step)["noise_pred"]
                ext_pos = batch_data["pos"] - init_cell_pos
                x_0_pred = (
                    ext_pos - (1.0 - hat_alpha_t).sqrt() * noise
                ) / hat_alpha_t.sqrt()
                epsilon = (
                    torch.zeros_like(ext_pos).normal_()
                    * self.psm_config.diffusion_noise_std
                )
                ext_pos = (
                    hat_alpha_t_1.sqrt() * x_0_pred
                    + (1.0 - hat_alpha_t_1 - sigma_t**2).sqrt()
                    * (ext_pos - hat_alpha_t.sqrt() * x_0_pred)
                    / (1.0 - hat_alpha_t).sqrt()
                    + sigma_t * epsilon
                )
                batch_data["pos"] = ext_pos + init_cell_pos
                batch_data["pos"] = complete_cell(batch_data["pos"], batch_data)
                batch_data["pos"] = batch_data["pos"].detach()
                batch_data["pos"] = batch_data["pos"].masked_fill(
                    ~padding_mask.unsqueeze(-1), 0.0
                )

            # forward for last step
            t = sampled_steps[0]
            hat_alpha_t = self.diffnoise.alphas_cumprod[t]

            # forward
            time_step = self.time_step_sampler.get_continuous_time_step(
                t, n_graphs, device=device, dtype=batch_data["pos"].dtype
            )
            noise = self.net(batch_data, time_step=time_step)["noise_pred"]
            ext_pos = batch_data["pos"] - init_cell_pos
            x_0_pred = (
                ext_pos - (1.0 - hat_alpha_t).sqrt() * noise
            ) / hat_alpha_t.sqrt()
            batch_data["pos"] = x_0_pred + init_cell_pos
            batch_data["pos"] = complete_cell(batch_data["pos"], batch_data)
            batch_data["pos"] = batch_data["pos"].detach()
            batch_data["pos"] = batch_data["pos"].masked_fill(
                ~padding_mask.unsqueeze(-1), 0.0
            )
        elif self.psm_config.diffusion_sampling == "ode":
            lattice_scatter_index = torch.tensor(
                [[[4], [2], [1]]], device=device, dtype=torch.long
            ).repeat([n_graphs, 1, 3])
            lattice_scatter_index += (
                batch_data["num_atoms_in_cell"].unsqueeze(-1).unsqueeze(-1)
            )
            for t in tqdm(range(self.psm_config.num_timesteps - 1, -1, -1)):
                time_step = self.time_step_sampler.get_continuous_time_step(
                    t, n_graphs, device=device, dtype=batch_data["pos"].dtype
                )
                beta_t = self.diffnoise.beta_list[t]
                noise = self.net(batch_data, time_step=time_step)["noise_pred"]
                score = -noise / (1.0 - self.diffnoise.alphas_cumprod[t]).sqrt()
                score = score.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
                pos = batch_data["pos"].clone() - init_cell_pos
                epsilon = torch.zeros_like(pos).normal_()

                batch_data["pos"] = (
                    (2 - (1.0 - beta_t).sqrt()) * pos
                    + 0.5 * beta_t * (score)
                    + init_cell_pos
                )
                batch_data["pos"] = complete_cell(batch_data["pos"], batch_data)
                batch_data["pos"] = batch_data["pos"].detach()
                batch_data["pos"] = batch_data["pos"].masked_fill(
                    ~padding_mask.unsqueeze(-1), 0.0
                )
        else:
            raise ValueError(
                f"Unknown diffusion sampling strategy {self.psm_config.diffusion_sampling}. Support only ddim and ddpm."
            )

        pred_pos = batch_data["pos"].clone()

        loss = torch.sum((pred_pos - orig_pos) ** 2, dim=-1, keepdim=True)

        return loss


def get_center_pos(batched_data):
    # get center of cell positions
    center = torch.sum(batched_data["cell"], dim=1, keepdim=True) / 2.0
    return center


def complete_cell(pos, batched_data):
    device = pos.device
    dtype = pos.dtype
    cell_matrix = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=dtype,
        device=device,
    )
    n_graphs, n_tokens = pos.size()[:2]
    gather_index = torch.tensor(
        [0, 4, 2, 1], device=device, dtype=torch.long
    ).unsqueeze(0).unsqueeze(-1).repeat([n_graphs, 1, 3]) + batched_data[
        "num_atoms_in_cell"
    ].unsqueeze(
        -1
    ).unsqueeze(
        -1
    )
    lattice = torch.gather(pos, 1, index=gather_index)
    corner = lattice[:, 0, :]
    lattice = lattice[:, 1:, :] - corner.unsqueeze(1)
    batched_data["cell"] = lattice
    cell = torch.matmul(cell_matrix, lattice) + corner.unsqueeze(1)
    scatter_index = torch.arange(8, device=device).unsqueeze(0).unsqueeze(-1).repeat(
        [n_graphs, 1, 3]
    ) + batched_data["num_atoms_in_cell"].unsqueeze(-1).unsqueeze(-1)
    pos = pos.scatter(1, scatter_index, cell)
    return pos


class PSM(nn.Module):
    """
    Class for training Physics science module
    """

    def __init__(self, args, psm_config: PSMConfig):
        super().__init__()
        self.max_positions = args.max_positions
        self.args = args

        # Implement the embedding
        self.embedding = PSMMixEmbedding(psm_config)

        # Implement the encoder
        self.encoder = PSMEncoder(args, psm_config)

        # Implement the decoder
        self.decoder = EquivariantDecoder(psm_config)

        # Implement the force and energy head
        # self.force_head = ...
        # self.energy_head = ...

        self.energy_head = nn.Linear(psm_config.embedding_dim, 1)

        self.psm_config = psm_config

    def _set_mask(self, mask_aa, mask_pos, residue_seq):
        """
        set mask here
        """
        pass

    def forward(
        self,
        batch_data,
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
        Forward pass for PSM. This first computes the token

        Args:
            - batch_data: keys need to be defined in the data module
        Returns:
            - need to be defined
        """

        pos = batch_data["pos"]
        n_graphs, n_nodes = pos.size()[:2]

        token_embedding, padding_mask, token_type = self.embedding(
            batch_data, time_step
        )

        encoder_output, pbc_expand_batched = self.encoder(
            token_embedding.transpose(0, 1), padding_mask, batch_data, token_type
        )
        decoder_x_output, decoder_vec_output, decoder_noise_pred_output = self.decoder(
            batch_data,
            encoder_output,
            batch_data["pos"],
            padding_mask,
            pbc_expand_batched,
        )
        energy = self.energy_head(decoder_x_output)
        energy = energy.squeeze(-1)
        atom_mask = torch.arange(
            n_nodes, dtype=torch.long, device=energy.device
        ).unsqueeze(0).repeat(n_graphs, 1) >= batch_data["num_atoms_in_cell"].unsqueeze(
            -1
        )
        energy = (
            energy.masked_fill(atom_mask, 0.0).sum(dim=-1)
            / batch_data["num_atoms_in_cell"]
        )
        forces = decoder_vec_output
        noise_pred = decoder_noise_pred_output
        return {
            "energy": energy,
            "forces": forces,
            "time_step": time_step,
            "noise_pred": noise_pred,
            "atom_mask": atom_mask,
        }

    def ft_forward(
        self,
        batch_data,
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
        forward function used in finetuning
        """
        pass

    def init_state_dict_weight(self, weight, bias):
        """
        Initialize the state dict weight.
        """
        pass
