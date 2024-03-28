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
from sfm.models.psm.modules.equivariant import EquivariantDecoder
from sfm.models.psm.modules.invariant_encoder import PSMEncoder
from sfm.models.psm.psm_config import PSMConfig
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.layer_norm import LayerNorm
from sfm.modules.quant_noise import quant_noise
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model

from .modules.timestep_encoder import DiffNoise


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
        pfm_config = PSMConfig(args)
        self.args = pfm_config.args
        if args.rank == 0:
            logger.info(self.args)

        self.loss = loss_fn(args)

        self.net = PSM(args, pfm_config)

        if load_ckpt:
            self.load_pretrained_weights(args, checkpoint_path=args.loadcheck_path)
        else:
            logger.info("No checkpoint is loaded")

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
        return self.net(batched_data, **kwargs)

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
        raise NotImplementedError

    def config_optimizer(self):
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            tuple[Optimizer, LRScheduler]:
        """
        pass


class PSM(nn.Module):
    """
    Class for training Physics science module
    """

    def __init__(self, args, psm_config):
        super().__init__()
        self.max_positions = args.max_positions
        self.args = args

        # Implement the encoder
        self.encoder = PSMEncoder(psm_config)

        # Implement the decoder
        self.decoder = EquivariantDecoder(psm_config)

        # Implement the Diffusion noise
        self.diffnoise = DiffNoise(psm_config)

        # Implement the force and energy head
        # self.force_head = ...
        # self.energy_head = ...

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
        """
        set diffusion noise here
        """
        pass

    def _set_mask(self, mask_aa, mask_pos, residue_seq):
        """
        set mask here
        """
        pass

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
        Forward pass for PSM. This first computes the token

        Args:
            - batched_data: keys need to be defined in the data module
        Returns:
            - need to be defined
        """
        pass

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
        Sample method for diffussion model
        """

        pass

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
        forward function used in finetuning
        """
        pass

    def init_state_dict_weight(self, weight, bias):
        """
        Initialize the state dict weight.
        """
        pass
