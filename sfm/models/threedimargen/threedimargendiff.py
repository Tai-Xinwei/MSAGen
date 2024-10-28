# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from calendar import c
from typing import Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sfm.logging import logger
from sfm.pipeline.accelerator.dataclasses import ModelOutput, TrainStrategy
from sfm.pipeline.accelerator.trainer import Model
from sfm.utils.optim.adam import AdamW
from sfm.utils.optim.set_lr import DECAY_COSINE_RATE, groupWarmupDecayLR

from .modules.threedimargendiff_modules import CrystalCriterions, ThreeDimARGenDiff


class ThreeDimARGenDiffModel(Model):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(self, config, loss_fn=None, not_init=False):
        super().__init__()
        if not_init:
            return

        self.config = config

        if loss_fn is not None:
            self.loss = loss_fn(config)
        else:
            self.loss = CrystalCriterions(config.vocab_size)

        if (
            config.strategy != TrainStrategy.ThreeD
            and config.strategy != TrainStrategy.Pipeline
        ):
            self.net = ThreeDimARGenDiff(config)

        elif config.strategy == TrainStrategy.Pipeline:
            # self.pipe_layers = ThreeDimARGenModelPP.to_layers(
            #     config,
            #     config,
            #     learnable_cutoff=0,
            #     load_ckpt=config.load_ckpt,
            # )
            raise NotImplementedError
        elif config.strategy == TrainStrategy.ThreeD:
            raise NotImplementedError

        # if os.path.exists(args.loadcheck_path):
        #    self.load_pretrained_weights(args, checkpoint_path=args.loadcheck_path)

    def load_pretrained_weights(self, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.
        """
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

    def max_positions(self):
        return self.net.max_positions

    def forward(self, batched_data, **kwargs):
        return self.net(**batched_data, **kwargs)

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        return self.loss(model_output, batch_data)

    def to_layers(self):
        return self.pipe_layers

    def config_optimizer(
        self, model: Optional[torch.nn.Module] = None
    ) -> Tuple[Optional[Optimizer], Optional[LRScheduler]]:
        if model is None:
            model = self

        optimizer = AdamW(
            model.parameters(),
            lr=self.config.max_lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
            eps=1e-8,
        )

        lr_scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=self.config.total_num_steps,
            warmup_max_lr=self.config.max_lr,
            warmup_num_steps=self.config.warmup_num_steps,
            decay_type=DECAY_COSINE_RATE,
        )
        return (optimizer, lr_scheduler)
