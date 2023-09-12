# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sfm.logging import logger
from sfm.pipeline.accelerator.dataclasses import ModelOutput, TrainStrategy
from sfm.pipeline.accelerator.trainer import Model

from .modules.scigpt_modules import Scigpt, SciGPTModelPP
from .scigpt_config import ScigptConfig


class ScigptModel(Model):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(self, args, loss_fn=None, not_init=False):
        super().__init__()
        if not_init:
            return

        # config = LlamaConfig(**args)
        config = ScigptConfig(args)

        self.args = config.args
        if args.rank == 0:
            logger.info(self.args)

        self.loss = loss_fn(args)

        if (
            args.strategy != TrainStrategy.ThreeD
            and args.strategy != TrainStrategy.Pipeline
        ):
            self.net = Scigpt(config)
        elif args.strategy == TrainStrategy.Pipeline:
            self.pipe_layers = SciGPTModelPP.to_layers(
                args,
                config,
                learnable_cutoff=0,
                load_ckpt=args.load_ckpt,
            )
            self.loss = loss_fn(args)
        elif args.strategy == TrainStrategy.ThreeD:
            raise NotImplementedError

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
        output = self.loss(logits, batch_data)
        loss = output[0]
        if len(output) > 1:
            log_loss = output[1]
        return ModelOutput(loss=loss, log_output=log_loss, num_examples=bs)

    def to_layers(self):
        return self.pipe_layers

    def config_optimizer(self) -> tuple[Optional[Optimizer], Optional[LRScheduler]]:
        return (None, None)
