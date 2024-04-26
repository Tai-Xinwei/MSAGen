# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sfm.logging import logger
from sfm.pipeline.accelerator.dataclasses import ModelOutput, TrainStrategy
from sfm.pipeline.accelerator.trainer import Model
from sfm.utils.optim.adam import AdamW
from sfm.utils.optim.set_lr import DECAY_COSINE_RATE, groupWarmupDecayLR

from .modules.threedimargennum_modules import ThreeDimARGenModelPP, ThreeDimARGenNum


class ThreeDimARGenNumModel(Model):
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

        if (
            config.strategy != TrainStrategy.ThreeD
            and config.strategy != TrainStrategy.Pipeline
        ):
            self.net = ThreeDimARGenNum(config)
        elif config.strategy == TrainStrategy.Pipeline:
            self.pipe_layers = ThreeDimARGenModelPP.to_layers(
                config,
                config,
                learnable_cutoff=0,
                load_ckpt=config.load_ckpt,
            )
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
        word_logits = model_output[0]
        bs = word_logits.shape[0]
        coordinates = model_output[1]
        # shift so that tokens < n predict n
        label_ids = batch_data["label_ids"]
        label_coordinates = batch_data["label_coordinates"]
        coordinates_mask = batch_data["coordinates_mask"]

        shift_label_ids = label_ids[..., 1:].contiguous()
        shift_coordinates_mask = coordinates_mask[..., 1:].contiguous()
        shift_word_logits = word_logits[:, :-1, :].contiguous()
        shift_coordinates = coordinates[:, :-1, :].contiguous()
        shift_word_logits = shift_word_logits[~shift_coordinates_mask.bool()]
        shift_coordinates = shift_coordinates[shift_coordinates_mask.bool()]

        # Calculate loss on word tokens
        loss_words_fct = CrossEntropyLoss()
        shift_words_labels = shift_label_ids[~shift_coordinates_mask.bool()]
        loss_words = loss_words_fct(
            shift_word_logits.view(-1, self.args.vocab_size),
            shift_words_labels.view(-1),
        )
        # Calculate loss on coordinate tokens
        loss_coord_fct = MSELoss()
        if label_coordinates.dtype != shift_coordinates.dtype:
            label_coordinates = label_coordinates.to(coordinates.dtype)
        loss_coord = loss_coord_fct(shift_coordinates, label_coordinates)
        # Combine losses
        loss = loss_words + loss_coord
        loss_log = {
            "loss": loss.item() if loss is not None else None,
            "loss_words": loss_words.item() if loss_words is not None else None,
            "loss_coord": loss_coord.item() if loss_coord is not None else None,
        }

        return ModelOutput(loss=loss, log_output=loss_log, num_examples=bs)

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
