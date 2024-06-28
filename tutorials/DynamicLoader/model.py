# -*- coding: utf-8 -*-

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from sfm.logging import logger
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model


class TutModel(Model):
    """
    Class for tutorial training
    """

    def __init__(
        self,
        args,
        not_init=False,
    ):
        super().__init__()
        if not_init:
            return
        self.args = args
        if args.rank == 0:
            logger.info(self.args)

        # Load pre-trained model configuration
        config = GPT2Config.from_pretrained("gpt2")

        self.net = SimpleTransformer(config)

    def forward(self, batched_data, **kwargs):
        return self.net(batched_data, **kwargs)

    def compute_loss(self, pred, batch) -> ModelOutput:
        output = (str(pred.loss.item()),)
        log_loss = ""
        if len(output) > 1:
            log_loss = output

        bs = len(batch)

        return ModelOutput(
            loss=pred.loss,
            log_output=log_loss,
            num_examples=bs,
        )

    # required class, can be empty
    def config_optimizer(self):
        return (None, None)


# Define a simple Transformer model class
class SimpleTransformer(nn.Module):
    def __init__(self, config):
        super(SimpleTransformer, self).__init__()
        self.transformer = GPT2LMHeadModel(config)

    def forward(self, input_ids, labels=None):
        outputs = self.transformer(input_ids, labels=input_ids)
        return outputs
