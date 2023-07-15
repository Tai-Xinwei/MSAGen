# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sfm.pipeline.dataclasses import ModelOutput


class Model(nn.Module, ABC):
    @abstractmethod
    def compute_loss(self, pred, batch) -> ModelOutput:
        pass

    def load_pretrained(self):
        """
        Load all pretrained weights, e.g., pretrained encoders or decoders.
        """
        pass

    @abstractmethod
    def config_optimizer(self) -> tuple[Optimizer, LRScheduler]:
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            tuple[Optimizer, LRScheduler]:
        """
        pass
