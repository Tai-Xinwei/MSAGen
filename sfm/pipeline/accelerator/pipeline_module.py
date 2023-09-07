# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from deepspeed.runtime.activation_checkpointing import checkpointing
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.model import Model
from sfm.utils.mypp_module import PipelineModule


class SFMPipelineModelMixin(Model, ABC):
    @abstractmethod
    def to_layers(self) -> List[Module]:
        pass


class SFMPipelineModule(PipelineModule, Model):
    def __init__(
        self,
        model: SFMPipelineModelMixin,
        loss_fn: Module,
        partition_method: str,
        num_stages: Optional[int] = None,
        loss_log_dict: Optional[dict] = {},
        part_list: Optional[List[int]] = None,
        topology=None,
    ):
        super().__init__(
            model.to_layers(),
            num_stages=num_stages,
            topology=topology,
            loss_fn=loss_fn,
            seed_layers=False,
            seed_fn=None,
            base_seed=1234,
            partition_method=partition_method,
            activation_checkpoint_interval=0,
            activation_checkpoint_func=checkpointing.checkpoint,
            checkpointable_layers=None,
            loss_log_dict=loss_log_dict,
            part_list=part_list,
        )
        self.model = model

    def compute_loss(self, pred, batch) -> ModelOutput:
        return self.model.compute_loss(pred, batch)

    def config_optimizer(self) -> Tuple[Optimizer, LRScheduler]:
        return self.model.config_optimizer()

    def before_training(self):
        self.model.before_training()

    def after_training(self):
        self.model.after_training()

    def before_batch(self):
        self.model.before_batch()

    def after_batch(self):
        self.model.after_batch()
