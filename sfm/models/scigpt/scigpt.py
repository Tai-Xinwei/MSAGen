# -*- coding: utf-8 -*-
from typing import Optional, Tuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sfm.criterions.autoregressive import AutoregressiveCriterion
from sfm.models.llama2.llama_modules import LlamaDecoderLayerPP, LlamaHead
from sfm.models.scigpt.config import ScigptConfig
from sfm.models.scigpt.modules import SciGPTEmbeddingsPP
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.pipeline_module import SFMPipelineModelMixin
from sfm.utils import PretrainedLayerSpec


class ScigptModel(SFMPipelineModelMixin):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(self, config: ScigptConfig):
        super().__init__()
        self.config = config
        self.loss = AutoregressiveCriterion(config)

    def to_layers(self):
        layers = []
        layers.append(
            PretrainedLayerSpec(
                SciGPTEmbeddingsPP,
                self.config,
                learnable_cutoff=self.config.learnable_cutoff,
                load_ckpt=self.config.load_from_pretrained,
            )
        )

        for i in range(self.config.num_hidden_layers):
            layers.append(
                PretrainedLayerSpec(
                    LlamaDecoderLayerPP,
                    self.config,
                    i,
                    load_ckpt=self.config.load_from_pretrained,
                )
            )
        layers.append(
            PretrainedLayerSpec(
                LlamaHead,
                self.config,
                learnable_cutoff=self.config.learnable_cutoff,
                load_ckpt=self.config.load_from_pretrained,
            )
        )

        return layers

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        logits = model_output[0]
        bs = logits.shape[0]
        output = self.loss(logits, batch_data)
        loss = output[0]
        if len(output) > 1:
            log_loss = output[1]
        else:
            log_loss = {}
        return ModelOutput(loss=loss, log_output=log_loss, num_examples=bs)

    def config_optimizer(self) -> Tuple[Optional[Optimizer], Optional[LRScheduler]]:
        return (None, None)
