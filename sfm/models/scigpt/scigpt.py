# -*- coding: utf-8 -*-
import os
from typing import Optional, Tuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sfm.criterions.autoregressive import AutoregressiveCriterion
from sfm.models.llama2.llama_modules import LlamaDecoderLayerPP, LlamaHead, LlamaNorm
from sfm.models.scigpt.config import ScigptConfig
from sfm.models.scigpt.modules import SciGPTEmbeddingsPP
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.pipeline_module import SFMPipelineModelMixin
from sfm.utils import PretrainedLayerSpec
from sfm.utils.optim.optimizer import myAdam
from sfm.utils.optim.set_lr import groupWarmupDecayLR


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
        ckpt_folder = self.config.pretrained_ckpt_path

        layers.append(
            PretrainedLayerSpec(
                SciGPTEmbeddingsPP,
                self.config,
                new_num_tokens=self.config.vocab_size,
                learnable_cutoff=self.config.learnable_cutoff,
                pretrained_ckpt_path=os.path.join(ckpt_folder, "model.hybrid_emb.pt"),
                load_ckpt=self.config.load_ckpt,
            )
        )

        for i in range(self.config.num_hidden_layers):
            layers.append(
                PretrainedLayerSpec(
                    LlamaDecoderLayerPP,
                    self.config,
                    i,
                    pretrained_ckpt_path=os.path.join(
                        ckpt_folder, f"model.layers.{i}.pt"
                    ),
                    load_ckpt=self.config.load_ckpt,
                )
            )

        layers.append(
            PretrainedLayerSpec(
                LlamaNorm,
                self.config,
                pretrained_ckpt_path=os.path.join(ckpt_folder, "model.norm.pt"),
                load_ckpt=self.config.load_ckpt,
            )
        )
        layers.append(
            PretrainedLayerSpec(
                LlamaHead,
                self.config,
                new_num_tokens=self.config.vocab_size,
                learnable_cutoff=self.config.learnable_cutoff,
                pretrained_ckpt_path=os.path.join(ckpt_folder, "model.lm_head.pt"),
                load_ckpt=self.config.load_ckpt,
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

    def config_optimizer(
        self, model
    ) -> Tuple[Optional[Optimizer], Optional[LRScheduler]]:
        optimizer, _ = myAdam(
            model,
            lr=self.config.max_lr,
            betas=[self.config.beta1, self.config.beta2],
            weight_decay=self.config.weight_decay,  # bugbug, weight_decay is not used in the optimizer
            eps=1e-8,
        )

        lr_scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=self.config.total_num_steps,
            warmup_max_lr=self.config.max_lr,
            warmup_num_steps=self.config.warmup_num_steps,
        )
        return (optimizer, lr_scheduler)
