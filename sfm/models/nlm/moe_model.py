# -*- coding: utf-8 -*-

import os

from torch.optim import AdamW

from sfm.criterions.lm_moe import LmMoeCriterion
from sfm.models.nlm.moe_config import MoeModelConfig
from sfm.models.nlm.moe_modules import (
    MoeDecoderLayerPP,
    MoeEmbeddingsPP,
    MoeHeadPP,
    MoeNormPP,
)
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.pipeline_module import SFMPipelineModelMixin
from sfm.utils import PretrainedLayerSpec
from sfm.utils.optim.optimizer import myAdamW
from sfm.utils.optim.set_lr import DECAY_COSINE_RATE, groupWarmupDecayLR


def adam_w(*args, **kwargs):
    return AdamW(*args, **kwargs, fused=True)


class Model(SFMPipelineModelMixin):
    def __init__(self, config: MoeModelConfig):
        super().__init__()
        self.config = config
        self.loss = LmMoeCriterion(config)

    def to_layers(self):
        layers = []
        ckpt_folder = self.config.pretrained_ckpt_path

        pretrained_ckpt_path = os.path.join(ckpt_folder, "model.hybrid_emb.pt")
        if not os.path.exists(pretrained_ckpt_path):
            pretrained_ckpt_path = os.path.join(ckpt_folder, "layer_00-model_states.pt")

        layers.append(
            PretrainedLayerSpec(
                MoeEmbeddingsPP,
                self.config,
                new_num_tokens=self.config.vocab_size,
                pretrained_ckpt_path=pretrained_ckpt_path,
                load_ckpt=self.config.load_ckpt,
            )
        )

        for i in range(self.config.num_hidden_layers):
            pretrained_ckpt_path = os.path.join(ckpt_folder, f"model.layers.{i}.pt")

            if not os.path.exists(pretrained_ckpt_path):
                pretrained_ckpt_path = os.path.join(
                    ckpt_folder, f"layer_{i+1:02d}-model_states.pt"
                )

            layers.append(
                PretrainedLayerSpec(
                    MoeDecoderLayerPP,
                    self.config,
                    layer_idx=i,
                    pretrained_ckpt_path=pretrained_ckpt_path,
                    load_ckpt=self.config.load_ckpt,
                )
            )

        pretrained_ckpt_path = os.path.join(ckpt_folder, "model.norm.pt")
        if not os.path.exists(pretrained_ckpt_path):
            pretrained_ckpt_path = os.path.join(ckpt_folder, "layer_33-model_states.pt")

        layers.append(
            PretrainedLayerSpec(
                MoeNormPP,
                self.config,
                pretrained_ckpt_path=pretrained_ckpt_path,
                load_ckpt=self.config.load_ckpt,
            )
        )

        pretrained_ckpt_path = os.path.join(ckpt_folder, "model.lm_head.pt")
        if not os.path.exists(pretrained_ckpt_path):
            pretrained_ckpt_path = os.path.join(ckpt_folder, "layer_34-model_states.pt")

        layers.append(
            PretrainedLayerSpec(
                MoeHeadPP,
                self.config,
                new_num_tokens=self.config.vocab_size,
                pretrained_ckpt_path=pretrained_ckpt_path,
                load_ckpt=self.config.load_ckpt,
            )
        )

        return layers

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        logits, gate_logits = model_output

        bs = logits.shape[0]
        output = self.loss(logits, batch_data, gate_logits)
        loss = output[0]

        if len(output) > 1:
            log_loss = output[1]
        else:
            log_loss = {}
        return ModelOutput(loss=loss, log_output=log_loss, num_examples=bs)

    def config_optimizer(self, model):
        if model is None:
            model = self

        optimizer, _ = myAdamW(
            model,
            impl=adam_w,
            lr=self.config.max_lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
            eps=1e-8,
            freeze_list=self.config.freeze_param_list,
            unfreeze_list=self.config.unfreeze_param_list,
        )

        lr_scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=self.config.total_num_steps,
            warmup_max_lr=self.config.max_lr,
            warmup_num_steps=self.config.warmup_num_steps,
            decay_type=DECAY_COSINE_RATE,
        )

        return optimizer, lr_scheduler
