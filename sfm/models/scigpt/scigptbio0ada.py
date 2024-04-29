# -*- coding: utf-8 -*-
import os
from typing import Optional, Tuple

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from sfm.criterions.autoregressive import AutoregressiveCriterion
from sfm.logging import logger
from sfm.models.llama2.llama_modules import LlamaDecoderLayerPP, LlamaHead, LlamaNorm
from sfm.models.scigpt.config import ScigptConfig
from sfm.models.scigpt.modules import SciGPTBioEmbeddingsPP
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.pipeline_module import SFMPipelineModelMixin
from sfm.utils import PretrainedLayerSpec
from sfm.utils.optim.optimizer import myAdam, myAdamW
from sfm.utils.optim.set_lr import DECAY_COSINE_RATE, groupWarmupDecayLR


class Scigptbio0adaModel(SFMPipelineModelMixin):
    def __init__(self, config: ScigptConfig):
        super().__init__()
        self.config = config
        self.loss = AutoregressiveCriterion(config)
        if config.infer:
            llama_config = LlamaConfig.from_pretrained(config.llm_model_name_or_path)
            self.decoder = LlamaForCausalLM(llama_config)

    def to_layers(self):
        layers = []
        ckpt_folder = self.config.pretrained_ckpt_path

        pretrained_ckpt_path = os.path.join(ckpt_folder, "model.hybrid_emb.pt")
        if not os.path.exists(pretrained_ckpt_path):
            pretrained_ckpt_path = os.path.join(ckpt_folder, "layer_00-model_states.pt")

        layers.append(
            PretrainedLayerSpec(
                SciGPTBioEmbeddingsPP,
                self.config,
                new_num_tokens=self.config.vocab_size,
                learnable_cutoff=self.config.learnable_cutoff,
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
                    LlamaDecoderLayerPP,
                    self.config,
                    i,
                    pretrained_ckpt_path=pretrained_ckpt_path,
                    load_ckpt=self.config.load_ckpt,
                )
            )

        pretrained_ckpt_path = os.path.join(ckpt_folder, "model.norm.pt")
        if not os.path.exists(pretrained_ckpt_path):
            pretrained_ckpt_path = os.path.join(ckpt_folder, "layer_33-model_states.pt")

        layers.append(
            PretrainedLayerSpec(
                LlamaNorm,
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
                LlamaHead,
                self.config,
                new_num_tokens=self.config.vocab_size,
                learnable_cutoff=self.config.learnable_cutoff,
                pretrained_ckpt_path=pretrained_ckpt_path,
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
        self, model=None
    ) -> Tuple[Optional[Optimizer], Optional[LRScheduler]]:
        if model is None:
            model = self

        # unfreeze_list = []

        # if self.config.tune_new_emb:
        #     unfreeze_list = [
        #         "lm_head.weight",
        #         "embed_tokens.weight"
        #     ]

        optimizer, _ = myAdam(
            model,
            unfreeze_list=self.config.unfreeze_param_list,
            lr=self.config.max_lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
            eps=1e-8,
        )

        # if self.config.ft:
        #     optimizer, _ = myAdam(
        #         model,
        #         unfreeze_list=self.args.unfreeze_list,
        #         lr=self.config.max_lr,
        #         betas=(self.config.beta1, self.config.beta2),
        #         weight_decay=self.config.weight_decay,
        #         eps=1e-8,
        #     )
        # else:
        #     optimizer = AdamW(
        #         model.parameters(),
        #         lr=self.config.max_lr,
        #         betas=(self.config.beta1, self.config.beta2),
        #         weight_decay=self.config.weight_decay,
        #         eps=1e-8,
        #     )

        lr_scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=self.config.total_num_steps,
            warmup_max_lr=self.config.max_lr,
            warmup_num_steps=self.config.warmup_num_steps,
            decay_type=DECAY_COSINE_RATE,
        )
        return (optimizer, lr_scheduler)
