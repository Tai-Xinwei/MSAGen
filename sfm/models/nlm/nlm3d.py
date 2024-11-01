# -*- coding: utf-8 -*-
import os
from typing import Optional, Tuple

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.models.llama.configuration_llama import LlamaConfig

from megatron.core import parallel_state, tensor_parallel
from megatron.model.enums import AttnMaskType
from sfm.logging import logger
from sfm.models.llama2.llama2mp_config import MPLlamaConfig
from sfm.models.llama2.llama_modules import LlamaDecoderLayerPP, LlamaHead, LlamaNorm
from sfm.models.llama2.llama_modules_3dmp import (
    FusedLlamaNorm,
    LlamaDecoderLayerMP,
    LlamaHeadMP,
    LlamaLLMEmbeddingsMP,
)
from sfm.models.scigpt.modules import SciGPTEmbeddingsPP

try:
    from sfm.models.llama2.llama_modules_3dmp_te import (
        TELlamaDecoderLayerMP,
        TELlamaModel,
    )
except ImportError:
    logger.info("TE not installed")

from sfm.models.nlm.moduels.autoregressive3d import (
    AutoregressiveCriterion,
    AutoregressiveThreeDCriterion,
)
from sfm.pipeline.accelerator.dataclasses import ModelOutput, TrainStrategy
from sfm.pipeline.accelerator.pipeline_module import SFMPipelineModelMixin
from sfm.utils import PretrainedLayerSpec
from sfm.utils.optim.optimizer import myAdam, myAdamW
from sfm.utils.optim.set_lr import DECAY_COSINE_RATE, groupWarmupDecayLR


class NLM3dModel(SFMPipelineModelMixin):
    def __init__(self, args, vocab_size: int, if_init=True):
        super().__init__()

        if if_init:
            llama_config = LlamaConfig.from_pretrained(args.dict_path)
            self.args = args
            vocab_size = (
                max(args.padded_vocab_size, vocab_size)
                if hasattr(args, "padded_vocab_size")
                else vocab_size
            )
            llama_config.vocab_size = vocab_size
            self.mp_config = self.init_mp_config(args, llama_config)

            if args.strategy == TrainStrategy.ThreeD:
                self.loss_fn = AutoregressiveThreeDCriterion(self.mp_config)
            elif args.strategy == TrainStrategy.Pipeline:
                raise Exception("Use ThreeD strategy for pipeline training.")
            else:
                self.net = TELlamaModel(args, self.mp_config)
                self.loss_fn = AutoregressiveCriterion(self.mp_config)

    def to_layers(self):
        pipe_layer = []

        # determine whether checkpoint exit:
        pretrained_ckpt_path = os.path.join(
            self.args.pretrained_ckpt_path, "layer_00-model_00-model_states.pt"
        )
        if os.path.exists(pretrained_ckpt_path):
            prefix1 = "layer_"
            frefix2 = "-model_00-model_states.pt"
        else:
            prefix1 = "layer_"
            frefix2 = "-model_states.pt"

        pipe_layer.extend(
            [
                PretrainedLayerSpec(
                    SciGPTEmbeddingsPP,
                    self.mp_config,
                    learnable_cutoff=self.args.learnable_cutoff,
                    new_num_tokens=self.mp_config.vocab_size,
                    load_ckpt=self.args.load_ckpt,
                    pretrained_ckpt_path=os.path.join(
                        self.args.pretrained_ckpt_path, f"{prefix1}00{frefix2}"
                    ),
                    lora_mode="full",
                    # tp_model_size=self.args.tensor_model_parallel_size,
                    # tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                )
            ]
        )

        for i in range(self.mp_config.num_hidden_layers):
            pipe_layer.append(
                PretrainedLayerSpec(
                    LlamaDecoderLayerPP,
                    self.mp_config,
                    i,
                    load_ckpt=self.args.load_ckpt,
                    pretrained_ckpt_path=os.path.join(
                        self.args.pretrained_ckpt_path,
                        f"{prefix1}{str(i+1).zfill(2)}{frefix2}",
                    ),
                    lora_mode="full",
                    # tp_model_size=self.args.tensor_model_parallel_size,
                    # tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                    # self_attn_mask_type=AttnMaskType.causal,
                )
            )

        pipe_layer.append(
            PretrainedLayerSpec(
                LlamaNorm,
                self.mp_config,
                load_ckpt=self.args.load_ckpt,
                pretrained_ckpt_path=os.path.join(
                    self.args.pretrained_ckpt_path,
                    f"{prefix1}{str(self.mp_config.num_hidden_layers+1)}{frefix2}",
                ),
                lora_mode="full",
                # tp_model_size=self.args.tensor_model_parallel_size,
                # tp_rank=parallel_state.get_tensor_model_parallel_rank(),
            )
        )
        pipe_layer.append(
            PretrainedLayerSpec(
                LlamaHead,
                self.mp_config,
                learnable_cutoff=self.args.learnable_cutoff,
                new_num_tokens=self.mp_config.vocab_size,
                load_ckpt=self.args.load_ckpt,
                pretrained_ckpt_path=os.path.join(
                    self.args.pretrained_ckpt_path,
                    f"{prefix1}{str(self.mp_config.num_hidden_layers+2)}{frefix2}",
                ),
                lora_mode="full",
                # tp_model_size=self.args.tensor_model_parallel_size,
                # tp_rank=parallel_state.get_tensor_model_parallel_rank(),
            )
        )

        return pipe_layer

    def compute_loss(self, model_output, label) -> ModelOutput:
        loss, log_loss = self.loss_fn(model_output, label)
        bs = model_output[0].shape[0]

        return ModelOutput(loss=loss, log_output=log_loss, num_examples=bs)

    def config_optimizer(
        self, model=None
    ) -> Tuple[Optional[Optimizer], Optional[LRScheduler]]:
        # return (None, None)

        if model is None:
            model = self

        optimizer, _ = myAdam(
            model,
            unfreeze_list=self.args.unfreeze_param_list,
            lr=self.args.max_lr,
            betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.weight_decay,
            eps=1e-8,
        )

        lr_scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=self.args.total_num_steps,
            warmup_max_lr=self.args.max_lr,
            warmup_num_steps=self.args.warmup_num_steps,
            decay_type=DECAY_COSINE_RATE,
        )
        return (optimizer, lr_scheduler)

    def init_mp_config(self, args, llama_config):
        config = MPLlamaConfig(
            args,
            vocab_size=llama_config.vocab_size,
            hidden_size=llama_config.hidden_size,
            intermediate_size=llama_config.intermediate_size,
            num_hidden_layers=llama_config.num_hidden_layers,
            num_attention_heads=llama_config.num_attention_heads,
            num_key_value_heads=llama_config.num_key_value_heads,
            hidden_act=llama_config.hidden_act,
            max_position_embeddings=llama_config.max_position_embeddings,
            initializer_range=llama_config.initializer_range,
            rms_norm_eps=llama_config.rms_norm_eps,
            use_cache=llama_config.use_cache,
            pad_token_id=llama_config.pad_token_id,
            bos_token_id=llama_config.bos_token_id,
            eos_token_id=llama_config.eos_token_id,
            pretraining_tp=llama_config.pretraining_tp,
            tie_word_embeddings=llama_config.tie_word_embeddings,
            rope_scaling=llama_config.rope_scaling,
            # rope_theta=llama_config.rope_theta,
            rope_theta=args.rope_theta,
            seq_length=args.seq_length,
            rotary_percent=args.rotary_percent,
        )
        logger.info(f"true rope_theta:{args.rope_theta}")
        return config

    def forward(self, input_tuple: Tuple[torch.Tensor, torch.Tensor]):
        input_ids, attention_mask = input_tuple[0]
        if self.args.strategy not in [TrainStrategy.ThreeD, TrainStrategy.Pipeline]:
            return self.net(input_ids, attention_mask)
        else:
            raise NotImplementedError
