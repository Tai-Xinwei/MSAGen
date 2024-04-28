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
from sfm.models.llama2.llama_modules_3dmp import (
    FusedLlamaNorm,
    LlamaDecoderLayerMP,
    LlamaHeadMP,
    LlamaLLMEmbeddingsMP,
)
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.pipeline_module import SFMPipelineModelMixin
from sfm.utils import PretrainedLayerSpec
from sfm.utils.optim.optimizer import myAdam, myAdamW
from sfm.utils.optim.set_lr import DECAY_COSINE_RATE, groupWarmupDecayLR


class NLM3dModel(SFMPipelineModelMixin):
    def __init__(self, args, vocab_size: int):
        super().__init__()

        llama_config = LlamaConfig.from_pretrained(args.pretrained_ckpt_path)
        args.padded_vocab_size = max(args.padded_vocab_size, vocab_size)
        vocab_size = args.padded_vocab_size
        llama_config.vocab_size = vocab_size
        self.mp_config = self.init_mp_config(args, llama_config)
        self.llama_config = self.mp_config
        self.args = args

    def to_layers(self):
        pipe_layer = []

        pipe_layer.extend(
            [
                PretrainedLayerSpec(
                    LlamaLLMEmbeddingsMP,
                    self.mp_config,
                    new_num_tokens=self.mp_config.vocab_size,
                    load_ckpt=self.args.load_ckpt,
                    pretrained_ckpt_path=os.path.join(
                        self.args.pretrained_ckpt_path, "layer_00-model_states.pt"
                    ),
                    lora_mode="full",
                    tp_model_size=self.args.tensor_model_parallel_size,
                    tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                )
            ]
        )

        for i in range(self.mp_config.num_hidden_layers):
            pipe_layer.append(
                PretrainedLayerSpec(
                    LlamaDecoderLayerMP,
                    self.mp_config,
                    i,
                    load_ckpt=self.args.load_ckpt,
                    pretrained_ckpt_path=os.path.join(
                        self.args.pretrained_ckpt_path,
                        f"layer_{str(i+1).zfill(2)}-model_states.pt",
                    ),
                    lora_mode="full",
                    tp_model_size=self.args.tensor_model_parallel_size,
                    tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                    self_attn_mask_type=AttnMaskType.causal,
                )
            )
        pipe_layer.append(
            PretrainedLayerSpec(
                FusedLlamaNorm,
                self.mp_config,
                load_ckpt=self.args.load_ckpt,
                pretrained_ckpt_path=os.path.join(
                    self.args.pretrained_ckpt_path, "layer_33-model_states.pt"
                ),
                lora_mode="full",
                tp_model_size=self.args.tensor_model_parallel_size,
                tp_rank=parallel_state.get_tensor_model_parallel_rank(),
            )
        )
        pipe_layer.append(
            PretrainedLayerSpec(
                LlamaHeadMP,
                self.mp_config,
                new_num_tokens=self.mp_config.vocab_size,
                load_ckpt=self.args.load_ckpt,
                pretrained_ckpt_path=os.path.join(
                    self.args.pretrained_ckpt_path, "layer_34-model_states.pt"
                ),
                lora_mode="full",
                tp_model_size=self.args.tensor_model_parallel_size,
                tp_rank=parallel_state.get_tensor_model_parallel_rank(),
            )
        )

        return pipe_layer

    def compute_loss(self, model_output, label) -> ModelOutput:
        if type(label) == tuple:
            labels = label[0]
            loss_mask = label[1]
        else:
            labels = label
            loss_mask = torch.ones_like(labels).bool()

        loss_mask = loss_mask[..., 1:].contiguous().transpose(0, 1)

        logits = model_output[0]
        bs = logits.shape[0]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # transpose and contiguous to enable model parallelism
        shift_logits = shift_logits.transpose(0, 1).contiguous()
        shift_labels = shift_labels.transpose(0, 1).contiguous()
        shift_labels = shift_labels.to(shift_logits.device)
        loss = tensor_parallel.vocab_parallel_cross_entropy(shift_logits, shift_labels)

        instruct_mask = shift_labels != -100

        loss_mask = instruct_mask & loss_mask
        # logger.info(f"{loss_mask}")
        loss_mask = loss_mask.view(-1)

        loss = loss.contiguous().view(-1)
        if loss_mask.sum() == 0:
            loss = torch.tensor(0.0).to(loss.device)
        else:
            loss = loss.masked_fill_(~loss_mask, 0.0)
            loss = torch.sum(loss) / (loss_mask).sum()

        log_loss = {"lm_loss": loss}

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
            seq_length=args.seq_length,
            rotary_percent=args.rotary_percent,
        )

        return config
