# -*- coding: utf-8 -*-
import os
from typing import Optional, Tuple

import torch
import transformer_engine as te
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformer_engine.pytorch.attention import RotaryPositionEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaRMSNorm

from megatron.core import parallel_state
from sfm.logging import logger
from sfm.models.nlm.moduels.autoregressive3d import AutoregressiveThreeDCriterion
from sfm.models.nlm.nlm3d import NLM3dModel
from sfm.pipeline.accelerator.dataclasses import ModelOutput, TrainStrategy
from sfm.utils.optim.optimizer import myAdam, myAdamW
from sfm.utils.optim.set_lr import DECAY_COSINE_RATE, groupWarmupDecayLR


class GenegptModel(NLM3dModel):
    def __init__(self, args, config, vocab_size: int, infer: bool = False):
        super().__init__(args, vocab_size)

        self.args = args
        self.mp_config = self.init_mp_config(args, config)
        if infer:
            self.net = NLMBaseCausalLM(args, self.mp_config)
        # print(config)
        # print(self.net)
        # self.loss_fn = AutoregressiveCriterion(self.mp_config)
        self.loss_fn = AutoregressiveThreeDCriterion(self.mp_config)

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

    def init_config(self, args, llama_config, geneconfig, vocab_size):
        llama_config.hidden_size = args.hidden_size
        llama_config.intermediate_size = args.intermediate_size
        llama_config.num_hidden_layers = args.num_hidden_layers
        llama_config._attn_implementation = "sdpa"
        llama_config.vocab_size = vocab_size
        llama_config.torch_dtype = "bfloat16" if args.bf16 else "float32"
        llama_config.bos_token_id = geneconfig.bos_token_id
        llama_config.eos_token_id = geneconfig.eos_token_id

        return llama_config

    def forward(self, input_tuple: Tuple[torch.Tensor, torch.Tensor]):
        input_ids, attention_mask = input_tuple[0]
        if self.args.strategy not in [TrainStrategy.ThreeD, TrainStrategy.Pipeline]:
            return self.net(input_ids, attention_mask)
        else:
            raise NotImplementedError


class TELlamaDecoderLayer(te.pytorch.TransformerLayer):
    def __init__(
        self,
        args,
        layer_idx: int = 0,
        **kwargs,
    ):
        if args.fp16:
            logger.info("Using fp16 in transformer layer")
            params_dtype = torch.float16
        elif args.bf16:
            logger.info("Using bf16 in transformer layer")
            params_dtype = torch.bfloat16

        if args.tensor_model_parallel_size > 1:
            tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            tp_group = None

        super().__init__(
            args.hidden_size,
            args.intermediate_size,
            args.num_attention_heads,
            bias=False,
            layernorm_epsilon=args.rms_norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=False,
            normalization="RMSNorm",
            activation="swiglu",
            attn_input_format="bshd",
            num_gqa_groups=args.num_key_value_heads,
            params_dtype=params_dtype,
            tp_size=args.tensor_model_parallel_size,
            set_parallel_mode=True if args.tensor_model_parallel_size > 1 else False,
            tp_group=tp_group,
        )
        te_rope = RotaryPositionEmbedding(args.hidden_size // args.num_attention_heads)
        self.te_rope_emb = te_rope(max_seq_len=args.max_position_embeddings).cuda()

    def forward(self, hidden_states, attention_mask, **kwargs):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `LlamaDecoderLayer`.
        """
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=self.te_rope_emb,
        )


class NLMBaseCausalLM(LlamaPreTrainedModel):
    def __init__(self, args, config: LlamaConfig):
        super().__init__(config)
        self.dummy = nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=True
        )
        self.args = args
        self.layers = nn.ModuleList([])
        for layer_idx in range(config.num_hidden_layers):
            self.layers.append(
                TELlamaDecoderLayer(args, layer_idx=layer_idx),
            )
        self.word_embeddings = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.learnable_cutoff = args.learnable_cutoff
        # self.word_embeddings.weight.register_hook(self.freeze_parital_weight_hook)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.lm_head.weight.register_hook(self.freeze_parital_weight_hook)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.word_embeddings(input_ids).transpose(0, 1)
        if (~attention_mask).any():
            temp_attn_maks = (attention_mask.unsqueeze(1).unsqueeze(2),)
        else:
            temp_attn_maks = None

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=temp_attn_maks)
        hidden_states = self.norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        return (lm_logits.transpose(0, 1),)

    @property
    def emb_weight(self):
        return self.word_embeddings.weight

    @property
    def lm_head_weight(self):
        return self.lm_head.weight

    def freeze_parital_weight_hook(self, grad):
        grad[: self.learnable_cutoff, :] = 0
        return grad
