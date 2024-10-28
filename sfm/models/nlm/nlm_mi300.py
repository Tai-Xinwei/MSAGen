# -*- coding: utf-8 -*-
import os
from typing import Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)

from sfm.logging import logger
from sfm.models.llama2.llama_modules import LlamaDecoderLayerPP
from sfm.models.nlm.moduels.autoregressive import AutoregressiveCriterion
from sfm.pipeline.accelerator.dataclasses import ModelOutput, TrainStrategy
from sfm.pipeline.accelerator.model import Model
from sfm.utils.optim.optimizer import myAdam
from sfm.utils.optim.set_lr import DECAY_COSINE_RATE, groupWarmupDecayLR


class NLMBaseAMDModel(Model):
    def __init__(self, args, vocab_size: int):
        super().__init__()

        llama_config = LlamaConfig.from_pretrained(args.dict_path)
        self.args = args

        llama_config = self.init_config(args, llama_config, vocab_size)

        self.net = NLMBaseCausalLM(args, llama_config)
        self.loss_fn = AutoregressiveCriterion(args)

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

    def init_config(self, args, llama_config, vocab_size):
        llama_config.hidden_size = args.hidden_size
        llama_config.intermediate_size = args.intermediate_size
        llama_config.num_hidden_layers = args.num_hidden_layers
        llama_config._attn_implementation = "sdpa"
        llama_config.vocab_size = vocab_size
        llama_config.torch_dtype = "bfloat16" if args.bf16 else "float32"

        return llama_config

    def forward(self, input_tuple: Tuple[torch.Tensor, torch.Tensor]):
        input_ids, attention_mask = input_tuple[0]
        if self.args.strategy not in [TrainStrategy.ThreeD, TrainStrategy.Pipeline]:
            return self.net(input_ids, attention_mask)
        else:
            raise NotImplementedError


class NLMBaseCausalLM(LlamaPreTrainedModel):
    def __init__(self, args, config: LlamaConfig):
        super().__init__(config)
        self.dummy = nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=True
        )
        self.args = args
        self.layers = nn.ModuleList([])
        for layer_index in range(config.num_hidden_layers):
            self.layers.append(
                LlamaDecoderLayerPP(args, layer_index=layer_index),
            )
        self.word_embeddings = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.learnable_cutoff = args.learnable_cutoff
        self.word_embeddings.weight.register_hook(self.freeze_parital_weight_hook)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight.register_hook(self.freeze_parital_weight_hook)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        llm_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape

        seq_length_with_past = seq_length
        past_key_values_length = 0

        device = input_ids.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # B, T, hidden_size
        hidden_states = self.word_embeddings(input_ids)

        # attention mask
        if llm_mask is None:
            llm_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )

        llm_mask = self._prepare_decoder_attention_mask(
            llm_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
            input_ids,
        )

        for layer in self.layers:
            input_tuple = (hidden_states, llm_mask, position_ids)
            (hidden_states, llm_mask, position_ids) = layer(input_tuple)

        hidden_states = self.norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        return (lm_logits,)

    @property
    def emb_weight(self):
        return self.word_embeddings.weight

    @property
    def lm_head_weight(self):
        return self.lm_head.weight

    def freeze_parital_weight_hook(self, grad):
        grad[: self.learnable_cutoff, :] = 0
        return grad

    # Copied from transformers.models.bart.modeling_bart._make_causal_mask
    def _make_causal_mask(
        self,
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), False, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), True)
        # mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [
                    torch.ones(
                        tgt_len, past_key_values_length, dtype=mask.dtype, device=device
                    ),
                    mask,
                ],
                dim=-1,
            )
        assert (
            mask.dtype == torch.bool
        ), f"expected mask to have dtype torch.bool, but got {mask.dtype}"

        return mask[None, None, :, :].expand(
            bsz, 1, tgt_len, tgt_len + past_key_values_length
        )

    # Copied from transformers.models.bart.modeling_bart._expand_mask
    def _expand_mask(
        self, mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
    ):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(
            bsz, 1, tgt_len, src_len
        )  # .to(dtype)

        return expanded_mask.to(torch.bool)

    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
        input_ids,
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask_bool = None
        if input_shape[-1] > 1:
            combined_attention_mask_bool = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)

            combined_attention_mask_bool = (
                expanded_attn_mask
                if combined_attention_mask_bool is None
                else expanded_attn_mask & combined_attention_mask_bool
            )

        return combined_attention_mask_bool
