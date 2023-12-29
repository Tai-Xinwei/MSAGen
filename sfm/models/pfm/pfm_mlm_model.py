# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.models.llama.modeling_llama import LlamaModel

from sfm.data.prot_data.processed_mlm_dataset import Batch
from sfm.models.llama2.llama_modules import (
    LlamaDecoderLayer,
    LlamaEmbeddingsBase,
    LlamaHead,
    LlamaNorm,
)
from sfm.models.pfm.pfm_mlm_config import PfmMlmConfig
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.model import Model
from sfm.utils.optim.optimizer import myAdam
from sfm.utils.optim.set_lr import DECAY_COSINE_RATE, groupWarmupDecayLR


class PfmMlmModel(Model):
    def __init__(self, config: PfmMlmConfig):
        super().__init__()
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)

        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(LlamaDecoderLayer(config))

        self.layers = nn.ModuleList(layers)

        # self.final_norm = LlamaNorm(config)
        self.lm_head = LlamaHead(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            module.weight.data[self.config.pad_token_id].zero_()

    def forward(self, batch: Batch):
        input_ids = batch.x
        bsz, seq_len = input_ids.shape

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)

        inputs_embeds = self.emb(input_ids)

        # Bsz, 1, Q, KV
        attention_mask = torch.zeros(bsz, 1, seq_len, seq_len, device=input_ids.device)
        padding_mask = (
            batch.pad_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, seq_len, -1)
        )

        attention_mask[~padding_mask] = torch.finfo(inputs_embeds.dtype).min

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        # hidden_states = self.final_norm((hidden_states, None, None))
        lm_logits = self.lm_head((hidden_states,))[0]

        return lm_logits

    def compute_loss(self, pred: torch.Tensor, batch: Batch):
        loss_fct = nn.CrossEntropyLoss()

        selected_logits = pred[batch.mask].float()
        selected_labels = batch.y[batch.mask]

        loss = loss_fct(selected_logits, selected_labels)
        ppl = torch.exp(loss.detach()).cpu().item()
        acc = (
            (selected_logits.argmax(dim=-1) == selected_labels)
            .float()
            .mean()
            .detach()
            .cpu()
            .item()
        )
        n_tokens = batch.y.numel() * 1.0
        n_selected_tokens = selected_labels.numel() * 1.0

        return ModelOutput(
            loss,
            num_examples=len(batch.x),
            log_output={
                "acc": acc,
                "ppl": ppl,
                "tokens": n_tokens,
                "selected_tokens": n_selected_tokens,
            },
        )

    def config_optimizer(self, model: Module = None) -> Tuple[Optimizer, LRScheduler]:
        if model is None:
            model = self

        optimizer, _ = myAdam(
            model,
            lr=self.config.max_lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
            eps=1e-8,
        )

        lr_scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=self.config.total_num_steps,
            warmup_max_lr=self.config.max_lr,
            warmup_num_steps=self.config.warmup_num_steps,
            decay_type=DECAY_COSINE_RATE,
        )
        return (optimizer, lr_scheduler)
