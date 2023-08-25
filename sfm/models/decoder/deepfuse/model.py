# -*- coding: utf-8 -*-
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.models.biogpt.modeling_biogpt import BioGptLearnedPositionalEmbedding
from transformers.models.llama.modeling_llama import _expand_mask, _make_causal_mask

from sfm.data.dec_data.datasets import MixedTokenData, TokenType
from sfm.logging import logger
from sfm.models.decoder.deepfuse.config import (
    DecDeepFuseConfig,
    EntityDecoderType,
    LayerUsage,
)
from sfm.models.decoder.deepfuse.hidden_state import HiddenState
from sfm.models.decoder.deepfuse.modules import (
    MixLayer,
    SeperateLayer,
    TextOnly,
    make_norm_dict,
)
from sfm.models.tamgent.scheduler import LinearWarmupCosineLRScheduler
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model
from sfm.utils.optim.adam import AdamW

"""
TODO:

1. We don't use any dropout now as LLaMA seems not using it.
2. Only support one entity type now.
3. KV caching for faster decoding
"""


class DecDeepFuseModel(Model):
    def __init__(self, config: DecDeepFuseConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.ModuleDict(
            {
                TokenType.Text.name: nn.Embedding(
                    config.vocab_size + config.new_token_count, config.hidden_size
                ),
                TokenType.Entity.name: nn.Embedding(
                    config.entity_vocab_size, config.entity_hidden_size
                ),
            }
        )

        if config.entity_decoder_model_type == EntityDecoderType.BioGPT:
            # Only BioGPT uses the learned positional embedding
            self.embed_positions = BioGptLearnedPositionalEmbedding(
                config.max_position_embeddings, config.entity_hidden_size
            )
        else:
            self.embed_positions = None

        self.decoder_layers = nn.ModuleList([])
        self.layer_usage = LayerUsage.from_str(config.layer_usage)

        assert (
            len(self.layer_usage) == config.num_hidden_layers
        ), f"Number of layers {config.num_hidden_layers} does not match layer usage {config.layer_usage}"

        assert (
            len([x for x in self.layer_usage if x != LayerUsage.NotUsed])
            == config.entity_num_hidden_layers
        ), f"Number of entity layers {config.entity_num_hidden_layers} does not match layer usage {config.layer_usage}"

        for usage in self.layer_usage:
            if usage == LayerUsage.Mixing:
                self.decoder_layers.append(MixLayer(config))
            elif usage == LayerUsage.Seperate:
                self.decoder_layers.append(SeperateLayer(config))
            elif usage == LayerUsage.NotUsed:
                self.decoder_layers.append(TextOnly(config))
            else:
                raise ValueError(f"Unknown layer usage {usage}")

        self.final_norm = make_norm_dict(config)

        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head = nn.ModuleDict(
            {
                TokenType.Text.name: nn.Linear(
                    config.hidden_size,
                    config.vocab_size + config.new_token_count,
                    bias=False,
                ),
                TokenType.Entity.name: nn.Linear(
                    config.entity_hidden_size, config.entity_vocab_size, bias=False
                ),
            }
        )

        self.load_from_pretrained()

    def extend_emb(self, emb: torch.Tensor) -> torch.Tensor:
        new_emb = torch.zeros(
            (emb.shape[0] + self.config.new_token_count, emb.shape[1]),
            device=emb.device,
        )
        new_emb[: emb.shape[0], :] = emb

        mean, std = emb.mean(dim=0), emb.std(dim=0)
        new_emb[-self.config.new_token_count :] = (
            torch.randn((self.config.new_token_count, emb.shape[1]), device=emb.device)
            * std
            + mean
        )

        return new_emb

    def load_from_pretrained(self):
        """
        Load pretrained LLaMA and entity decoder.
        Here we need to rename the parameters to match the current model.
        """
        logger.info("Loading pretrained models")
        logger.info(
            f"Loading LLaMA from {self.config.llama_model} and entity decoder from {self.config.entity_decoder_model}"
        )

        entity_decoder_state_dict = torch.load(
            f"{self.config.entity_decoder_model}/pytorch_model.bin"
        )

        mapped_state_dict = {}

        # Embedding
        lambda_state_dict = torch.load(f"{self.config.llama_model}/model.hybrid_emb.pt")

        mapped_state_dict["embed_tokens.Text.weight"] = self.extend_emb(
            lambda_state_dict["embed_tokens.weight"]
        )

        mapped_state_dict["embed_tokens.Entity.weight"] = entity_decoder_state_dict[
            "biogpt.embed_tokens.weight"
        ]

        mapped_state_dict["embed_positions.weight"] = entity_decoder_state_dict[
            "biogpt.embed_positions.weight"
        ]

        # Each layer
        entity_layer_id = 0
        for i, layer in enumerate(self.decoder_layers):
            logger.info(f"Loading layer {i}, type {type(layer).__name__}")
            llama_state_dict = torch.load(
                f"{self.config.llama_model}/model.layers.{i}.pt"
            )

            for k, v in layer.map_state_dict(
                prefix=f"decoder_layers.{i}",
                llama_state_dict=llama_state_dict,
                entity_layer_id=entity_layer_id,
                entity_decoder_state_dict=entity_decoder_state_dict,
            ).items():
                mapped_state_dict[k] = v

            if self.layer_usage[i] != LayerUsage.NotUsed:
                entity_layer_id += 1
        # Final norm
        lambda_state_dict = torch.load(f"{self.config.llama_model}/model.norm.pt")
        mapped_state_dict["final_norm.Text.weight"] = lambda_state_dict["norm.weight"]
        mapped_state_dict["final_norm.Entity.weight"] = entity_decoder_state_dict[
            "biogpt.layer_norm.weight"
        ]
        mapped_state_dict["final_norm.Entity.bias"] = entity_decoder_state_dict[
            "biogpt.layer_norm.bias"
        ]

        # output layer
        lambda_state_dict = torch.load(f"{self.config.llama_model}/model.lm_head.pt")
        mapped_state_dict["lm_head.Text.weight"] = self.extend_emb(
            lambda_state_dict["lm_head.weight"]
        )
        mapped_state_dict["lm_head.Entity.weight"] = entity_decoder_state_dict[
            "output_projection.weight"
        ]

        # Init the rest of the parameters, e.g., adapters
        total_random_init_params = 0
        for k, v in self.state_dict().items():
            if k not in mapped_state_dict:
                kind = "adapter" if "adapter" in k else "layer"

                logger.info(
                    f"Random init {kind} {k}, shape {v.shape}, dtype {v.dtype}, size {v.nelement()}"
                )

                mapped_state_dict[k] = v
                total_random_init_params += v.nelement()

        logger.info(f"Total random init params count: {total_random_init_params:,}")

        self.load_state_dict(mapped_state_dict)

    # See transformers.models.bart.modeling_bart.BartDecoder
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        batch: MixedTokenData,
    ) -> HiddenState:
        h = HiddenState.from_dense(batch.token_seq, batch.token_type_mask)

        h = h.apply_all_types_mapping(self.embed_tokens)

        (
            bsz,
            seq_len,
        ) = batch.token_type_mask.shape

        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=h.device)
        position_ids = position_ids.unsqueeze(0)

        attention_mask = batch.non_padding_mask.to(h.dtype)

        # Only BioGPT uses the learned positional embedding
        if self.embed_positions is not None:
            positions = self.embed_positions(attention_mask, past_key_values_length=0)
            x = (
                h.x_dict[TokenType.Entity]
                + positions[batch.token_type_mask == TokenType.Entity.value]
            )
            h = h.update_x_dict(TokenType.Entity, x)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (bsz, seq_len), x, 0
        )

        tensors = h.to_tuple() + (attention_mask, position_ids)

        for layer in self.decoder_layers:
            tensors = layer(tensors)

        h = HiddenState.from_tuple(tensors[:-2])
        h = h.apply_all_types_mapping(self.lm_head)

        return h

    def compute_loss(self, pred: HiddenState, batch: MixedTokenData):
        loss_by_type = {}
        loss_fct = nn.CrossEntropyLoss()
        total_loss = 0

        padding_state = HiddenState.from_dense(
            batch.non_padding_mask, batch.token_type_mask
        )

        label_state = HiddenState.from_dense(batch.label_seq, batch.token_type_mask)

        for token_type in TokenType:
            # The logits and lables have been shifted by one in the data loader
            logits = pred.x_dict[token_type].float()
            labels = label_state.x_dict[token_type].long()
            non_pad_mask = padding_state.x_dict[token_type].bool()

            loss = loss_fct(logits[non_pad_mask], labels[non_pad_mask])

            loss_by_type[f"{token_type}_loss"] = loss.item()
            total_loss += loss * self.config.loss_weight[token_type.name]

        return ModelOutput(
            loss=total_loss,
            num_examples=batch.batch_size,
            log_output=loss_by_type,
        )

    def config_optimizer(self) -> Tuple[Optimizer, LRScheduler]:
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()

        optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(self.config.weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]

        optimizer = AdamW(
            optim_params,
            lr=float(self.config.init_lr),
            weight_decay=float(self.config.weight_decay),
            betas=(self.config.beta1, self.config.beta2),
        )
        max_epoch = self.config.total_num_epochs
        warmup_start_lr = self.config.warmup_lr
        warmup_epochs = self.config.warmup_num_epochs
        iters_per_epoch = self.config.iters_per_epoch
        min_lr = self.config.min_lr
        scheduler = LinearWarmupCosineLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            iters_per_epoch=iters_per_epoch,
            min_lr=min_lr,
            init_lr=self.config.init_lr,
            warmup_steps=warmup_epochs * iters_per_epoch,
            warmup_start_lr=warmup_start_lr,
        )
        return optimizer, scheduler
