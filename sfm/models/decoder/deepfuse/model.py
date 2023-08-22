# -*- coding: utf-8 -*-
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.models.biogpt.modeling_biogpt import BioGptLearnedPositionalEmbedding
from transformers.models.llama.modeling_llama import _expand_mask, _make_causal_mask

from sfm.data.dec_data.dataset import MixedTokenData, TokenType
from sfm.logging import logger
from sfm.models.decoder.deepfuse.config import (
    DecDeepFuseConfig,
    LayerUsage,
    SciDeocerType,
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


class DecDeepFuse(Model):
    def __init__(self, config: DecDeepFuseConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        if config.entity_decoder_model_type == SciDeocerType.BioGPT:
            # Only BioGPT uses the learned positional embedding
            self.embed_positions = BioGptLearnedPositionalEmbedding(
                config.max_position_embeddings, config.entity_hidden_size
            )
        else:
            self.embed_positions = None

        self.decoder_layers = nn.ModuleList([])
        self.layer_usage = LayerUsage.from_str(config.layer_usage)

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

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.load_from_pretrained()

    def load_from_pretrained(self):
        """
        Load pretrained LLaMA and entity decoder.
        Here we need to rename the parameters to match the current model.
        """
        logger.info("Loading pretrained models")
        logger.info(f"Loading LLaMA from {self.config.llama_model}")
        logger.info(f"Loading entity decoder from {self.config.entity_decoder_model}")

        entity_decoder_state_dict = torch.load(
            f"{self.config.entity_decoder_model}/pytorch_model.bin"
        )

        mapped_state_dict = {}

        # Embedding
        lambda_state_dict = torch.load(f"{self.config.llama_model}/model.hybrid_emb.pt")
        mapped_state_dict["embed_tokens.weight"] = torch.cat(
            [
                lambda_state_dict["embed_tokens.weight"],
                entity_decoder_state_dict["biogpt.embed_tokens.weight"],
            ],
            dim=0,
        )

        mapped_state_dict["embed_positions.weight"] = entity_decoder_state_dict[
            "biogpt.embed_positions.weight"
        ]

        # Each layer
        entity_layer_id = 0
        for i, layer in enumerate(self.decoder_layers):
            llama_state_dict = torch.load(
                f"{self.config.llama_model}/model_layers.{i}.pth"
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
            "biogpt.final_norm.weight"
        ]
        mapped_state_dict["final_norm.Entity.bias"] = entity_decoder_state_dict[
            "biogpt.final_norm.bias"
        ]

        # output layer
        lambda_state_dict = torch.load(f"{self.config.llama_model}/model.lm_head.pt")
        mapped_state_dict["lm_head.weight"] = torch.cat(
            [
                lambda_state_dict["lm_head.weight"],
                entity_decoder_state_dict["output_projection.weight"],
            ],
        )

        # Init the rest of the parameters, e.g., adapters
        total_random_init_params = 0
        for k, v in self.state_dict().items():
            if k not in mapped_state_dict:
                logger.info(
                    f"Random init {k}, shape {v.shape}, dtype {v.dtype}, size {v.nelement()}"
                )

                mapped_state_dict[k] = v
                total_random_init_params += v.nelement()

        logger.info(f"Total random init params: {total_random_init_params}")

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

    def _make_key_padding_mask(
        self, seq_len: torch.Tensor, max_len: int, dtype: torch.dtype
    ) -> torch.Tensor:
        bsz = seq_len.shape[0]
        mask = torch.full((bsz, max_len), 0, dtype=dtype, device=seq_len.device)

        # TODO: convert to index computation
        for i in range(bsz):
            mask[i, : seq_len[i]] = 1

        return mask

    def forward(
        self,
        batch: MixedTokenData,
    ) -> torch.Tensor:
        x = self.embed_tokens(batch.token_seq)
        bsz, seq_len, _ = x.shape
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0)

        attention_mask = self._make_key_padding_mask(
            batch.token_seq_len, seq_len, x.dtype
        )

        if self.embed_positions is not None:
            positions = self.embed_positions(attention_mask, past_key_values_length=0)
            x = x + positions

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (bsz, seq_len), x, 0
        )

        h = HiddenState.from_dense(x, batch.token_type_mask)
        tensors = h.to_tuple() + (attention_mask, position_ids)

        for layer in self.decoder_layers:
            tensors = layer(tensors)

        h = HiddenState.from_tuple(tensors[:-2])
        x = h.to_dense()
        x = self.lm_head(x)

        return x

    def compute_loss(self, pred, batch: MixedTokenData):
        logits = pred.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch.token_seq[..., 1:].contiguous()

        loss_weight = torch.ones(self.config.vocab_size, device=shift_logits.device)

        text_token_range = batch.entity_id_rage[TokenType.Text]
        loss_weight[
            text_token_range.start : text_token_range.end
        ] = self.config.text_loss_weight

        entity_token_range = batch.entity_id_rage[TokenType.Entity]
        loss_weight[
            entity_token_range.start : entity_token_range.end
        ] = self.config.entity_loss_weight

        loss_fct = nn.CrossEntropyLoss(weight=loss_weight)
        loss = loss_fct(
            shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
        )

        return ModelOutput(
            loss=loss,
            num_examples=batch.batch_size,
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
                "weight_decay": float(self.args.weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]

        optimizer = AdamW(
            optim_params,
            lr=float(self.args.init_lr),
            weight_decay=float(self.args.weight_decay),
            betas=(self.args.beta1, self.args.beta2),
        )
        max_epoch = self.args.total_num_epochs
        warmup_start_lr = self.args.warmup_lr
        warmup_epochs = self.args.warmup_num_epochs
        iters_per_epoch = self.args.iters_per_epoch
        min_lr = self.args.min_lr
        scheduler = LinearWarmupCosineLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            iters_per_epoch=iters_per_epoch,
            min_lr=min_lr,
            init_lr=self.args.init_lr,
            warmup_steps=warmup_epochs * iters_per_epoch,
            warmup_start_lr=warmup_start_lr,
        )
        return optimizer, scheduler
