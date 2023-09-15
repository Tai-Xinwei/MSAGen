# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import GenerationMixin

from sfm.data.dec_data.datasets import ENTITY_MARKERS, MixedTokenData, TokenType
from sfm.logging import logger
from sfm.models.decoder.deepfuse.config import DecDeepFuseConfig, LayerUsage
from sfm.models.decoder.deepfuse.hidden_state import HiddenState
from sfm.models.decoder.deepfuse.modules import (
    Embed,
    Head,
    MixLayer,
    SeperateLayer,
    TextOnly,
)
from sfm.models.tamgent.scheduler import LinearWarmupCosineLRScheduler
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.pipeline_module import SFMPipelineModelMixin
from sfm.utils.optim.adam import AdamW

"""
TODO:

1. We don't use any dropout now as LLaMA seems not using it.
2. Only support one entity type now.
"""


class GenerationOutput(object):
    def __init__(
        self, hidden: HiddenState, past_key_values: List[torch.Tensor]
    ) -> None:
        self.hidden = hidden
        self.past_key_values = past_key_values

    @property
    def logits(self):
        # As the vocab size are different, we use the max vocab size
        max_vocab_size = max([v.shape[-1] for v in self.hidden.x_dict.values()])
        x_dict_new = {}

        for token_type, x in self.hidden.x_dict.items():
            x_dict_new[token_type] = torch.zeros(
                (x.shape[0], max_vocab_size), device=x.device
            )
            x_dict_new[token_type][:, : x.shape[-1]] = x

        h_new = HiddenState(x_dict_new, self.hidden.token_type_mask)
        return h_new.to_dense()

    def __contains__(self, item):
        return item in self.__dict__

    def __index__(self, item):
        return self.__dict__[item]


class DecDeepFuseModel(SFMPipelineModelMixin, GenerationMixin):
    def __init__(self, config: DecDeepFuseConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = Embed(config=config)

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

        self.head = Head(config)

        if config.load_from_pretrained:
            self.load_from_pretrained()
            self.freeze_params()

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

        mapped_state_dict["embed.embed_tokens.Text.weight"] = self.extend_emb(
            lambda_state_dict["embed_tokens.weight"]
        )

        mapped_state_dict[
            "embed.embed_tokens.Entity.weight"
        ] = entity_decoder_state_dict["biogpt.embed_tokens.weight"]

        mapped_state_dict["embed.embed_positions.weight"] = entity_decoder_state_dict[
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
        mapped_state_dict["head.final_norm.Text.weight"] = lambda_state_dict[
            "norm.weight"
        ]
        mapped_state_dict["head.final_norm.Entity.weight"] = entity_decoder_state_dict[
            "biogpt.layer_norm.weight"
        ]
        mapped_state_dict["head.final_norm.Entity.bias"] = entity_decoder_state_dict[
            "biogpt.layer_norm.bias"
        ]

        # output layer
        lambda_state_dict = torch.load(f"{self.config.llama_model}/model.lm_head.pt")
        mapped_state_dict["head.lm_head.Text.weight"] = self.extend_emb(
            lambda_state_dict["lm_head.weight"]
        )
        mapped_state_dict["head.lm_head.Entity.weight"] = entity_decoder_state_dict[
            "output_projection.weight"
        ]

        # Init the rest of the parameters, e.g., adapters
        total_random_init_params = 0
        has_inv_freq = False
        for k, v in self.state_dict().items():
            if k not in mapped_state_dict:
                kind = "adapter" if "adapter" in k else "layer"

                logger.info(
                    f"Random init {kind} {k}, shape {v.shape}, dtype {v.dtype}, size {v.nelement()}"
                )

                mapped_state_dict[k] = v
                total_random_init_params += v.nelement()

            if "inv_freq" in k:
                has_inv_freq = True

        logger.info(f"Total random init params count: {total_random_init_params:,}")

        if not has_inv_freq:
            # remove all "inv_freq" as they are buffers, which cannot be loaded
            # seems a bug in huggingface
            for k in list(mapped_state_dict.keys()):
                if "inv_freq" in k:
                    del mapped_state_dict[k]

        self.load_state_dict(mapped_state_dict)

    def freeze_params(self):
        if self.config.freeze_text_model:
            logger.info("Freezing text encoder")
            for param in self.embed.embed_tokens[TokenType.Text.name].parameters():
                param.requires_grad = False

            for param in self.head.final_norm[TokenType.Text.name].parameters():
                param.requires_grad = False

            for param in self.head.lm_head[TokenType.Text.name].parameters():
                param.requires_grad = False

            for layer in self.decoder_layers:
                layer.freeze_text_model()

            if self.config.finetune_text_extra_emb:
                logger.info("Finetuning text extra embeddings")
                finetuned_emb_count = len(ENTITY_MARKERS) + 1  # +1 for PAD

                def grad_filter_hook(grad):
                    # Only finetune of the last finetuned_emb_count embeddings
                    # But keep the gradients of the rest of the embeddings
                    grad[:-finetuned_emb_count] = 0
                    return grad

                for param in self.embed.embed_tokens[TokenType.Text.name].parameters():
                    param.requires_grad = True
                    param.register_hook(lambda grad: grad_filter_hook(grad))

                for param in self.head.lm_head[TokenType.Text.name].parameters():
                    param.requires_grad = True
                    param.register_hook(lambda grad: grad_filter_hook(grad))

        if self.config.freeze_entity_model:
            logger.info("Freezing entity encoder")

            for param in self.embed.embed_tokens[TokenType.Entity.name].parameters():
                param.requires_grad = False

            for param in self.head.final_norm[TokenType.Entity.name].parameters():
                param.requires_grad = False

            for param in self.head.lm_head[TokenType.Entity.name].parameters():
                param.requires_grad = False

            for layer in self.decoder_layers:
                layer.freeze_entity_model()

    def forward(
        self,
        batch: MixedTokenData,
        return_dict: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> HiddenState:
        tensors = batch.to_tuple()[0] + (
            attention_mask,
            position_ids,
            None if past_key_values is None else past_key_values[-1],
        )

        tensors = self.embed(tensors)

        h = HiddenState.from_tuple(tensors[:-2])
        attention_mask, position_ids = tensors[-2:]

        all_past_key_values = []
        for idx, layer in enumerate(self.decoder_layers):
            past_key_value = past_key_values[idx] if past_key_values else None

            # Here we intentionally use the forwad_impl rather than forward
            # as we want to use kV caching
            h, past_key_value = layer.forward_impl(
                h=h,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            all_past_key_values.append(past_key_value)

        tensors = h.to_tuple()

        h_tuple = self.head(tensors)

        h = HiddenState.from_tuple(h_tuple)
        if use_cache:
            # this means we are generating
            return GenerationOutput(
                hidden=h,
                past_key_values=all_past_key_values,
            )

        else:
            return h

    def compute_loss(self, pred: HiddenState, batch: MixedTokenData):
        if type(pred) is tuple:
            pred = HiddenState.from_tuple(pred)
            batch = MixedTokenData.from_tuple(batch)

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

            loss_by_type[f"{token_type.name}_loss"] = loss.item()
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

    def to_layers(self) -> List[Module]:
        return [self.embed] + list(self.decoder_layers) + [self.head]

    # for generation mixin
    def can_generate(self):
        return True

    @property
    def main_input_name(self):
        return "input_ids"

    @property
    def device(self):
        return next(self.parameters()).device

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        batch: MixedTokenData,
        use_cache: bool,
        past_key_values: Optional[List[torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_token_mapping: Optional[List[Tuple[int, int]]] = None,
    ):
        if past_key_values:
            # When past_key_values are provided, we need to update the batch
            # and recompute the token_type_mask.
            # We also need to update the token_ids from different vocab

            assert (
                special_token_mapping is not None
            ), "special_token_mapping must be provided when past_key_values is provided"

            text_id_to_entity = {idx[0]: idx[1] for idx in special_token_mapping}
            entity_id_to_text = {idx[1]: idx[0] for idx in special_token_mapping}

            bsz = batch.batch_size

            new_input_ids = torch.zeros(
                (bsz, 1), dtype=batch.token_seq.dtype, device=batch.token_seq.device
            )
            new_token_type_mask = torch.zeros(
                (bsz, 1),
                dtype=batch.token_type_mask.dtype,
                device=batch.token_type_mask.device,
            )

            for i in range(bsz):
                last_token_type = batch.token_type_mask[i][-1].item()
                last_gen_token = input_ids[i][-1].item()
                if last_token_type == TokenType.Text.value:
                    if (
                        last_gen_token in text_id_to_entity
                    ):  # start of entity generation
                        new_input_ids[i, 0] = text_id_to_entity[last_gen_token]
                        new_token_type_mask[i, 0] = TokenType.Entity.value
                    else:
                        new_input_ids[i, 0] = last_gen_token
                        new_token_type_mask[i, 0] = TokenType.Text.value
                elif last_token_type == TokenType.Entity.value:
                    if last_gen_token in entity_id_to_text:  # end of entity generation
                        new_input_ids[i, 0] = entity_id_to_text[last_gen_token]
                        new_token_type_mask[i, 0] = TokenType.Text.value
                    else:
                        new_input_ids[i, 0] = last_gen_token
                        new_token_type_mask[i, 0] = TokenType.Entity.value

            batch.token_seq = new_input_ids
            batch.token_type_mask = new_token_type_mask

            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids[:, -1].unsqueeze(-1)

            return {
                "batch": batch,
                "use_cache": use_cache,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }

        else:
            # First span, no past_key_values
            # No need to update the batch
            return {
                "batch": batch,
                "use_cache": use_cache,
                "past_key_values": None,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }

    def _extract_past_from_model_output(
        self, outputs: GenerationOutput, standardize_cache_format: bool = False
    ):
        return outputs.past_key_values

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
