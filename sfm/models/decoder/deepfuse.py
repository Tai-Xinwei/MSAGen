# -*- coding: utf-8 -*-
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.models.llama.modeling_llama import (
    ACT2FN,
    LlamaDecoderLayer,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    _expand_mask,
    _make_causal_mask,
    apply_rotary_pos_emb,
    repeat_kv,
)

from sfm.data.dec_data.dataset import MixedTokenData, TokenType
from sfm.logging import logger
from sfm.models.tamgent.scheduler import LinearWarmupCosineLRScheduler
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig, ModelOutput
from sfm.pipeline.accelerator.trainer import Model
from sfm.utils.optim.adam import AdamW


@dataclass
class DecDeepFuseConfig(DistributedTrainConfig):
    freeze_text_encoder: bool = True

    llama_model: str = "/blob/shufxi/llama/7B"
    molxpt_model: str = "/blob/shufxi/molxpt"
    max_txt_len_llama: int = 500
    max_txt_len_smiles: int = 2048
    end_sym: str = "\n"

    train_data_path: str = ""
    val_data_path: str = ""

    hidden_size: int = 1024
    intermediate_size: int = 4096
    hidden_act: str = "gelu"

    entity_hidden_size: int = 1024

    num_attention_heads: int = 16
    num_key_value_heads: int = 1  # KV grouping
    entity_num_attention_heads: int = (
        16  # For now, assume to be less than or equal to num_attention_heads
    )
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[Dict[str, float]] = None

    num_adapter_layers: int = 2
    adapter_hidden_size: int = 1024
    adapter_activation: str = "gelu"

    vocab_size: int = 0  # total vocab size

    # This is a string representing how the science decoder layers are used.
    # For example, "MSNMSN" means that:
    # the first and fourth layers are used for Mixing,
    # the second and fifth layers are used Separately, i.e., no attention between them,
    # and the third and sixth layers are Not used, i.e., only text layers are used.
    # The string lenth should be equal to the number of text layers.
    layer_usage: str = ""

    text_loss_weight: float = 0.0
    entity_loss_weight: float = 1.0

    pretraining_tp: int = 0

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def entity_head_dim(self):
        return self.entity_hidden_size // self.entity_num_attention_heads

    @property
    def max_position_embeddings(self):
        return self.max_txt_len_llama + self.max_txt_len_smiles


def make_norm_dict(config: DecDeepFuseConfig) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            TokenType.Text.name: LlamaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            ),
            TokenType.Entity.name: LlamaRMSNorm(
                config.entity_hidden_size, eps=config.rms_norm_eps
            ),
        }
    )


@dataclass
class HiddenState:
    """
    The x_dict reprensent a jadged tensor because the each token can have differernt dim
    """

    # Each type holds a tensor of shape [N, *] where N is the number of this type of tokens
    x_dict: Dict[TokenType, torch.Tensor]

    token_type_mask: torch.Tensor

    @property
    def batch_size(self):
        return self.token_type_mask.shape[0]

    @property
    def seq_len(self):
        return self.token_type_mask.shape[1]

    def __add__(self, other):
        return HiddenState(
            x_dict={k: v + other.x_dict[k] for k, v in self.x_dict.items()},
            token_type_mask=self.token_type_mask,
        )

    # Create a copy of this object and update one item of x_dict
    def update_x_dict(self, token_type: TokenType, x: torch.Tensor):
        x_dict = {k: v for k, v in self.x_dict.items()}
        x_dict[token_type] = x
        return HiddenState(x_dict, self.token_type_mask)

    def apply_all_type_mapping(self, fn: nn.ModuleDict, **kwargs):
        x_dict_new = {k: fn[k.name](x, **kwargs) for k, x in self.x_dict.items()}
        return HiddenState(x_dict_new, self.token_type_mask)

    def to_dense(self) -> torch.Tensor:
        # All tensor must have the dim [N, *, ...]
        # If we want to concat them, we need to make sure that the last dims is the same

        extra_dims = self.x_dict[TokenType.Text].shape[1:]

        # Check all dim are the same
        for k, v in self.x_dict.items():
            assert v.shape[1:] == extra_dims

        ret = torch.zeros(
            self.batch_size,
            self.seq_len,
            *extra_dims,
            device=self.x_dict[TokenType.Text].device,
            dtype=self.x_dict[TokenType.Text].dtype,
        )
        for k, v in self.x_dict.items():
            ret[self.token_type_mask == k.value] = v

        return ret

    @classmethod
    def from_dense(cls, x: torch.Tensor, token_type_mask: torch.Tensor):
        x_dict = {}
        for k in TokenType:
            x_dict[k] = x[token_type_mask == k.value]
        return cls(x_dict, token_type_mask)

    def to_single_type_batch(self, token_type: TokenType, attention_mask: torch.Tensor):
        """
        Convert this ragged batch to a batch (BxTxC) with only one token type.
        Other kinds of tokens are masked out.
        This is useful when we want to apply a layer to only one token type.
        """
        embed_dim = self.x_dict[token_type].shape[-1]

        batch = torch.zeros(
            (self.batch_size, self.seq_len, embed_dim),
            device=self.x_dict[token_type].device,
            dtype=self.x_dict[token_type].dtype,
        )

        batch[self.token_type_mask == token_type.value] = self.x_dict[token_type]

        attention_mask_new = (
            attention_mask  # [B, 1, SRC_LEN, TGT_LEN], the 2nd dim head
        )

        not_to_attend_mask = (self.token_type_mask != token_type.value)[
            :, None, None, :
        ].expand_as(attention_mask_new)

        val = torch.finfo(attention_mask.dtype).min
        attention_mask_new[not_to_attend_mask] = val

        return batch, attention_mask_new

    def update_single_type_batch(self, token_type: TokenType, x: torch.Tensor):
        """
        Update the x_dict with the new batch, after applying a layer to only one type.
        """
        x = x[self.token_type_mask == token_type.value]
        return self.update_x_dict(token_type, x)


class Adapter(nn.Module):
    def __init__(self, config: DecDeepFuseConfig, in_dim, out_dim):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([])
        hidden_dims = (
            [in_dim]
            + [config.adapter_hidden_size] * (config.num_adapter_layers - 1)
            + [out_dim]
        )

        for i in range(config.num_adapter_layers):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i != config.num_adapter_layers - 1:
                self.layers.append(ACT2FN[config.adapter_activation])

    def forward(self, x: torch.TensorType) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


class MLP(nn.Module):
    def __init__(self, config: DecDeepFuseConfig, token_type: TokenType) -> None:
        super().__init__()
        self.token_type = token_type
        if token_type == TokenType.Text:
            self.mlp = LlamaMLP(config)
        else:
            config_copy = deepcopy(config)
            config_copy.hidden_size = config.entity_hidden_size

            # TODO: entity model may trained from fairseq, thus use different activation
            self.mlp = LlamaMLP(config_copy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AttnQKVProj(nn.Module):
    def __init__(self, config: DecDeepFuseConfig, token_type: TokenType, kind: str):
        super().__init__()
        self.config = config
        self.token_type = token_type
        self.kind = kind

        if token_type == TokenType.Text:
            if kind in ["k", "v"]:
                self.layer = nn.Linear(
                    config.hidden_size, config.num_key_value_heads * config.head_dim
                )
            else:
                self.layer = nn.Linear(config.hidden_size, config.hidden_size)
            self.out_proj = nn.Identity()
        else:
            self.layer = nn.Linear(config.entity_hidden_size, config.entity_hidden_size)
            self.out_proj = Adapter(
                config,
                self.config.entity_head_dim,
                self.config.head_dim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The seq_len here is the number of tokens in the batch
        seq_len, _ = x.shape

        x = self.layer(x)

        if self.token_type != TokenType.Text:
            # First, project on head dim

            x = x.reshape(
                seq_len,
                self.config.entity_num_attention_heads,
                self.config.entity_head_dim,
            )

            x = self.out_proj(x)

            # Then, duplicate the heads
            replica = (
                self.config.num_attention_heads
                // self.config.entity_num_attention_heads
            )
            x = (
                x[:, None, :, :]
                .expand(-1, replica, -1, -1)
                .reshape(seq_len, self.config.num_attention_heads, self.config.head_dim)
            )

        else:
            if self.kind in ["k", "v"]:
                # repeat k/v heads if n_kv_heads < n_heads
                replica = (
                    self.config.num_attention_heads // self.config.num_key_value_heads
                )
                x = (
                    x.reshape(
                        seq_len, self.config.num_key_value_heads, self.config.head_dim
                    )[:, None, :, :]
                    .expand(-1, replica, -1, -1)
                    .reshape(
                        seq_len, self.config.num_attention_heads, self.config.head_dim
                    )
                )
            else:
                x = x.reshape(
                    seq_len, self.config.num_attention_heads, self.config.head_dim
                )

        return x


class AttnOutputProj(nn.Module):
    def __init__(self, config: DecDeepFuseConfig, token_type: TokenType) -> None:
        super().__init__()
        self.config = config
        self.token_type = token_type

        if token_type == TokenType.Text:
            self.proj = nn.Identity()
            self.layer = nn.Linear(config.hidden_size, config.hidden_size)

        else:
            self.proj = nn.Linear(
                self.config.head_dim,
                self.config.entity_head_dim,
            )
            self.layer = nn.Linear(config.entity_hidden_size, config.entity_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, head, _ = x.shape

        # merge the heads
        if self.token_type != TokenType.Text:
            replica = (
                self.config.num_attention_heads
                // self.config.entity_num_attention_heads
            )
            assert head == self.config.entity_num_attention_heads * replica
            x = x.reshape(
                bsz,
                replica,
                self.config.entity_num_attention_heads,
                self.config.entity_head_dim,
            )
            x = x.mean(dim=1)
            x = self.proj(x)

        x = x.reshape(bsz, -1)

        x = self.layer(x)
        return x


class FusedAttn(nn.Module):
    """
    We first project all Q, K, V to text space, conduct attntion.
    Then project them back to entity space
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        self.q_proj_dict = nn.ModuleDict()
        self.k_proj_dict = nn.ModuleDict()
        self.v_proj_dict = nn.ModuleDict()
        self.o_proj_dict = nn.ModuleDict()

        for token_type in TokenType:
            self.q_proj_dict[token_type.name] = AttnQKVProj(
                config, token_type, kind="q"
            )
            self.k_proj_dict[token_type.name] = AttnQKVProj(
                config, token_type, kind="k"
            )
            self.v_proj_dict[token_type.name] = AttnQKVProj(
                config, token_type, kind="v"
            )
            self.o_proj_dict[token_type.name] = AttnOutputProj(config, token_type)

        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        h: HiddenState,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> HiddenState:
        query_hidden = h.apply_all_type_mapping(self.q_proj_dict)
        key_hidden = h.apply_all_type_mapping(self.k_proj_dict)
        value_hidden = h.apply_all_type_mapping(self.v_proj_dict)

        query_states = query_hidden.to_dense()
        key_states = key_hidden.to_dense()
        value_states = value_hidden.to_dense()

        bsz, q_len, _, _ = query_states.shape  # [B, T, H, D]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # TODO: fater decoding by caching KV
        cos, sin = self.rotary_emb(value_states, seq_len=h.seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)

        attn_output_hidden = HiddenState.from_dense(attn_output, h.token_type_mask)
        attn_output_hidden = attn_output_hidden.apply_all_type_mapping(self.o_proj_dict)

        return attn_output_hidden


class TextOnly(nn.Module):
    """
    When the decoder layer is less than LLaMA layer, we use this to only process text

    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.text_layer = LlamaDecoderLayer(config)

    def forward(
        self,
        h: HiddenState,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> HiddenState:
        x, attention_mask = h.to_single_type_batch(TokenType.Text, attention_mask)
        x = self.text_layer(x, attention_mask=attention_mask, position_ids=position_ids)

        x = x[0]  # LLaMA returns a tuple
        return h.update_single_type_batch(TokenType.Text, x)


class MixLayer(nn.Module):
    """
    The layer that mix the text and entity operations
    """

    def __init__(self, config: DecDeepFuseConfig) -> None:
        super().__init__()
        self.config = config

        self.attn = FusedAttn(config)
        self.mlp = nn.ModuleDict({k.name: MLP(config, k) for k in TokenType})

        self.attn_norm = make_norm_dict(config)
        self.mlp_norm = make_norm_dict(config)

    def forward(
        self,
        h: HiddenState,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> HiddenState:
        res = h
        h = h.apply_all_type_mapping(self.attn_norm)
        h = self.attn(h, attention_mask=attention_mask, position_ids=position_ids)
        h = h + res

        res = h
        h = h.apply_all_type_mapping(self.mlp_norm)
        h = h.apply_all_type_mapping(self.mlp)
        h = h + res

        return h


class SeperateLayer(nn.Module):
    """
    Both text and entity have their own layer
    """

    def __init__(self, config: DecDeepFuseConfig) -> None:
        super().__init__()
        self.config = config

        # TODO: note that the entity layer is not the same as the text layer
        self.layers = nn.ModuleDict(
            {k.name: LlamaDecoderLayer(config) for k in TokenType}
        )

    def forward(
        self,
        h: HiddenState,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> HiddenState:
        for token_type in TokenType:
            x, attention_mask_t = h.to_single_type_batch(token_type, attention_mask)
            x = self.layers[token_type.name](
                x, attention_mask=attention_mask_t, position_ids=position_ids
            )[0]
            h = h.update_single_type_batch(token_type, x)

        return h


class DecDeepFuse(Model):
    def __init__(self, config: DecDeepFuseConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.decoder_layers = nn.ModuleList([])

        for usage in config.layer_usage:
            if usage == "M":
                self.decoder_layers.append(MixLayer(config))
            elif usage == "S":
                self.decoder_layers.append(SeperateLayer(config))
            elif usage == "N":
                self.decoder_layers.append(TextOnly(config))
            else:
                raise ValueError(f"Unknown layer usage {usage}")

        self.final_norm = make_norm_dict(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.load_from_pretrained()

    def load_from_pretrained(self):
        """Load pretrained LLaMA and entity decoder"""
        logger.info("Loading pretrained models")

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
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (bsz, seq_len), x, 0
        )

        h = HiddenState.from_dense(x, batch.token_type_mask)

        for layer in self.decoder_layers:
            x = layer(
                h,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

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
