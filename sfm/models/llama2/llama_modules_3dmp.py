# -*- coding: utf-8 -*-
import math
import os
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    LlamaConfig,
    LlamaPreTrainedModel,
    LlamaTokenizer,
    LlamaTokenizerFast,
)
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
)

from megatron.core import parallel_state, tensor_parallel
from megatron.model.enums import AttnMaskType, AttnType, LayerType
from megatron.model.language_model import Embedding
from megatron.model.module import MegatronModule
from megatron.model.rotary_pos_embedding import RotaryEmbedding
from megatron.model.transformer import (
    ParallelAttention,
    ParallelMLP,
    ParallelTransformerLayer,
)
from sfm.logging import logger
from sfm.utils import PretrainedLayerSpec, TiedPretrainedLayerSpec

try:
    from apex.normalization import MixedFusedRMSNorm
except:
    raise ImportError("Please install apex from install/install.sh")


class ParallelLlamaMLPAdapter(MegatronModule):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str,
        dropout: float = 0.0,
        enable_expert_tensor_parallelism: bool = False,
        moe: bool = False,
    ):
        super().__init__()
        # gated parallel mlp
        self.gate_proj = tensor_parallel.ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
        )

        self.down_proj = self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            intermediate_size,
            output_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
        )

        self.up_proj = tensor_parallel.ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
        )

        self.act_fn = ACT2FN[hidden_act]
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        gated_x, _ = self.gate_proj(x)
        up_x, _ = self.up_proj(x)

        intermidiate_x = self.act_fn(gated_x) * up_x

        x, _ = self.down_proj(intermidiate_x)

        return x


class ParallelLlamaMLP(MegatronModule):
    def __init__(
        self,
        config: LlamaConfig,
        enable_expert_tensor_parallelism=False,
        moe=False,
    ):
        super().__init__()
        self.config = config
        self.hidden_act = config.hidden_act

        # gated parallel mlp
        self.gate_proj = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
        )

        self.down_proj = tensor_parallel.RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
        )

        self.up_proj = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
        )

        self.act_fn = ACT2FN[self.hidden_act]

    def forward(self, x):
        gated_x, _ = self.gate_proj(x)
        up_x, _ = self.up_proj(x)

        intermidiate_x = self.act_fn(gated_x) * up_x

        x, _ = self.down_proj(intermidiate_x)

        return x


class LlamaDecoderLayerMP(MegatronModule):
    def __init__(
        self,
        config,
        no_layer,
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.padding,
        drop_path_rate=0.0,
        num_experts=1,
    ):
        super(LlamaDecoderLayerMP, self).__init__()
        self.config = config
        self.input_layernorm = MixedFusedRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.self_attn = ParallelAttention(
            config,
            layer_number=1,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type,
        )

        self.post_attention_layernorm = MixedFusedRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = ParallelLlamaMLP(config)

        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.no_layer = no_layer

        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads
            if config.kv_channels is None
            else config.kv_channels
        )

        if config.rotary_percent < 1.0:
            rotary_dim = int(rotary_dim * config.rotary_percent)

        # partial rotary embeddings, which is better than full rotary
        # Wang and Komatsuzaki et al
        # https://github.com/kingoflolz/mesh-transformer-jax/
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim)

    def forward(
        self,
        input_tuple,
        # hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # output_attentions: Optional[bool] = False,
        # use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        hidden_states, attention_mask_bool = input_tuple
        # size of hidden_states: (seq_len, batch_size, hidden_size)

        # # convert mask from bool to int
        # attention_mask = attention_mask_bool.to(torch.int)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        batch_len = min(hidden_states.shape[0], self.seq_length)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask_bool,
            rotary_pos_emb=self.rotary_pos_emb(batch_len),
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attention_mask_bool)
        return outputs


class FusedLlamaNorm(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.norm = MixedFusedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        hidden_states, _ = input_tuple

        hidden_states = self.norm(hidden_states)

        return (hidden_states,)


def lm_logits_fn(embedding, input_tuple):
    """LM logits using word embedding weights."""
    input_ = input_tuple[0]
    logits = F.linear(input_, embedding.emb_weight)
    return logits


class LlamaEmbeddingsMP(Embedding):
    def __init__(self, config: LlamaConfig, learnable_cutoff: int = 32001):
        super().__init__(
            config.hidden_size,
            config.vocab_size,
            max_sequence_length=config.max_position_embeddings,
            embedding_dropout_prob=0.0,
            config=config,
            num_tokentypes=0,
            embedding_weights_in_fp32=False,
        )
        self.config = config
        self.learnable_cutoff = learnable_cutoff
        self.word_embeddings.weight.register_hook(self.freeze_parital_weight_hook)

    @property
    def emb_weight(self):
        return self.word_embeddings.weight

    def freeze_parital_weight_hook(self, grad):
        grad[: self.learnable_cutoff, :] = 0
        return grad

    def forward(
        self, input_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        mol_emb, mol_padding_mask, llm_mask, input_ids = input_tuple

        # Get text embeddings from language model
        ## set position_ids, it's not used.
        position_ids = torch.zeros_like(
            input_ids, device=input_ids.device, dtype=torch.long
        )
        # export demension of text embeddings [seq_len, batch_size, hidden_size]
        text_embeds = super().forward(input_ids, position_ids)
        # transpose to [batch_size, seq_len, hidden_size] for hybrid embedding
        text_embeds = text_embeds.transpose(0, 1)

        return mol_emb, mol_padding_mask, text_embeds, llm_mask, input_ids


class LlamaHeadMP(MegatronModule):
    def __init__(self, config: LlamaConfig, learnable_cutoff: int = 32001):
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.lm_head = tensor_parallel.ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            config=config,
            init_method=config.init_method,
        )

    @property
    def emb_weight(self):
        return self.lm_head.weight

    def freeze_parital_weight_hook(self, grad):
        grad[: self.learnable_cutoff, :] = 0
        return grad

    def forward(self, inputs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        #  demension of hidden_states [seq_len, batch_size, hidden_size]

        if isinstance(inputs, tuple):
            hidden_states = inputs[0]
        else:
            hidden_states = inputs

        # export demension of lm_logits [seq_len, batch_size, hidden_size]
        lm_logits = self.lm_head(hidden_states)[0]

        return lm_logits


class LlamaModelMP(LlamaPreTrainedModel):
    def __init__(self, args, config: LlamaConfig):
        super().__init__(config)
        self.dummy = nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=True
        )

        self.pipe_layer = []

    @classmethod
    def to_layers(cls, args, config, new_num_tokens=None, load_ckpt=False):
        cls.pipe_layer = []
        for i in range(config.num_hidden_layers):
            cls.pipe_layer.append(
                PretrainedLayerSpec(
                    LlamaDecoderLayerMP,
                    config,
                    i,
                    load_ckpt=load_ckpt,
                    pretrained_ckpt_path=os.path.join(
                        args.llm_model_name_or_path, "model.layers.{}.pt".format(i)
                    ),
                    lora_mode="freeze",
                )
            )
        cls.pipe_layer.append(
            PretrainedLayerSpec(
                FusedLlamaNorm,
                config,
                load_ckpt=load_ckpt,
                pretrained_ckpt_path=os.path.join(
                    args.llm_model_name_or_path, "model.norm.pt"
                ),
                lora_mode="freeze",
            )
        )
        cls.pipe_layer.append(
            PretrainedLayerSpec(
                LlamaHeadMP,
                config,
                new_num_tokens=new_num_tokens,
                load_ckpt=load_ckpt,
                pretrained_ckpt_path=os.path.join(
                    args.llm_model_name_or_path, "model.lm_head.pt"
                ),
                lora_mode="freeze",
            )
        )

        return cls.pipe_layer


class LlamaForCausalLMTP(LlamaForCausalLM):
    def __init__(self, args, config: LlamaConfig):
        super().__init__(config)
        self.dummy = nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=True
        )
