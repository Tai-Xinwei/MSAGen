# -*- coding: utf-8 -*-
import math
import os
from typing import Any, List, Mapping, Optional, Tuple

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
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from megatron.core import parallel_state, tensor_parallel
from megatron.model.enums import AttnMaskType, AttnType, LayerType
from megatron.model.language_model import Embedding
from megatron.model.rotary_pos_embedding import RotaryEmbedding
from megatron.model.transformer import (
    ParallelAttention,
    ParallelMLP,
    ParallelTransformerLayer,
)
from sfm.logging import logger
from sfm.models.llama2.llama_modules import LlamaDecoderLayerPP, LlamaHead, LlamaNorm
from sfm.modules.sfmmodule import SFMModule
from sfm.utils import PretrainedLayerSpec, TiedPretrainedLayerSpec

try:
    from apex.normalization import MixedFusedRMSNorm
except:
    raise ImportError("Please install apex from install/install_megatron.sh")


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.




    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class ParallelLlamaMLPAdapter(SFMModule):
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

        self.down_proj = tensor_parallel.RowParallelLinear(
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
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        reset the parameters of the weight
        """
        nn.init.uniform_(
            self.gate_proj.weight,
            -1.0 / math.sqrt(self.hidden_size),
            1.0 / math.sqrt(self.hidden_size),
        )
        nn.init.uniform_(
            self.down_proj.weight,
            -1.0 / math.sqrt(self.intermediate_size),
            1.0 / math.sqrt(self.intermediate_size),
        )
        nn.init.uniform_(
            self.up_proj.weight,
            -1.0 / math.sqrt(self.hidden_size),
            1.0 / math.sqrt(self.hidden_size),
        )

    def forward(self, x):
        gated_x, _ = self.gate_proj(x)
        up_x, _ = self.up_proj(x)

        intermidiate_x = self.act_fn(gated_x) * up_x
        x, _ = self.down_proj(intermidiate_x)

        return x


class ParallelLlamaMLP(SFMModule):
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


class ParallelLlamaAttention(SFMModule):
    def __init__(
        self,
        config,
        layer_number,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
    ):
        super(ParallelLlamaAttention, self).__init__(config=config)
        self.hidden_size = config.hidden_size
        rotary_dim = (
            config.hidden_size // config.num_attention_heads
            if config.kv_channels is None
            else config.kv_channels
        )

        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.seq_length = config.seq_length
        self.num_key_value_groups = (
            config.num_attention_heads // self.num_key_value_heads
        )
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.num_attention_heads_per_tp = (
            self.num_attention_heads // config.tensor_model_parallel_size
        )
        self.num_key_value_heads_per_tp = (
            self.num_key_value_heads // config.tensor_model_parallel_size
        )

        self.q_proj = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
        )
        self.k_proj = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
        )
        self.v_proj = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
        )

        self.o_proj = tensor_parallel.RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
        )

        self.rotary_emb = LlamaRotaryEmbedding(
            rotary_dim, config.max_position_embeddings
        )

        # self.freqs_cis = precompute_freqs_cis(
        #     # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
        #     # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
        #     config.hidden_size // config.num_attention_heads,
        #     config.max_position_embeddings * 2,
        # ).cuda()

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb_func=None,
        rotary_pos_emb=None,
        position_ids=None,
    ):
        # hidden_states: (seq_len, batch_size, hidden_size)
        xq = self.q_proj(hidden_states)[0]
        xk = self.k_proj(hidden_states)[0]
        xv = self.v_proj(hidden_states)[0]

        bsz, seqlen, _ = xq.shape
        # seqlen, bsz, _ = xq.shape
        # xq = xq.transpose(0, 1)
        # xk = xk.transpose(0, 1)
        # xv = xv.transpose(0, 1)

        xq = xq.view(bsz, seqlen, self.num_attention_heads_per_tp, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_key_value_heads_per_tp, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_key_value_heads_per_tp, self.head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        kv_seq_len = xk.shape[-2]
        cos, sin = self.rotary_emb(xv, seq_len=kv_seq_len)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, position_ids)

        # repeat k/v heads if n_kv_heads < n_heads
        xk = repeat_kv(
            xk, self.num_key_value_groups
        )  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(
            xv, self.num_key_value_groups
        )  # (bs, seqlen, n_local_heads, head_dim)

        bs, nh, Slen, Hidden = xq.shape

        with torch.backends.cuda.sdp_kernel(
            enable_math=True, enable_mem_efficient=True, enable_flash=False
        ):
            context_layer = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=attention_mask
            )
        context_layer = (
            context_layer.transpose(1, 2).contiguous().reshape(bs, Slen, nh * Hidden)
        )

        return self.o_proj(context_layer)


class LlamaDecoderLayerMP(SFMModule):
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
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.self_attn = ParallelLlamaAttention(
            config,
            layer_number=1,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type,
        )

        self.mlp = ParallelLlamaMLP(config)

        self.dummy = torch.nn.Linear(1, 1)
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

        # self.rotary_pos_emb = RotaryEmbedding(rotary_dim)

    def auto_partition_load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        tp_model_size: int,
        tp_rank: int,
        strict: bool = True,
    ):
        # update key mapping
        keys = list(state_dict.keys())
        new_state_dict = {}
        for key in keys:
            param = state_dict[key]
            # if key == "self_attn.rotary_emb.inv_freq":
            #     self.self_attn.rotary_emb.inv_freq[:] = param[:]
            # elif key == "self_attn.o_proj.weight":
            #     new_state_dict["self_attn.dense.weight"] = param
            # elif key == "self_attn.q_proj.weight":
            #     q_proj_weight = param
            # elif key == "self_attn.k_proj.weight":
            #     k_proj_weight = param
            # elif key == "self_attn.v_proj.weight":
            #     v_proj_weight = param
            # else:
            new_state_dict[key] = param

        # new_state_dict["self_attn.query_key_value.weight"] = (
        #     torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
        #     .reshape(
        #         3,
        #         self.config.num_attention_heads,
        #         self.config.hidden_size // self.config.num_attention_heads,
        #         self.config.hidden_size,
        #     )
        #     .transpose(0, 1)
        #     .reshape(3 * self.config.hidden_size, self.config.hidden_size)
        # )

        del state_dict

        return super().auto_partition_load_state_dict(
            state_dict=new_state_dict,
            tp_model_size=tp_model_size,
            tp_rank=tp_rank,
            strict=strict,
        )

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
        hidden_states, attention_mask_bool, position_ids = input_tuple

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_mask = torch.zeros_like(
            attention_mask_bool, dtype=hidden_states.dtype, device=hidden_states.device
        )
        attention_mask.masked_fill_(
            ~attention_mask_bool, torch.finfo(hidden_states.dtype).min
        )

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            # rotary_pos_emb_func=lambda x: self.rotary_pos_emb(x, self.seq_length),
            # rotary_pos_emb=rotary_pos_emb,
            position_ids=position_ids,
        )[0]
        hidden_states = residual + hidden_states
        # torch.save({f"hidden_states": hidden_states}, f"/home/peiran/mnt/mntsfm2/output/hidden_states_mid_mp_0.pt")

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # torch.save({f"hidden_states_postln{self.no_layer}": hidden_states}, f"/home/peiran/mnt/mntsfm2/output/hidden_states_postln_mp_{self.no_layer}.pt")
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # torch.save({f"hidden_states{self.no_layer}": hidden_states}, f"/home/peiran/mnt/mntsfm2/output/hidden_states_mp_{self.no_layer}.pt")
        outputs = (
            hidden_states.contiguous(),
            attention_mask_bool.contiguous(),
            position_ids.contiguous(),
        )
        return outputs


class FusedLlamaNorm(SFMModule):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        # self.norm = MixedFusedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        hidden_states, _, _ = input_tuple

        hidden_states = self.norm(hidden_states)
        # logger.info(f"hidden_states -1: {hidden_states}")
        # torch.save({"hidden_states-1": hidden_states}, "/home/peiran/mnt/mntsfm2/output/hidden_states_mp.pt")

        return (hidden_states,)


def lm_logits_fn(embedding, input_tuple):
    """LM logits using word embedding weights."""
    input_ = input_tuple[0]
    logits = F.linear(input_, embedding.emb_weight)
    return logits


class LlamaEmbeddingsMP(Embedding, SFMModule):
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

    def auto_partition_load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        tp_model_size: int,
        tp_rank: int,
        strict: bool = True,
    ):
        keys = list(state_dict.keys())
        new_state_dict = {}
        for key in keys:
            param = state_dict[key]
            if key == "embed_tokens.weight":
                if param.size()[0] < self.config.vocab_size:
                    mean_embedding = torch.mean(param, dim=0, keepdim=True).expand(
                        [self.config.vocab_size - param.size()[0], param.size()[1]]
                    )
                    std_embedding = torch.std(param, dim=0, keepdim=True).expand(
                        [self.config.vocab_size - param.size()[0], param.size()[1]]
                    )
                    param = torch.cat(
                        [param, torch.normal(mean=mean_embedding, std=std_embedding)],
                        dim=0,
                    )
                new_state_dict["word_embeddings.weight"] = param

        del state_dict

        return super().auto_partition_load_state_dict(
            new_state_dict, tp_model_size, tp_rank, strict
        )

    def freeze_parital_weight_hook(self, grad):
        # offset the learnable cutoff by vocabulary partitioning in tensor parallel
        if self.learnable_cutoff >= self.word_embeddings.vocab_end_index:
            grad[:, :] = 0
        elif self.learnable_cutoff > self.word_embeddings.vocab_start_index:
            grad[
                : self.learnable_cutoff - self.word_embeddings.vocab_start_index, :
            ] = 0
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
        # Get text embeddings from language model
        mol_idx_mask = input_ids < 0  # B, T

        text_embeds = super().forward(
            input_ids.masked_fill(mol_idx_mask, 0), position_ids
        )
        # transpose to [batch_size, seq_len, hidden_size] for hybrid embedding
        text_embeds = text_embeds.transpose(0, 1)

        return mol_emb, mol_padding_mask, text_embeds, llm_mask, input_ids


class NumMLPMP(nn.Module):
    def __init__(
        self,
        config,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(NumMLPMP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.fc1 = tensor_parallel.ColumnParallelLinear(
            input_size=in_features,
            output_size=hidden_features,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=True,
        )
        self.act = act_layer()
        self.fc2 = tensor_parallel.RowParallelLinear(
            input_size=hidden_features,
            output_size=1,
            config=config,
            init_method=config.output_layer_init_method,
            bias=True,
            input_is_parallel=True,
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)[0]
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)[0]
        x = self.drop(x)
        return x


class LlamaHeadMP(SFMModule):
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

        self.lm_head.weight.register_hook(self.freeze_parital_weight_hook)

        self.learnable_cutoff = learnable_cutoff
        self.num_head = NumMLPMP(config, config.hidden_size, 4 * config.hidden_size, 1)

    @property
    def emb_weight(self):
        return self.lm_head.weight

    def auto_partition_load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        tp_model_size: int,
        tp_rank: int,
        strict: bool = True,
    ):
        keys = list(state_dict.keys())
        new_state_dict = {}
        for key in keys:
            param = state_dict[key]
            if key == "lm_head.weight":
                if param.size()[0] < self.config.vocab_size:
                    mean_embedding = torch.mean(param, dim=0, keepdim=True).expand(
                        [self.config.vocab_size - param.size()[0], param.size()[1]]
                    )
                    std_embedding = torch.std(param, dim=0, keepdim=True).expand(
                        [self.config.vocab_size - param.size()[0], param.size()[1]]
                    )
                    param = torch.cat(
                        [param, torch.normal(mean=mean_embedding, std=std_embedding)],
                        dim=0,
                    )
                new_state_dict[key] = param

        del state_dict

        return super().auto_partition_load_state_dict(
            new_state_dict, tp_model_size, tp_rank, strict
        )

    def freeze_parital_weight_hook(self, grad):
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        linear_column_start = self.lm_head.output_size_per_partition * tp_rank
        linear_column_end = linear_column_start + self.lm_head.output_size_per_partition
        if self.learnable_cutoff >= linear_column_end:
            grad[:, :] = 0
        elif self.learnable_cutoff > linear_column_start:
            grad[: self.learnable_cutoff - linear_column_start, :] = 0
        return grad

    def forward(self, inputs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        #  demension of hidden_states [seq_len, batch_size, hidden_size]

        if isinstance(inputs, tuple):
            hidden_states = inputs[0]
        else:
            hidden_states = inputs

        lm_logits = self.lm_head(hidden_states)[0]  # .transpose(0, 1)

        num_logits = self.num_head(hidden_states)

        return lm_logits, num_logits


class LlamaModelMP(LlamaPreTrainedModel):
    def __init__(self, args, config: LlamaConfig):
        super().__init__(config)
        self.dummy = nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=True
        )

        self.pipe_layer = []

    @classmethod
    def to_layers(
        cls, args, config, new_num_tokens=None, load_ckpt=False, layer_id=0, ckp_list=[]
    ):
        cls.pipe_layer = []
        for i in range(config.num_hidden_layers):
            cls.pipe_layer.append(
                PretrainedLayerSpec(
                    LlamaDecoderLayerMP,
                    config,
                    i,
                    load_ckpt=load_ckpt,
                    pretrained_ckpt_path=os.path.join(
                        args.llm_model_name_or_path, f"model.layers.{i}.pt"
                    ),
                    lora_mode="freeze",
                    tp_model_size=args.tensor_model_parallel_size,
                    tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                    self_attn_mask_type=AttnMaskType.causal,
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
                tp_model_size=args.tensor_model_parallel_size,
                tp_rank=parallel_state.get_tensor_model_parallel_rank(),
            )
        )
        layer_id += 1
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
                tp_model_size=args.tensor_model_parallel_size,
                tp_rank=parallel_state.get_tensor_model_parallel_rank(),
            )
        )

        return cls.pipe_layer


class LlamaForCausalLMTP(LlamaForCausalLM):
    def __init__(self, args, config: LlamaConfig):
        super().__init__(config)
        self.dummy = nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=True
        )
