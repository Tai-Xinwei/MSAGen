# -*- coding: utf-8 -*-

import torch
from megablocks.layers import common, dmoe, moe
from megablocks.layers.arguments import Arguments as MegablocksArguments
from megablocks.layers.moe import mlp
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.mixtral.modeling_mixtral import (
    MixtralFlashAttention2,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)

from sfm.logging import logger
from sfm.models.nlm.moe_config import MoeModelConfig
from sfm.utils.pipelinemode import pipemode

try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm as RMSNorm

    logger.info("using apex fused RMSNorm")
except ImportError:
    logger.info("using MixtralRMSNorm")
    RMSNorm = MixtralRMSNorm

# see https://github.com/databricks/megablocks/issues/83
# reduce memory usage
mlp.MLP = lambda a: None


def to_magablocks_config(config: MoeModelConfig) -> MegablocksArguments:
    return MegablocksArguments(
        # Model arguments.
        hidden_size=config.hidden_size,
        ffn_hidden_size=config.intermediate_size,
        num_layers=1,
        bias=False,
        return_bias=False,
        activation_fn=ACT2FN[config.hidden_act],
        # MoE arguments.
        moe_num_experts=config.num_local_experts,
        moe_top_k=config.num_experts_per_tok,
        moe_capacity_factor=1.0,  # Seems to be unused in DMOE
        moe_normalize_expert_weights=1.0,
        moe_loss_weight=config.router_aux_loss_coef,
        moe_jitter_eps=config.router_jitter_noise,
        # Parallelism arguments.
        moe_expert_model_parallelism=False,
        expert_parallel_group=None,
        moe_weight_parallelism=False,
        weight_parallel_group=None,
        pipeline_model_parallel_size=1,
        num_layers_per_virtual_pipeline_stage=None,
        # Compute arguments
        memory_optimized_mlp=config.moe_memory_optimized_mlp,
        mlp_type="glu",
        mlp_impl=config.moe_impl,  # grouped, sparse
        # Initialization arguments.
        fp16=config.fp16,
        bf16=config.bf16,
        device=torch.tensor([]).device,  # the default device
    )


class MoeEmbeddingsPP(nn.Module):
    def __init__(self, config: MoeModelConfig):
        super().__init__()
        self.config = config
        self.learnable_cutoff = config.learnable_cutoff
        self.dummy = torch.nn.Linear(1, 1)  # Make DeepSpeed happy

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        self.embed_tokens.weight.register_hook(self.freeze_parital_weight_hook)

        self.param_dict = {
            "input_ids": torch.Tensor,
        }

    @pipemode
    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        text_embeds = self.embed_tokens(input_ids)
        position_ids = (
            torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            .expand(bsz, -1)
            .contiguous()
        )

        gate_scores = torch.zeros(
            self.config.num_hidden_layers, bsz, seq_len, self.config.num_local_experts
        ).to(self.embed_tokens.weight)

        return text_embeds, position_ids, gate_scores

    def freeze_parital_weight_hook(self, grad):
        grad[: self.learnable_cutoff, :] = 0
        return grad


class SafeMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.config.router_jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.config.router_jitter_noise,
                1.0 + self.config.router_jitter_noise,
            )
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=1, dtype=torch.float
        )

        routing_scores = routing_weights.reshape(batch_size, sequence_length, -1)

        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0 and self.training:
                # When no tokens are selected, we need to mock the current_state
                # Or there will be no gradient for the expert
                # Then the PP will fail
                # see https://github.com/microsoft/DeepSpeed/issues/5066
                top_x_ = torch.zeros(1).to(hidden_states.device).to(torch.int32)
                top_x_list = top_x_.tolist()
                current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
                fake_state = expert_layer(current_state * 0)
                final_hidden_states.index_add_(
                    0, top_x_, fake_state.to(hidden_states.dtype)
                )
            else:
                # in torch it is faster to index using lists than torch tensors
                top_x_list = top_x.tolist()
                idx_list = idx.tolist()

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
                current_hidden_states = (
                    expert_layer(current_state)
                    * routing_weights[top_x_list, idx_list, None]
                )

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )

        # Note: as MegaBlocks returns the rougting scores rather than the logits,
        # we need to return the scores here
        return final_hidden_states, routing_scores


class MegaBlockMoeBlock(dmoe.dMoE):
    def __init__(self, config: MoeModelConfig):
        args = to_magablocks_config(config)
        super().__init__(args)
        self.config = config
        self.args = args

    def forward(self, x):
        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth.
        x = common.cast_if_autocast_enabled(x)

        # Compute the expert scores and assignments.
        scores, expert_weights, top_experts = self.router(x)

        # Compute the experts.
        ret = self.experts(x, scores, expert_weights, top_experts)

        # We compute the loss by ourselves
        # so we need to clear the loss by MegaBlocks
        moe.clear_load_balancing_loss()
        return ret, scores

    def state_dict(self):
        """
        Build a state dict that is compatible with mixtral.
        """
        router_state_dict = self.router.layer.state_dict()

        state_dict = dict()
        state_dict["gate.weight"] = router_state_dict["weight"]
        for i in range(self.config.num_local_experts):
            slice_begin = i * self.config.intermediate_size
            slice_end = (i + 1) * self.config.intermediate_size
            state_dict[f"experts.{i}.w1.weight"] = self.experts.mlp.w1[
                slice_begin:slice_end
            ].T
            state_dict[f"experts.{i}.w2.weight"] = self.experts.mlp.w2[
                slice_begin:slice_end
            ]
            state_dict[f"experts.{i}.w3.weight"] = self.experts.mlp.v1[
                slice_begin:slice_end
            ].T

        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        Load a state dict that is compatible with mixtral.
        """
        router_state_dict = dict()
        router_state_dict["weight"] = state_dict["gate.weight"]
        self.router.layer.load_state_dict(router_state_dict, strict, assign)

        w1 = torch.zeros(self.config.hidden_size, self.config.intermediate_size)
        w2 = torch.zeros(self.config.hidden_size, self.config.intermediate_size)
        v1 = torch.zeros(self.config.hidden_size, self.config.intermediate_size)

        for i in range(self.config.num_local_experts):
            slice_begin = i * self.config.intermediate_size
            slice_end = (i + 1) * self.config.intermediate_size
            w1[slice_begin:slice_end] = state_dict[f"experts.{i}.w1.weight"].T
            w2[slice_begin:slice_end] = state_dict[f"experts.{i}.w2.weight"]
            v1[slice_begin:slice_end] = state_dict[f"experts.{i}.w3.weight"].T

        expert_state_dict = {"w1": w1, "w2": w2, "v1": v1}

        self.experts.mlp.load_state_dict(expert_state_dict, strict, assign)


class MoeDecoderLayerPP(nn.Module):
    def __init__(self, config: MoeModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.dummy = torch.nn.Linear(1, 1)  # Make DeepSpeed happy

        self.self_attn = MixtralFlashAttention2(config, layer_idx=layer_idx)
        if config.moe_impl == "vanilla":
            self.block_sparse_moe = SafeMixtralSparseMoeBlock(config)
        else:
            self.block_sparse_moe = MegaBlockMoeBlock(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

        self.param_dict = {
            "hidden_states": torch.Tensor,
            "position_ids": torch.Tensor,
            "gate_scores": torch.Tensor,
        }

    @pipemode
    def forward(self, hidden_states, position_ids, gate_scores):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=None,  # default to causal attention
            position_ids=position_ids,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_scores = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        gate_scores[self.layer_idx] = router_scores

        return hidden_states, position_ids, gate_scores


class MoeNormPP(nn.Module):
    def __init__(self, config: MoeModelConfig):
        super().__init__()
        self.config = config
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dummy = torch.nn.Linear(1, 1)  # Make DeepSpeed happy

        self.param_dict = {
            "hidden_states": torch.Tensor,
            "position_ids": torch.Tensor,
            "gate_scores": torch.Tensor,
        }

    @pipemode
    def forward(self, hidden_states, position_ids, gate_scores):
        return self.norm(hidden_states), gate_scores


class MoeHeadPP(nn.Module):
    def __init__(self, config: MoeModelConfig):
        super().__init__()
        self.config = config
        self.learnable_cutoff = config.learnable_cutoff
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight.register_hook(self.freeze_parital_weight_hook)
        self.dummy = torch.nn.Linear(1, 1)  # Make DeepSpeed happy

        self.param_dict = {
            "hidden_states": torch.Tensor,
            "gate_scores": torch.Tensor,
        }

    @property
    def emb_weight(self):
        return self.lm_head.weight

    def freeze_parital_weight_hook(self, grad):
        grad[: self.learnable_cutoff, :] = 0
        return grad

    @pipemode
    def forward(self, hidden_states, gate_scores):
        return self.lm_head(hidden_states), gate_scores
