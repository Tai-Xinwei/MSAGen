# -*- coding: utf-8 -*-
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from megatron.core import ModelParallelConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal


@dataclass
class TransformerConfig(ModelParallelConfig):
    """Configuration object for megatron-core transformers.

    Attributes:

    # model architecture
    num_layers (int): Number of transformer layers in a transformer block.
    hidden_size (int): Transformer hidden size.
    ffn_hidden_size (int): Transformer Feed-Forward Network hidden size.
                            This is set to 4*hidden_size if not provided. Defaults to None.')
    num_attention_heads (int): Number of transformer attention heads.
    num_key_value_heads (int): This is the number of key_value heads that should be used to implement Grouped Query Attention. If
                               `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
                               `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used.
                               For more details checkout [this paper](https://arxiv.org/pdf/2305.13245.pdf).
                               If it is not specified, will default to `num_attention_heads`.
    kv_channels (int): Projection weights dimension in multi-head attention.
                        This is set to hidden_size // num_attention_heads if not provided.
                        Defaults to None.
    hidden_dropout (float): Dropout probability for transformer hidden state. Defaults to 0.1.
    attention_dropout (float): Post attention dropout probability. Defaults to 0.1.
    fp32_residual_connection (bool): If true, move residual connections to fp32.
    apply_residual_connection_post_layernorm (bool): If true, uses the original BERT residule connection ordering.
                                                     Defaults to False.
    layernorm_epsilon (float): Layernorm epsilon. Defaults to 1e-5.

    layernorm_zero_centered_gamma (bool): if set to 'True', the LayerNorm is adjusted to center the gamma values
                                          around 0. This improves numerical stability. Defaults to False.

    add_bias_linear (bool): Include a bias term in all linear layers (QKV projections, after core attention, and two
                            in MLP layer). Default is True.

    gated_linear_unit (bool): Use a gated linear unit for the first linear layer in the MLP. Defaults to False.

    activation_func (Callable): Activation function to use for the non-linearity in the MLP. Defaults to F.gelu.

    # initialization
    init_method (Callable): Method to initialize weights. Note that bias is always set to
                            zero. Should be a function that takes a single Tensor and
                            initializes it. Defaults to
                            megatron.core.utils.init_method_normal(init_method_std) which is
                            torch.nn.init.normal_ with mean=0.0 and std=init_method_Std.

    output_layer_init_method (Callable): Method to initialize weights of the output layer of
                                         both attention and MLP blocks. Defaults to
                                         megatron.core.utils.scaled_init_method_normal(init_method_std)
                                         which is torch.nn.init.normal_ with mean=0.0 and
                                         std=init_method_std / math.sqrt(2.0 * num_layers).

    init_method_std (float): Standard deviation of the zero mean normal for the default
                             initialization method, not used if init_method and
                             output_layer_init_method are provided. Defaults to 0.02.

    # mixed-precision
    apply_query_key_layer_scaling (bool): If true, scale Q * K^T by 1 / layer-number. Defaults to True.
    attention_softmax_in_fp32 (bool): If true, run attention masking and softmax in fp32.
                                      This should be true if apply_query_key_layer_scaling is true.

    # fusion
    bias_gelu_fustion (bool): If true, fuses bias and gelu. Defaults to False.
    masked_softmax_fusion (bool): If true, uses softmax fusion.
    persist_layer_norm (bool): If true, uses the persistent fused layer norm kernel.
                               This kernel only supports a fixed set of hidden sizes.
                               Defaults to False.
    bias_dropout_fusion (bool): If true, uses bias dropout fusion.

    # activation recomputation

    recompute_granularity (str): megatron-core supports 'selective' activation checkpointing where only the memory
                                 intensive part of attention is checkpointed.  These memory intensive activations
                                 are also less compute intensive which makes activation checkpointing more efficient
                                 for LLMs (20B+).  See Reducing Activation Recomputation in Large Transformer
                                 Models: https://arxiv.org/abs/2205.05198 for more details.  'full' will checkpoint
                                 the entire transformer layer.  Must be 'selective' or 'full'. Defaults to None.

    recompute_method (str): uniform will uniformly divide the total number of transformer layers in a transformer
                            block and recompute the input activation of each divided chunk at the specified
                            granularity.  block will recompute the input activations for only a set number of
                            transformer layers per pipeline stage.  The rest of the layers in the pipeline stage
                            will not have any activations recomputed.  Must be 'uniform' or 'block'. Defaults to
                            None.

    recompute_num_layers (int): When recompute_method is uniform, recompute_num_layers is the number of transformer
                                layers in each uniformly divided recompute unit.  When recompute_method is block,
                                recompute_num_layers is the number of transformer layers to recompute within each
                                pipeline stage.  Defaults to None.

    distribute_saved_activations (bool): If true, distribute recomputed activations across the model parallel
                                         group. Defaults to None.

    """

    # model architecture
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = None

    ffn_hidden_size: int = None
    kv_channels: int = None
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    fp32_residual_connection: bool = False
    # @jcasper should we keep this option?
    apply_residual_connection_post_layernorm: bool = False
    layernorm_epsilon: float = 1e-5
    layernorm_zero_centered_gamma: bool = False
    add_bias_linear: bool = True
    gated_linear_unit: bool = False
    activation_func: Callable = F.gelu

    # initialization
    init_method: Callable = None
    output_layer_init_method: Callable = None
    init_method_std: float = 0.02

    # mixed-precision
    apply_query_key_layer_scaling: bool = True
    attention_softmax_in_fp32: bool = True

    # communication

    # fusion
    bias_gelu_fusion: bool = False  # TODO: this should be bias_activation_fusion ?
    masked_softmax_fusion: bool = False
    persist_layer_norm: bool = False
    bias_dropout_fusion: bool = False  # TODO: this should be bias_dropout_add_fusion?

    # activation recomputation
    recompute_granularity: str = None
    recompute_method: str = None
    recompute_num_layers: int = None
    distribute_saved_activations: bool = None

    ####################
    # MoE related
    ####################
    moe_router_load_balancing_type: str = "aux_loss"
    """Determines the load balancing strategy for the router. "aux_loss" corresponds to the load
    balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds to the balancing
    algorithm used in S-BASE, and "none" implies no load balancing."""

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_grouped_gemm: bool = False
    """When there are multiple experts per rank, compress multiple local (potentially small) gemms
    in a single kernel launch to improve the utilization and performance by leveraging the Grouped
    GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).

    """

    moe_aux_loss_coeff: float = 0  # 1e-2 would be a good start value for load balance loss.
    """Scaling coefficient for the aux loss. A starting value of 1e-2 is recommended."""

    moe_z_loss_coeff: float = None  # 1e-3 would be a good start value for z-loss
    """Scaling coefficient for the z-loss. A starting value of 1e-3 is recommended."""

    moe_input_jitter_eps: float = None
    """Add noise to the input tensor by applying jitter with a specified epsilon value."""

    moe_token_dropping: bool = False  # TODO: Support token dropping.
    """This feature involves selectively dropping and padding tokens for each expert to achieve a
    specified capacity, similar to GShard, Switch-Transformer, and DeepSpeed-MoE. Note that this is
    currently unsupported so should remain False."""

    moe_token_dispatcher_type: str = "allgather"
    """The type of token dispatcher to use. The default is 'allgather'. Options are 'allgather' and 'alltoall'."""
    moe_per_layer_logging: bool = False
    """Enable per-layer logging for MoE, currently supports auxiliary loss and z loss."""

    ####################
    # miscellaneous
    ####################
    clone_scatter_output_in_embedding: bool = True
    """When set to True, clone the output of scatter_to_sequence_parallel_region in embedding layer
    to facilitate garbage collection of input."""

    disable_parameter_transpose_cache: bool = False
    """When set to true, the parameter transposes are not cached for subsequent iterations."""

    enable_cuda_graph: bool = False
    """When set to true, TransformerLayer blocks are wrapped with CUDA graph."""

    # These 2 attributes are WAR for TRTLLM export. DO NOT USE!! WILL BE DEPRECATED SOON!!
    max_position_embeddings: int = 0
    """Deprecated. Do not use."""

    rotary_percent: float = 0
    """Deprecated. Do not use."""

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        super().__post_init__()
        if self.fp16 and self.bf16:
            raise ValueError(
                f"Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True."
            )

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0

        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.recompute_granularity is not None:
            if self.recompute_granularity not in ["full", "selective"]:
                raise ValueError(
                    f'When using recompute_granuarlity: {self.recompute_granularity} must be "full" or "selective".'
                )

            if self.recompute_method is not None:
                if self.recompute_method not in ["block", "uniform"]:
                    raise ValueError(
                        f'recompute_method: {self.recompute_method} must be "block" or "uniform".'
                    )
            else:
                raise ValueError(
                    f'Using recompute_granularity: {self.recompute_granularity} so recompute_method must be "block" or "uniform"'
                )

            if self.recompute_num_layers is None:
                raise ValueError(
                    f"When using recompute_granularity: {self.recompute_granularity} so recompute_num_layers must be between "
                    f"1 and num_layers_per_pipeline_rank: {self.num_layers // self.pipeline_model_parallel_size}"
                )

            if self.distribute_saved_activations and self.sequence_parallel_enabled:
                raise ValueError(
                    f"distribute_saved_activations: {self.distribute_saved_activations} must be false when sequence parallel is enabled: {self.sequence_parallel_enabled}"
                )

            if self.virtual_pipeline_model_parallel_size is not None:
                if not self.num_layers % self.virtual_pipeline_model_parallel_size == 0:
                    raise ValueError(
                        f"num_layers: {self.num_layers} must be divisible by virtual_model_parallel_size {self.virtual_pipeline_model_parallel_size}"
                    )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.bias_gelu_fusion:
            if not self.add_bias_linear:
                raise ValueError(
                    "When bias_gelu_fusion is True, add_bias_linear must also be True."
                )

            if self.activation_func != F.gelu:
                raise ValueError(
                    "When bias_gelu_fusion is True, activation_func must be F.gelu."
                )

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std, self.num_layers
            )
