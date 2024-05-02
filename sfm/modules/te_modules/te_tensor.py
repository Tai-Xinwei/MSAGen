# -*- coding: utf-8 -*-
import dataclasses
import os
from importlib.metadata import version
from typing import Callable

import torch
import transformer_engine as te
from torch import Tensor

from megatron.core import ModelParallelConfig
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


def _get_extra_te_kwargs(config: TransformerConfig):
    extra_transformer_engine_kwargs = {
        "params_dtype": config.params_dtype,
    }
    extra_transformer_engine_kwargs["device"] = torch.cuda.current_device()
    return extra_transformer_engine_kwargs


def condition_init_method(config, init_method):
    return init_method if config.perform_initialization else (lambda w: None)


class TELayerNorm:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm`
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(
        cls,
        hidden_size: int,
        eps: float = 1e-5,
    ):
        instance = te.pytorch.LayerNorm(
            hidden_size=hidden_size,
            eps=eps,
        )
        return instance


class TERMSNorm:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `RMSNorm`
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(
        cls,
        hidden_size: int,
        eps: float = 1e-5,
    ):
        instance = te.pytorch.RMSNorm(
            hidden_size=hidden_size,
            eps=eps,
        )
        return instance


class TELinear(te.pytorch.Linear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group().
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: str,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        tp_comm_buffer_name: str = None,
    ):
        self.config = config

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = (
            self.config.disable_parameter_transpose_cache
        )
        if skip_weight_param_allocation:
            raise ValueError(
                "Transformer Engine linear layers do not support skip_weight_param_allocation"
            )

        extra_kwargs = _get_extra_te_kwargs(config)

        extra_kwargs["ub_split_ag"] = False
        extra_kwargs["ub_atomic_gemm_ag"] = False
        extra_kwargs["ub_split_rs"] = False
        extra_kwargs["ub_atomic_gemm_rs"] = False

        assert (
            tp_comm_buffer_name is not None
        ), "Buffer name should be set to configure communication overlap settings"
        extra_kwargs["ub_name"] = tp_comm_buffer_name

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=get_tensor_model_parallel_group(check_initialized=False),
            tp_size=self.config.tensor_model_parallel_size,
            get_rng_state_tracker=get_cuda_rng_tracker
            if get_cuda_rng_tracker().is_initialized()
            else None,
            # init_method=condition_init_method(config, init_method),
            bias=bias,
            return_bias=self.te_return_bias,
            parallel_mode=parallel_mode,
            **extra_kwargs,
        )

    def forward(self, x):
        # _is_first_microbatch = (
        #     None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        # )
        _is_first_microbatch = None
        out = super().forward(x, is_first_microbatch=_is_first_microbatch)
        self.is_first_microbatch = False

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None


class TEColumnParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `ColumnParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool = True,
        is_expert: bool = False,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: str = None,
    ):
        if gather_output:
            raise ValueError(
                "Transformer Engine linear layers do not support gather_output = True"
            )

        if is_expert:
            raise ValueError("Transformer Engine linear layers do not yet support MoE")

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
        )

    # def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
    #     """ Sharding along axis 0, bias sharded """
    #     state_dict = self.state_dict(prefix='', keep_vars=True)
    #     return make_sharded_tensors_for_checkpoint(
    #         state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
    #     )


class TERowParallelLinear(TELinear):
    """
    Wrapper for the Transformer-Engine's `Linear` layer but specialized similar
    to megatron's `RowParallelLinear` layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool = True,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,
    ):
        if not input_is_parallel:
            raise ValueError(
                "Transformer Engine linear layers do not support input_is_parallel = False"
            )

        if is_expert:
            raise ValueError("Transformer Engine linear layers do not yet support MoE")

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=condition_init_method(config, init_method),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,  # We don't currently use this for row parallel layers
            tp_comm_buffer_name=tp_comm_buffer_name,
        )

    # def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
    #     """ Sharding along axis 1, bias not sharded """
    #     state_dict = self.state_dict(prefix='', keep_vars=True)
    #     return make_sharded_tensors_for_checkpoint(
    #         state_dict, prefix, {'weight': 1}, sharded_offsets
    #     )
