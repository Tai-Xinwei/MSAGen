# -*- coding: utf-8 -*-
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Utilities for models."""

import math

import torch
from deepspeed.runtime.zero import GatheredParameters

from megatron import get_args


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def gather_and_init(param, init_method):
    with GatheredParameters(param, modifier_rank=0):
        init_method(param)


def attention_mask_func(attention_scores, attention_mask):
    args = get_args()
    attention_score_mask = torch.zeros_like(attention_scores)
    if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
        attention_mask_ = attention_mask
        actual_seqlen = attention_scores.size()[2]
        if actual_seqlen != attention_mask_.size()[2]:
            # attention_mask has size [1, 1, seqlen, seqlen]
            attention_mask_ = attention_mask_[
                :, :, :actual_seqlen, :actual_seqlen
            ].contiguous()
        attention_score_mask.masked_fill_(attention_mask_, torch.finfo(attention_scores.dtype).min)
    else:
        attention_score_mask.masked_fill_(attention_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores + attention_score_mask


def get_linear_layer(rows, columns, init_method, gather_params_on_init=False):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if get_args().perform_initialization:
        with GatheredParameters(
            layer.weight, modifier_rank=0, enable=gather_params_on_init
        ):
            init_method(layer.weight)
    with torch.no_grad():
        with GatheredParameters(
            layer.bias, modifier_rank=0, enable=gather_params_on_init
        ):
            layer.bias.zero_()
    return layer


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return (
        0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    )


def openai_gelu(x):
    return gelu_impl(x)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return (
        x
        * 0.5
        * (
            torch.erf(x / 1.41421).to(dtype=x.dtype)
            + torch.ones_like(x).to(dtype=x.dtype)
        )
    )
