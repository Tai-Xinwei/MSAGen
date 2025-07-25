# -*- coding: utf-8 -*-
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Gradient clipping."""

import torch

try:
    from torch._six import inf as inf
except ModuleNotFoundError:
    from torch import inf as inf

from deepspeed.accelerator import get_accelerator

if get_accelerator().device_name() == "cuda":
    from apex.multi_tensor_apply import multi_tensor_applier
    import amp_C

from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.model.module import param_is_not_shared


def clip_grad_norm_fp32(
    parameters, grads_for_norm, max_norm, norm_type=2, model_parallel_group=None
):
    """Clips gradient norm of an iterable of parameters whose gradients
       are in fp32.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        grads_for_norm (Iterable[Tensor]): an iterable of Tensors or a single
            Tensor that will be used for calculating the grad norm.
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        model_parallel_group (group): given the nature of the distributed
            optimizer, this is passed as an argument.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []
    for param in parameters:
        if param.grad is not None:
            assert param.grad.type() == "torch.{}.FloatTensor".format(
                get_accelerator().device_name()
            )
            grads.append(param.grad.detach())

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda,
            op=torch.distributed.ReduceOp.MAX,
            group=model_parallel_group,
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            if get_accelerator().device_name() == "cuda":
                dummy_overflow_buf = torch.cuda.IntTensor([0])
                # Use apex's multi-tensor applier for efficiency reasons.
                # Multi-tensor applier takes a function and a list of list
                # and performs the operation on that list all in one kernel.
                if grads_for_norm:
                    grad_norm, _ = multi_tensor_applier(
                        amp_C.multi_tensor_l2norm,
                        dummy_overflow_buf,
                        [grads_for_norm],
                        False,  # no per-parameter norm
                    )
                else:
                    grad_norm = torch.cuda.FloatTensor([0])
            else:
                grad_norm = torch.norm(grads_for_norm, p=2.0)
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm**norm_type
        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm**norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group
        )
        total_norm = total_norm.item() ** (1.0 / norm_type)

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        if get_accelerator().device_name() == "cuda":
            dummy_overflow_buf = get_accelerator().IntTensor([0])
            multi_tensor_applier(
                amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff
            )
        else:
            for g in grads:
                g.detach().mul_(clip_coeff.to(g.device))

    return total_norm


def count_zeros_fp32(parameters, model_parallel_group):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    total_num_zeros = torch.cuda.FloatTensor([0.0])
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grad = param.grad.detach()
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros = num_zeros + total_num_zeros

    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(
        total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group
    )

    total_num_zeros = total_num_zeros.item()

    return total_num_zeros
