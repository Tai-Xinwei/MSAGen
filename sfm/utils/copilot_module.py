# -*- coding: utf-8 -*-
import glob
import os
import re as regex
from functools import partial

import torch
import torch.nn as nn
from deepspeed import comm as dist
from deepspeed.runtime import utils as ds_utils
from deepspeed.runtime.activation_checkpointing import checkpointing
from deepspeed.runtime.state_dict_factory import SDLoaderFactory
from deepspeed.utils import logger
from graphormer.utils.pretrained_layer_spec import LoraLayerSpec, PretrainedLayerSpec

# from deepspeed.runtime.pipe.topology import PipeDataParallelTopology, PipelineParallelGrid
from .myPipelineParallelGrid import PipeDataParallelTopology, myPipelineParallelGrid
from .mypp_module import PipelineModule


class PipelineError(Exception):
    """Errors related to the use of deepspeed.PipelineModule"""


class LayerSpec:
    """Building block for specifying pipeline-parallel modules.

    LayerSpec stores the type information and parameters for each stage in a
    PipelineModule. For example:

    .. code-block:: python

        nn.Sequence(
            torch.nn.Linear(self.in_dim, self.hidden_dim, bias=False),
            torch.nn.Linear(self.hidden_hidden, self.out_dim)
        )

    becomes

    .. code-block:: python

        layer_specs = [
            LayerSpec(torch.nn.Linear, self.in_dim, self.hidden_dim, bias=False),
            LayerSpec(torch.nn.Linear, self.hidden_hidden, self.out_dim)]
        ]
    """

    def __init__(self, typename, *module_args, **module_kwargs):
        self.typename = typename
        self.module_args = module_args
        self.module_kwargs = module_kwargs

        if not issubclass(typename, nn.Module):
            raise RuntimeError("LayerSpec only supports torch.nn.Module types.")

        if dist.is_initialized():
            self.global_rank = dist.get_rank()
        else:
            self.global_rank = -1

    def __repr__(self):
        return ds_utils.call_to_str(
            self.typename.__name__, self.module_args, self.module_kwargs
        )

    def build(self, log=False):
        """Build the stored specification."""
        if log:
            logger.info(f"RANK={self.global_rank} building {repr(self)}")

        return self.typename(*self.module_args, **self.module_kwargs)


class TiedLayerSpec(LayerSpec):
    def __init__(
        self,
        key,
        typename,
        *module_args,
        forward_fn=None,
        tied_weight_attr="weight",
        **module_kwargs,
    ):
        super().__init__(typename, *module_args, **module_kwargs)
        self.key = key
        self.forward_fn = forward_fn
        self.tied_weight_attr = tied_weight_attr


class CopilotModule(PipelineModule):
    """Modules to be parallelized with pipeline parallelism.

    The key constraint that enables pipeline parallelism is the
    representation of the forward pass as a sequence of layers
    and the enforcement of a simple interface between them. The
    forward pass is implicitly defined by the module ``layers``. The key
    assumption is that the output of each layer can be directly fed as
    input to the next, like a ``torch.nn.Sequence``. The forward pass is
    implicitly:

    .. code-block:: python

        def forward(self, inputs):
            x = inputs
            for layer in self.layers:
                x = layer(x)
            return x

    .. note::
        Pipeline parallelism is not compatible with ZeRO-2 and ZeRO-3.

    Args:
        layers (Iterable): A sequence of layers defining pipeline structure. Can be a ``torch.nn.Sequential`` module.
        num_stages (int, optional): The degree of pipeline parallelism. If not specified, ``topology`` must be provided.
        topology (``deepspeed.runtime.pipe.ProcessTopology``, optional): Defines the axes of parallelism axes for training. Must be provided if ``num_stages`` is ``None``.
        loss_fn (callable, optional): Loss is computed ``loss = loss_fn(outputs, label)``
        seed_layers(bool, optional): Use a different seed for each layer. Defaults to False.
        seed_fn(type, optional): The custom seed generating function. Defaults to random seed generator.
        base_seed (int, optional): The starting seed. Defaults to 1234.
        partition_method (str, optional): The method upon which the layers are partitioned. Defaults to 'parameters'.
        activation_checkpoint_interval (int, optional): The granularity activation checkpointing in terms of number of layers. 0 disables activation checkpointing.
        activation_checkpoint_func (callable, optional): The function to use for activation checkpointing. Defaults to ``deepspeed.checkpointing.checkpoint``.
        checkpointable_layers(list, optional): Checkpointable layers may not be checkpointed. Defaults to None which does not additional filtering.
    """

    def __init__(
        self,
        layers,
        num_stages=None,
        topology=None,
        loss_fn=None,
        seed_layers=False,
        seed_fn=None,
        base_seed=1234,
        partition_method="parameters",
        part_list=None,
        activation_checkpoint_interval=0,
        activation_checkpoint_func=checkpointing.checkpoint,
        checkpointable_layers=None,
        device=None,
        freeze_layer=None,
    ):
        super(CopilotModule, self).__init__(
            layers,
            num_stages,
            topology,
            loss_fn,
            seed_layers,
            seed_fn,
            base_seed,
            partition_method,
            part_list,
            activation_checkpoint_interval,
            activation_checkpoint_func,
            checkpointable_layers,
            device,
        )
        self.freeze_layer = freeze_layer

    def forward(self, forward_input):
        # We need to offset the seed by the microbatch ID. Save it in a local var to
        # ensure it is preserved in the closure. Otherwise checkpointed forward funcs
        # will see a different offset.
        self.micro_offset += 1

        def exec_range_func(start, end):
            """Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            """
            local_micro_offset = self.micro_offset + 1

            def exec_func(*inputs):
                # Single tensor inputs need to be unwrapped
                if len(inputs) == 1:
                    inputs = inputs[0]
                for idx, layer in enumerate(self.forward_funcs[start:end]):
                    self.curr_layer = idx + self._local_start
                    if self.seed_layers:
                        new_seed = (
                            self.base_seed * local_micro_offset
                        ) + self.curr_layer
                        if self.seed_fn:
                            self.seed_fn(new_seed)
                        else:
                            ds_utils.set_random_seed(new_seed)
                    if (
                        self.freeze_layer is not None
                        and self.curr_layer in self.freeze_layer
                    ):
                        with torch.no_grad():
                            inputs = layer(inputs)
                    else:
                        inputs = layer(inputs)
                return inputs

            return exec_func

        if self.activation_checkpoint_interval == 0:
            func = exec_range_func(0, len(self.forward_funcs))
            x = func(forward_input)
        else:
            num_layers = len(self.forward_funcs)
            x = forward_input
            for start_idx in range(0, num_layers, self.activation_checkpoint_interval):
                end_idx = min(
                    start_idx + self.activation_checkpoint_interval, num_layers
                )

                funcs = self.forward_funcs[start_idx:end_idx]
                # Since we either pass tensors or tuples of tensors without unpacking, we
                # need to be careful not to double-wrap tensors with tuple.
                if not isinstance(x, tuple):
                    x = (x,)

                if self._is_checkpointable(funcs):
                    x = self.activation_checkpoint_func(
                        exec_range_func(start_idx, end_idx), *x
                    )
                else:
                    x = exec_range_func(start_idx, end_idx)(*x)
        return x
