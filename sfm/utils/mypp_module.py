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
from deepspeed.runtime.pipe.topology import (
    PipeDataParallelTopology,
    PipelineParallelGrid,
)
from deepspeed.runtime.state_dict_factory import SDLoaderFactory

# from deepspeed.utils import logger
from sfm.logging import logger
from sfm.utils.pretrained_layer_spec import PretrainedLayerSpec, TiedPretrainedLayerSpec

# from .myPipelineParallelGrid import PipeDataParallelTopology, myPipelineParallelGrid


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


def partition_by_layers(binary_weights, num_stages, num_total_layers):
    if num_stages <= 1:
        parts = [0, num_total_layers]
    else:
        num_total_part_layers = int(torch.sum(torch.tensor(binary_weights)))
        if num_total_part_layers < num_stages:
            raise ValueError(
                f"number of partitioned layers {num_total_part_layers} is smaller than the number of pipeline stages {num_stages}"
            )
        min_layers_per_stage = num_total_part_layers // num_stages
        num_rest_layers = num_total_part_layers % num_stages
        num_layers_by_stages = [min_layers_per_stage] * num_stages
        if num_rest_layers > 0:
            if num_rest_layers == 1:
                # assign to second stage
                num_layers_by_stages[1] += 1
            else:
                # skip first stage, specialized for generalist
                rest_layer_step_size = (num_stages - 2) // (num_rest_layers - 1)
                for stage_id in range(
                    1, 1 + num_rest_layers * rest_layer_step_size, rest_layer_step_size
                ):
                    num_layers_by_stages[stage_id] += 1
        num_layers_by_stages_cum_sum = torch.tensor(num_layers_by_stages).cumsum(dim=-1)
        parts = [0]
        cur_weight = binary_weights[0]
        cur_stage = 0
        for layer_id in range(num_total_layers - 1):
            cur_weight += binary_weights[layer_id + 1]
            if cur_weight > num_layers_by_stages_cum_sum[cur_stage]:
                parts.append(layer_id + 1)
                cur_stage += 1
        parts.append(num_total_layers)
        assert (
            int(num_layers_by_stages_cum_sum[-1]) == num_total_part_layers
        ), f"{int(num_layers_by_stages_cum_sum[-1])} vs. {num_total_part_layers}"
    assert (
        len(parts) == num_stages + 1
    ), f"{len(parts)} vs. {num_stages + 1}, {num_layers_by_stages_cum_sum}"
    return parts


class PipelineModule(nn.Module):
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
        loss_log_dict={},
    ):
        super().__init__()

        if num_stages is None and topology is None:
            raise RuntimeError("must provide num_stages or topology")

        self.micro_offset = 0
        self.device = f"cuda:{device}"
        self.loss_fn = loss_fn

        self.loss_log_dict = {}
        for k, v in loss_log_dict.items():
            self.loss_log_dict[k] = torch.Tensor([loss_log_dict[k]]).cuda()

        self.checkpointable_layers = checkpointable_layers
        if checkpointable_layers is not None:
            assert isinstance(
                checkpointable_layers, list
            ), "param `checkpointable_layers` must be type of list."

        self.seed_layers = seed_layers
        self.seed_fn = seed_fn
        self.base_seed = base_seed
        if dist.get_rank() == 0:
            try:
                seed_str = self.seed_fn.__name__
            except AttributeError:
                seed_str = None
            print(
                f"SEED_LAYERS={self.seed_layers} BASE_SEED={self.base_seed} SEED_FN={seed_str}"
            )

        # Setup world info
        self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
        self.global_rank = dist.get_rank(group=self.world_group)
        self.world_size = dist.get_world_size(group=self.world_group)
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        assert self.local_rank is not None

        if topology:
            self._topo = topology
            self.num_stages = self._topo.get_dim("pipe")
        else:
            self.num_stages = num_stages
            if topology is None:
                if self.world_size % self.num_stages != 0:
                    raise RuntimeError(
                        f"num_stages ({self.num_stages}) must divide distributed world size ({self.world_size})"
                    )
                dp = self.world_size // num_stages
                topology = PipeDataParallelTopology(num_pp=num_stages, num_dp=dp)
                self._topo = topology

        # Construct communicators for pipeline topology
        self._grid = PipelineParallelGrid(
            process_group=self.world_group, topology=self._topo
        )

        self.stage_id = self._topo.get_coord(self.global_rank).pipe

        # Initialize partition information
        logger.info(
            f"RANK={self.global_rank} initializing pipeline, num_stages={self.num_stages}, num_layers={len(layers)}"
        )
        self._layer_specs = list(layers)
        self._num_layers = len(self._layer_specs)
        self._local_start = 0
        self._local_stop = None
        self._partition_layers(method=partition_method, part_list=part_list)

        self.forward_funcs = []
        # self.forward_funcs = nn.ModuleList().to(f'cuda:{self.local_rank}')
        self.fwd_map = {}
        self.tied_modules = nn.ModuleDict()
        self.tied_weight_attrs = {}

        # Offset the random seed by the stage ID.
        # newseed = torch.cuda.initial_seed() + self._grid.get_stage_id()
        # ds_utils.set_random_seed(newseed)

        # with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        self._build()
        self.to(f"cuda:{self.local_rank}")

        self.tied_comms = self._index_tied_modules()
        self._synchronize_tied_weights()

        self.activation_checkpoint_interval = activation_checkpoint_interval
        self.activation_checkpoint_func = activation_checkpoint_func

    def _build(self):
        specs = self._layer_specs

        for local_idx, layer in enumerate(specs[self._local_start : self._local_stop]):
            layer_idx = local_idx + self._local_start
            if self.seed_layers:
                if self.seed_fn:
                    self.seed_fn(self.base_seed + layer_idx)
                else:
                    ds_utils.set_random_seed(self.base_seed + layer_idx)

            # Recursively build PipelineModule objects
            if isinstance(layer, PipelineModule):
                raise NotImplementedError("RECURSIVE BUILD NOT YET IMPLEMENTED")

            # LayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, nn.Module):
                name = str(layer_idx)
                self.forward_funcs.append(layer)
                self.fwd_map.update({name: len(self.forward_funcs) - 1})
                self.add_module(name, layer)

            # TiedLayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, (TiedLayerSpec)):
                # Build and register the module if we haven't seen it before.
                if layer.key not in self.tied_modules:
                    self.tied_modules[layer.key] = layer.build(self.device)
                    self.tied_weight_attrs[layer.key] = layer.tied_weight_attr

                if layer.forward_fn is None:
                    # Just use forward()
                    self.forward_funcs.append(self.tied_modules[layer.key])
                else:
                    # User specified fn with args (module, input)
                    self.forward_funcs.append(
                        partial(layer.forward_fn, self.tied_modules[layer.key])
                    )

            elif isinstance(layer, (TiedPretrainedLayerSpec)):
                # Build and register the module if we haven't seen it before.
                if layer.key not in self.tied_modules:
                    self.tied_modules[layer.key] = layer.build(self.device, load=True)
                    self.tied_weight_attrs[layer.key] = layer.tied_weight_attr

                if layer.forward_fn is None:
                    # Just use forward()
                    self.forward_funcs.append(self.tied_modules[layer.key])
                else:
                    # User specified fn with args (module, input)
                    self.forward_funcs.append(
                        partial(layer.forward_fn, self.tied_modules[layer.key])
                    )

            # LayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, LayerSpec):
                module = layer.build()
                name = str(layer_idx)
                self.forward_funcs.append(module)
                self.fwd_map.update({name: len(self.forward_funcs) - 1})
                self.add_module(name, module)

            elif isinstance(layer, (PretrainedLayerSpec)):
                module = layer.build(self.device, load=True)
                name = str(layer_idx)
                self.forward_funcs.append(module)
                self.fwd_map.update({name: len(self.forward_funcs) - 1})
                self.add_module(name, module)

            # Last option: layer may be a functional (e.g., lambda). We do nothing in
            # that case and just use it in forward()
            else:
                self.forward_funcs.append(layer)

        # All pipeline parameters should be considered as model parallel in the context
        # of our FP16 optimizer
        for p in self.parameters():
            p.ds_pipe_replicated = False

    def _count_layer_params(self):
        """Count the trainable parameters in individual layers.

        This routine will only build one layer at a time.

        Returns:
            A list of the number of parameters in each layer.
        """
        param_counts = [0] * len(self._layer_specs)
        for idx, layer in enumerate(self._layer_specs):
            if isinstance(layer, (LayerSpec, PretrainedLayerSpec)):
                l = layer.build()
                params = filter(lambda p: p.requires_grad, l.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
            elif isinstance(layer, nn.Module):
                params = filter(lambda p: p.requires_grad, layer.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
        return param_counts

    def _find_layer_type(self, layername):
        idxs = []
        typeregex = regex.compile(layername, regex.IGNORECASE)
        for idx, layer in enumerate(self._layer_specs):
            name = None
            if isinstance(layer, (LayerSpec, PretrainedLayerSpec)):
                name = layer.typename.__name__
            elif isinstance(layer, nn.Module):
                name = layer.__class__.__name__
            else:
                try:
                    name = layer.__name__
                except AttributeError:
                    continue
            if typeregex.search(name):
                idxs.append(idx)

        if len(idxs) == 0:
            raise RuntimeError(
                f"Partitioning '{layername}' found no valid layers to partition."
            )
        return idxs

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

    def _partition_layers(self, method="uniform", part_list=None):
        num_stages = self._topo.get_dim("pipe")
        stage_id = self._topo.get_coord(self.global_rank).pipe

        if self.global_rank == 0:
            logger.info(f"Partitioning pipeline stages with method {method}")

        method = method.lower()

        # Each stage gets a simple uniform number of layers.
        if method == "uniform":
            num_layers = len(self._layer_specs)
            self.parts = ds_utils.partition_uniform(
                num_items=num_layers, num_parts=num_stages
            )
        elif method == "parameters":
            param_counts = self._count_layer_params()
            self.parts = ds_utils.partition_balanced(
                weights=param_counts, num_parts=num_stages
            )
        elif method.startswith("type:"):
            layertype = method.split(":")[1]
            binary_weights = [0] * len(self._layer_specs)
            for idx in self._find_layer_type(layertype):
                binary_weights[idx] = 1
            self.parts = partition_by_layers(
                binary_weights, num_stages, len(self._layer_specs)
            )
        elif method == "profile":
            raise NotImplementedError(f"Partitioning method {method} not implemented.")
        elif method == "manual":
            assert part_list is not None
            # param_counts = self._count_layer_params()
            self.parts = part_list
        else:
            raise NotImplementedError(f"Partitioning method {method} not implemented.")

        # Print some information on the partitioning.
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f"stage={stage} layers={stop - start}")
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    # name = str(layer)
                    if isinstance(layer, (LayerSpec, PretrainedLayerSpec)):
                        name = layer.typename.__name__
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    if method != "manual" and (not method.startswith("type:")):
                        print(
                            f"    {idx+start:2d}: {name} -- {param_counts[start+idx]/1e6:.3f}M"
                        )
                    else:
                        print(f"    {idx+start:2d}: {name}")
            if self.loss_fn:
                try:
                    print(f"  loss: {self.loss_fn.__name__}")
                except AttributeError:
                    print(f"  loss: {self.loss_fn.__class__.__name__}")

        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])

    def allreduce_tied_weight_gradients(self):
        """All reduce the gradients of the tied weights between tied stages"""
        for key, comm in self.tied_comms.items():
            weight = getattr(self.tied_modules[key], comm["weight_attr"])
            dist.all_reduce(weight.grad, group=comm["group"])

    def get_tied_weights_and_groups(self):
        weight_group_list = []
        for key, comm in self.tied_comms.items():
            weight = getattr(self.tied_modules[key], comm["weight_attr"])
            weight_group_list.append((weight, comm["group"]))
        return weight_group_list

    def _synchronize_tied_weights(self):
        for key, comm in self.tied_comms.items():
            dist.broadcast(
                getattr(comm["module"], comm["weight_attr"]),
                src=min(comm["ranks"]),
                group=comm["group"],
            )

    def _index_tied_modules(self):
        """Build communication structures for tied modules."""
        tied_comms = {}
        if self._topo.get_dim("pipe") == 1:
            return tied_comms

        specs = self._layer_specs
        tie_keys = set(
            s.key
            for s in specs
            if isinstance(s, (TiedLayerSpec, TiedPretrainedLayerSpec))
        )
        for key in tie_keys:
            # Find the layers that the tied module appears in
            tied_layers = []
            for idx, layer in enumerate(specs):
                if (
                    isinstance(layer, (TiedLayerSpec, TiedPretrainedLayerSpec))
                    and layer.key == key
                ):
                    tied_layers.append(idx)
            # Find all stages with this tied module
            # TODO: Would be nice to remove the nested data/model parallelism loops and
            # TODO: instead generalize in some way, since we really just care about the
            # TODO: stage that owns the tied layer. Then loop over each (dp, mp, ...)
            # TODO: fiber to generate process groups.
            tied_stages = set(self.stage_owner(idx) for idx in tied_layers)
            for dp in range(self._grid.data_parallel_size):
                for mp in range(self._grid.get_slice_parallel_world_size()):
                    tied_ranks = []
                    for s in sorted(tied_stages):
                        if self._grid.get_slice_parallel_world_size() > 1:
                            tied_ranks.append(
                                self._grid.stage_to_global(
                                    stage_id=s, data=dp, model=mp
                                )
                            )
                        else:
                            tied_ranks.append(
                                self._grid.stage_to_global(stage_id=s, data=dp)
                            )
                    group = dist.new_group(ranks=tied_ranks)

                    # Record this tied module if we own a local copy of it.
                    if self.global_rank in tied_ranks:
                        assert key in self.tied_modules
                        if key in self.tied_modules:
                            tied_comms[key] = {
                                "ranks": tied_ranks,
                                "group": group,
                                "weight_attr": self.tied_weight_attrs[key],
                                "module": self.tied_modules[key],
                            }
                            # Only count the tied module once in the eyes of the FP16 optimizer
                            if self.global_rank != tied_ranks[0]:
                                for p in self.tied_modules[key].parameters():
                                    p.ds_pipe_replicated = True
        """
        if len(tied_comms) > 0:
            print(f'RANK={self.global_rank} tied_comms={tied_comms}')
        """

        return tied_comms

    def partitions(self):
        return self.parts

    def stage_owner(self, layer_idx):
        assert 0 <= layer_idx < self._num_layers
        for stage in range(self._topo.get_dim("pipe")):
            if self.parts[stage] <= layer_idx < self.parts[stage + 1]:
                return stage
        raise RuntimeError(f"Layer {layer_idx} not owned? parts={self.parts}")

    def _set_bounds(self, start=None, stop=None):
        """Manually define the range of layers that will be built on this process.

        These boundaries are treated as list slices and so start is inclusive and stop is
        exclusive. The default of None for both results in all layers being built
        locally.
        """
        self._local_start = start
        self._local_stop = stop

    def set_checkpoint_interval(self, interval):
        assert interval >= 0
        self.checkpoint_interval = interval

    def topology(self):
        """ProcessTopology object to query process mappings."""
        return self._topo

    def mpu(self):
        return self._grid

    def num_pipeline_stages(self):
        return self._topo.get_dim("pipe")

    def ckpt_prefix(self, checkpoints_path, tag):
        """Build a prefix for all checkpoint files written by this module."""
        # All checkpoint files start with this
        rank_name = "module"

        # Data parallelism is omitted from the naming convention because we are agnostic
        # to this in the checkpoint.
        omit_dims = frozenset(["data"])
        axes = [a for a in self._grid._topo.get_axis_names() if a not in omit_dims]
        for dim in axes:
            rank = getattr(self._grid._topo.get_coord(rank=self.global_rank), dim)
            rank_name += f"-{dim}_{rank:02d}"

        ckpt_name = os.path.join(checkpoints_path, str(tag), rank_name)
        return ckpt_name

    def ckpt_layer_path(self, ckpt_dir, local_layer_idx):
        """Customize a prefix for a specific pipeline module layer."""
        idx = local_layer_idx + self._local_start
        layer_ckpt_path = os.path.join(ckpt_dir, f"layer_{idx:02d}")
        rank_repr = self._grid._topo.get_rank_repr(rank=self.global_rank)
        if rank_repr != "":
            layer_ckpt_path += f"-{rank_repr}"
        layer_ckpt_path += "-model_states.pt"
        return layer_ckpt_path

    def ckpt_layer_path_list(self, ckpt_dir, local_layer_idx, model_ckpt_list=None):
        """Get all ckpt file list for a specific pipeline module layer."""
        idx = local_layer_idx + self._local_start
        if model_ckpt_list is None:
            layer_ckpt_path = os.path.join(ckpt_dir, f"layer_{idx:02d}-")
            layer_ckpt_path += "*model_states.pt"
            ckpt_files = glob.glob(layer_ckpt_path)
            ckpt_files.sort()
        else:
            ckpt_files = [model_ckpt_list[idx]]

        return ckpt_files

    def save_state_dict(self, save_dir, checkpoint_engine, exclude_frozen_params=False):
        # Processes having the same model parallel rank on different data parallel instances
        # have identical layer weights.  We can distribute the task of saving the layer weights
        # among the data parallel ranks.  For example, if a pipeline stage has 9 layers and
        # if there are 2 data parallel instances, rank 0 will save the first 5 layers and
        # rank 1 will save the last 4.
        dp_rank = self._grid.data_parallel_id
        dp_size = self._grid.data_parallel_size
        num_layers = len(self.forward_funcs)
        if self.checkpoint_parallel_write_pipeline:
            # spread layers evenly across data parallel ranks
            offsets = ds_utils.partition_uniform(num_layers, dp_size)
            start, end = offsets[dp_rank], offsets[dp_rank + 1]
        else:
            # data parallel rank 0 writes all layers
            if dp_rank != 0:
                return
            start, end = 0, num_layers
        layer_list = self.forward_funcs[start:end]

        os.makedirs(save_dir, exist_ok=True)
        for idx, layer in enumerate(layer_list):
            model_ckpt_path = self.ckpt_layer_path(save_dir, start + idx)
            if not hasattr(layer, "state_dict"):
                continue
            # We pass cloned tensors to torch.save() to avoid checkpoint bloat which occurs because torch.save()
            # saves the underlying storage rather than the slice of the storage corresponding to individual tensors.
            # This is a problem in DeepSpeed because we often allocate tensors using slices of large flattened buffers.
            # Tensor cloning helps to avoid this problem because the storage of cloned tensors are closer to the true size.
            # It is expected that the garbage collector will reclaim the cloned tensor storage to avoid memory bloat.
            # See https://pytorch.org/docs/stable/notes/serialization.html#preserve-storage-sharing
            orig_state_dict = layer.state_dict()
            final_state_dict = type(orig_state_dict)(
                {
                    k: v.clone()
                    for k, v in orig_state_dict.items()
                    if k.find("_extra_state") == -1
                }
            )
            checkpoint_engine.save(final_state_dict, model_ckpt_path)

    def load_state_dir(self, load_dir, checkpoint_engine, strict=True):
        for idx, layer in enumerate(self.forward_funcs):
            # Functions, etc. will not have state_dicts
            if not hasattr(layer, "load_state_dict"):
                continue

            # get all checkpoint files for the layer.
            model_ckpt_list = self.ckpt_layer_path_list(load_dir, idx)
            mp_rank = self._grid.get_slice_parallel_rank()
            mp_world_size = self._grid.get_slice_parallel_world_size()

            sd_loader = SDLoaderFactory.get_sd_loader(
                model_ckpt_list, version=2.0, checkpoint_engine=checkpoint_engine
            )
            load_path, checkpoint, _ = sd_loader.load(
                mp_world_size, mp_rank, module_key=None, is_pipe_parallel=True
            )

            layer.load_state_dict(checkpoint, strict=strict)

            # if self._grid.data_parallel_id == 0:
            #     logger.info(
            #         f'RANK={self.global_rank} Loaded layer={idx+self._local_start} file={load_path}'
            #     )

        self._synchronize_tied_weights()

    def fast_load_state_dir(
        self,
        checkpoint_engine,
        load_dir=None,
        strict=True,
        model_ckpt_list=None,
        mfm_lora=False,
    ):
        for idx, layer in enumerate(self.forward_funcs):
            # Functions, etc. will not have state_dicts
            if not hasattr(layer, "load_state_dict"):
                continue

            # get all checkpoint files for the layer.
            model_ckpt = self.ckpt_layer_path_list(
                load_dir, idx, model_ckpt_list=model_ckpt_list
            )
            mp_rank = self._grid.get_slice_parallel_rank()
            mp_world_size = self._grid.get_slice_parallel_world_size()

            sd_loader = SDLoaderFactory.get_sd_loader(
                model_ckpt, version=2.0, checkpoint_engine=checkpoint_engine
            )
            load_path, checkpoint, _ = sd_loader.load(
                mp_world_size, mp_rank, module_key=None, is_pipe_parallel=True
            )

            ckpt_name = model_ckpt[0]
            # print("model check:", ckpt_name, type(ckpt_name))

            if (
                ckpt_name.split("/")[-1] == "model.hybrid_emb.pt"
                or ckpt_name.split("/")[-1] == "layer_37-model_states.pt"
            ):
                strict = False
            else:
                strict = True

            # if mfm_lora:
            #     nl = int(ckpt_name.split('-')[0].split('_')[-1])
            #     if nl > 0 and nl < 37:
            #         for k, v in checkpoint.item():

            layer.load_state_dict(checkpoint, strict=strict)

            # if self._grid.data_parallel_id == 0:
            #     logger.info(
            #         f'RANK={self.global_rank} Loaded layer={idx+self._local_start} file={load_path}'
            #     )

        self._synchronize_tied_weights()

    def _is_checkpointable(self, funcs):
        # This is an unfortunate hack related to torch and deepspeed activation checkpoint implementations.
        # Some layers like torch.nn.Embedding will not receive grads if checkpointed, which breaks things.
        # I presume it's related to the discrete inputs that cannot require_grad? Need to revisit.
        if self.__class__.__name__ in ("GPTModelPipe", "GPT2ModelPipe"):
            return all(
                "ParallelTransformerLayerPipe" in f.__class__.__name__ for f in funcs
            )
        if self.checkpointable_layers is not None:
            return all(
                f.__class__.__name__ in self.checkpointable_layers for f in funcs
            )

        params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]
        return any(len(list(p)) > 0 for p in params)
