# -*- coding: utf-8 -*-
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import contextlib

import torch
from deepspeed.accelerator import get_accelerator
from torch import _C
from torch.utils.checkpoint import detach_variable

from megatron import get_args
from megatron.core.parallel_state import (
    get_data_parallel_rank,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.utils import safely_set_viewless_tensor_data
from megatron.memory import allocate_mem_buff

from .utils import gather_split_1d_tensor, split_tensor_into_1d_equal_chunks

# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = "model-parallel-rng"
_DATA_PARALLEL_RNG_TRACKER_NAME = 'data-parallel-rng'

# Whether apply model parallelsim to checkpointed hidden states.
_CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER = None


def init_checkpointed_activations_memory_buffer():
    """Initializ the memory buffer for the checkpointed activations."""
    args = get_args()

    per_layer = (
        args.micro_batch_size
        * args.max_position_embeddings
        * args.hidden_size
        // args.tensor_model_parallel_size
    )
    assert (
        args.num_layers % args.checkpoint_num_layers == 0
    ), "number of layers is not divisible by checkpoint-num-layers"
    num_checkpointer_layers = args.num_layers // args.checkpoint_num_layers
    numel = per_layer * num_checkpointer_layers
    dtype = torch.half
    if not args.fp16:
        dtype = torch.float

    global _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER
    assert (
        _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is None
    ), "checkpointed activations memory buffer is already allocated."
    _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER = allocate_mem_buff(
        "checkpointed activations", numel, dtype, track_usage=False
    )


def reset_checkpointed_activations_memory_buffer():
    """Reset the memory used for checkpointing."""
    if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
        _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER.reset()


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with get_accelerator().device(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device(get_accelerator().device_name())
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(get_accelerator().device_name(), device)

        def cb():
            idx = device.index
            if idx is None:
                idx = get_accelerator().current_device()
            default_generator = get_accelerator().default_generator(idx)
            default_generator.set_state(new_state)

    get_accelerator().lazy_call(cb)


def get_data_parallel_rng_tracker_name():
    global _DATA_PARALLEL_RNG_TRACKER_NAME
    return _DATA_PARALLEL_RNG_TRACKER_NAME

class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        self.reset()

    def is_initialized(self):
        return self._is_initialized

    def reset(self):
        """Set to the initial state (no tracker)."""

        # Track if initialized.
        self._is_initialized = False

        # Map from a string name to the cuda rng state.
        self.states_ = {}

        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self._is_initialized = True
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        self._is_initialized = True
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception('cuda rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
_CUDA_RNG_STATE_TRACKER_INITIALIZED = True


def initialize_rng_tracker(use_te_rng_tracker: bool = False):
    global _CUDA_RNG_STATE_TRACKER
    global _CUDA_RNG_STATE_TRACKER_INITIALIZED
    if _CUDA_RNG_STATE_TRACKER_INITIALIZED:
        return
    if use_te_rng_tracker:
        try:
            import transformer_engine.pytorch as te
        except:
            raise RuntimeError("use_te_rng_tracker requires TransformerEngine, but not installed")
    if use_te_rng_tracker:
        _CUDA_RNG_STATE_TRACKER = te.distributed.CudaRNGStatesTracker()
    else:
        _CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    _CUDA_RNG_STATE_TRACKER_INITIALIZED = True

def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    initialize_rng_tracker()
    return _CUDA_RNG_STATE_TRACKER


def model_parallel_cuda_manual_seed(seed):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel GPUs but different across
                       different model paralle groups. This is used for
                       example for dropout in the non-tensor-model-parallel regions.
        tensor-model-parallel state: This state is different among a set of model
                              parallel GPUs, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    tensor_model_parallel_seed = offset + get_tensor_model_parallel_rank()
    # Data parallel gets the original seed.
    data_parallel_seed = seed

    if torch.distributed.get_rank() == 0:
        print(
            "> initializing model parallel cuda seeds on global rank {}, "
            "model parallel rank {}, and data parallel rank {} with "
            "model parallel seed: {} and data parallel seed: {}".format(
                torch.distributed.get_rank(),
                get_tensor_model_parallel_rank(),
                get_data_parallel_rank(),
                tensor_model_parallel_seed,
                data_parallel_seed,
            ),
            flush=True,
        )
    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    get_accelerator().manual_seed(data_parallel_seed)
    _CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)
    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(
        _MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed
    )


class CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
    two main changes:
        1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
        2) the states in the model parallel tracker are also properly
           tracked/set/reset.
    """

    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, *args):
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = get_accelerator().get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        with torch.no_grad():
            outputs = run_function(*args)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0],
                split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True),
            )

        # HACK: currently when DeepSpeed is used, we always set
        # distribute_saved_activations to false, and use the following older
        # activation checkpointing mechanisms
        if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
            ctx.input_0_shape = args[0].data.shape
            args[0].data = split_tensor_into_1d_equal_chunks(args[0].data)
            args[0].data = _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER.add(args[0].data)

        # Store everything.
        ctx.save_for_backward(*args)

        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        inputs = ctx.saved_tensors
        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape),
            )
        # HACK: currently when DeepSpeed is used, we always set
        # distribute_saved_activations to false, and use the following older
        # activation checkpointing mechanisms
        if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
            inputs[0].data = gather_split_1d_tensor(inputs[0].data)
            inputs[0].data = inputs[0].data.view(ctx.input_0_shape)

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = get_accelerator().get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif (
            len(outputs) == 2
            and isinstance(outputs[1], torch.Tensor)
            and torch.equal(
                outputs[1], torch.tensor(0).to(get_accelerator().device_name())
            )
        ):
            # a hacky solution to overcome issue when running old script examples/pretrain_gpt_distributed.sh
            outputs = (outputs[0],)
        torch.autograd.backward(outputs, args)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else inp
            for inp in detached_inputs
        )
        return (None, None) + grads


def checkpoint(function, distribute_saved_activations, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    return CheckpointFunction.apply(function, distribute_saved_activations, *args)
