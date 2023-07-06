import deepspeed
import os
from deepspeed import comm as dist
from deepspeed.utils.timer import ThroughputTimer
from deepspeed.utils import log_dist, OnDevice, logger

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.utils import PartitionedTensor
from deepspeed.runtime.dataloader import RepeatingLoader
# from deepspeed.runtime.pipe.module import PipelineModule, PipelineError
# from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.runtime.pipe import p2p, schedule
from deepspeed.runtime.engine import DeepSpeedEngine, DeepSpeedOptimizerCallable, DeepSpeedSchedulerCallable, MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.runtime import zero
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.utils import logger, log_dist, instrument_w_nvtx
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.utils import DummyOptim
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
from deepspeed.runtime.utils import clip_grad_norm_

from typing import Optional, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch
from types import MethodType
from packaging import version as pkg_version
from .mypp_module import PipelineModule, PipelineError
from .mypp_engine import myPipeEngine
from .copilot_zerooptimizer import CopilotZeroOptimizer


try:
    import apex
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    # Fail silently so we don't spam logs unnecessarily if user isn't using amp
    APEX_INSTALLED = False

TARGET_ID = -2
LOG_STAGE = -2
DATA_PARALLEL_ID = -2


def is_even(number):
    return number % 2 == 0


mem_alloced = 0
mem_cached = 0


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


class CopilotPipeEngine(myPipeEngine):

    # def _zero_llama_grad(self, freeze_list=None):
        # param_id = 0
        # for param in self.module.parameters():
        #     if param_id in self.para_dict and param.grad is not None:
        #         print(self.id2paramname[param_id])
        #         param.grad.data.mul_(0.0)
        #     param_id += 1


        # for group in self.optimizer.param_groups:
            # print(len(group['params']))
            # nl = name.split('.')[0]
            
            # if int(nl) >= 40 and param.grad is not None:
            #     print(name, nl)
            #     param.grad.data.mul_(0.0)
                # param.grad.data.zero_()

            # if name.find("mol_adapter") == -1 and param.grad is not None:
            #     print(name, nl)
            #     param.grad.data.mul_(0.0)


    def _configure_zero_optimizer(self, optimizer):
        zero_stage = self.zero_optimization_stage()
        model_dtype, grad_accum_dtype = self.get_data_types()
        timers = self.timers if self.wall_clock_breakdown() else None

        if optimizer is None:
            optimizer = DummyOptim(list(self.module.parameters()))

        if self.zero_legacy_stage1():
            raise Exception(
                "The deprecated version of ZeRO Stage 1 is not supported in deepspeed >= 0.5.9. Please downgrade to a version less than 0.5.9 if you need to use this deprecated version of ZeRO."
            )

        if zero_stage <= ZeroStageEnum.gradients:
            overlap_comm = self.zero_overlap_comm()
            contiguous_gradients = self.zero_contiguous_gradients()
            round_robin_gradients = self.zero_round_robin_gradients()
            assert not isinstance(optimizer, DummyOptim), "zero stage {} requires an optimizer".format(zero_stage)

            log_dist(f'Creating {model_dtype} ZeRO stage {zero_stage} optimizer',
                     ranks=[0])
            # Overlap and contiguous grads are meaningless in stage 1 and are ignored
            if zero_stage == ZeroStageEnum.optimizer_states:
                overlap_comm = False
                round_robin_gradients = False
                # Non-MoE requires contiguous grads to be disabled w. stage 1
                if not self.has_moe_layers:
                    contiguous_gradients = False

            if isinstance(self.module, PipelineModule):
                if overlap_comm:
                    logger.warning(
                        "Pipeline parallelism does not support overlapped communication, will be disabled."
                    )
                    overlap_comm = False
            optimizer = CopilotZeroOptimizer(
                optimizer,
                self.param_names,
                timers=timers,
                static_loss_scale=self.loss_scale(),
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=self.dynamic_loss_scale_args(),
                clip_grad=self.gradient_clipping(),
                contiguous_gradients=contiguous_gradients,
                reduce_bucket_size=self.zero_reduce_bucket_size(),
                allgather_bucket_size=self.zero_allgather_bucket_size(),
                dp_process_group=self.data_parallel_group,
                expert_parallel_group=self.expert_parallel_group
                if self.has_moe_layers else None,
                expert_data_parallel_group=self.expert_data_parallel_group
                if self.has_moe_layers else None,
                reduce_scatter=self.zero_reduce_scatter(),
                overlap_comm=overlap_comm,
                cpu_offload=self.zero_cpu_offload(),
                mpu=self.mpu,
                postscale_gradients=self.postscale_gradients(),
                gradient_predivide_factor=self.gradient_predivide_factor(),
                gradient_accumulation_steps=self.gradient_accumulation_steps(),
                ignore_unused_parameters=self.zero_ignore_unused_parameters(),
                partition_grads=zero_stage == ZeroStageEnum.gradients,
                round_robin_gradients=round_robin_gradients,
                has_moe_layers=self.has_moe_layers,
                fp16_master_weights_and_gradients=self.fp16_master_weights_and_gradients(
                ),
                communication_data_type=self.communication_data_type,
                elastic_checkpoint=self.zero_elastic_checkpoint())

        elif zero_stage == ZeroStageEnum.weights:
            assert not self.has_moe_layers, "MoE not supported with Stage 3"
            if isinstance(optimizer, DummyOptim):
                log_dist("Creating ZeRO Offload", ranks=[0])
                optimizer = DeepSpeedZeRoOffload(
                    self.module,
                    timers=timers,
                    ds_config=self.config,
                    overlap_comm=self.zero_overlap_comm(),
                    prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                    max_reuse_distance=self.zero_max_reuse_distance(),
                    max_live_parameters=self.zero_max_live_parameters(),
                    param_persistence_threshold=self.zero_param_persistence_threshold(),
                    model_persistence_threshold=self.zero_model_persistence_threshold(),
                    offload_param_config=self.zero_offload_param(),
                    mpu=self.mpu)
            else:
                log_dist(f'Creating {model_dtype} ZeRO stage {zero_stage} optimizer',
                         ranks=[0])
                from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
                optimizer = DeepSpeedZeroOptimizer_Stage3(
                    self.module,
                    optimizer,
                    timers=timers,
                    ds_config=self.config,
                    static_loss_scale=self.loss_scale(),
                    dynamic_loss_scale=self.dynamic_loss_scale(),
                    dynamic_loss_args=self.dynamic_loss_scale_args(),
                    clip_grad=self.gradient_clipping(),
                    contiguous_gradients=self.zero_contiguous_gradients(),
                    reduce_bucket_size=self.zero_reduce_bucket_size(),
                    prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                    max_reuse_distance=self.zero_max_reuse_distance(),
                    max_live_parameters=self.zero_max_live_parameters(),
                    param_persistence_threshold=self.zero_param_persistence_threshold(),
                    model_persistence_threshold=self.zero_model_persistence_threshold(),
                    dp_process_group=self.data_parallel_group,
                    reduce_scatter=self.zero_reduce_scatter(),
                    overlap_comm=self.zero_overlap_comm(),
                    offload_optimizer_config=self.zero_offload_optimizer(),
                    offload_param_config=self.zero_offload_param(),
                    sub_group_size=self.zero_sub_group_size(),
                    mpu=self.mpu,
                    postscale_gradients=self.postscale_gradients(),
                    gradient_predivide_factor=self.gradient_predivide_factor(),
                    gradient_accumulation_steps=self.gradient_accumulation_steps(),
                    aio_config=self.aio_config(),
                    communication_data_type=self.communication_data_type)

        else:
            raise NotImplementedError("ZeRO stage {} not implemented".format(zero_stage))

        return optimizer

    def _take_model_step(self, lr_kwargs, block_eigenvalue={}):
        if self.gradient_clipping() > 0.0:
            if not (self.fp16_enabled() or self.bfloat16_enabled() or self.amp_enabled()
                    or self.zero_optimization()):
                self.clip_fp32_gradients()
            elif self.amp_enabled():
                # AMP's recommended way of doing clipping
                # https://nvidia.github.io/apex/advanced.html#gradient-clipping
                master_params = amp.master_params(self.optimizer)
                clip_grad_norm_(parameters=master_params,
                                max_norm=self.gradient_clipping(),
                                mpu=self.mpu)
    
        # if self.copilot_train:
            # self._zero_llama_grad()

        self.optimizer.step()

        if hasattr(self.optimizer, '_global_grad_norm'):
            self._global_grad_norm = self.optimizer._global_grad_norm

        # Quantize the updated parameter if there is no overflow
        if self.quantizer:
            tensor_to_quantize = self.optimizer.bit16_groups if self.zero_optimization_stage(
            ) == 2 else self.optimizer.fp16_groups
            if self.compression_scheduler.weight_quantization_enabled:
                self.quantizer.quantize(
                    tensor_to_quantize,
                    (self.optimizer.overflow if self.fp16_enabled() else False),
                    self.eigenvalue_enabled(),
                    block_eigenvalue,
                )
        # zero grad in basic optimizer could be unreliable and may not exhibit
        # the behaviour that we want
        if self.bfloat16_enabled():
            # TODO: Temporary until bf16_optimizer and zero_optimizer are integrated
            if self.zero_optimization() and hasattr(self.optimizer, "zero_grad"):
                self.optimizer.zero_grad()
            else:
                pass
        elif self.zero_optimization() or self.fp16_enabled() or self.amp_enabled():
            self.optimizer.zero_grad()
        else:
            self.zero_grad()

        report_progress = self.global_rank == 0 if self.global_rank else True

        # Check overflow here since in DS fp16 optimizer, the overflow is updated in above step() function.
        overflow = False
        if hasattr(self.optimizer, "overflow"):
            overflow = self.optimizer.overflow
        self._step_applied = not overflow

        if overflow:
            self.skipped_steps += 1
        else:
            self.compression_scheduler.step()
            if self.lr_scheduler is not None:
                try:
                    self.lr_scheduler.step(**(lr_kwargs or {}))
                except TypeError:
                    # XXX Hack to work with Megatron 2.0 and DeepSpeed pipelines.
                    # We don't currently have a way to specify lr_kwargs from
                    # pipe_engine.train_batch()
                    self.lr_scheduler.step(increment=self.train_batch_size())

        if report_progress and (self.global_steps + 1) % self.steps_per_print() == 0:
            self._report_progress(self.global_steps + 1)

        self.global_steps += 1
        self.global_samples += self.train_batch_size()

    def _exec_optimizer_step(self, lr_kwargs=None):
        if self.wall_clock_breakdown():
            self.timers('step_microstep').start()
            self.timers('step').start()
        self.mem_status('BEFORE STEP', reset_max=True)

        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs)
        self._force_grad_boundary = False

        self.mem_status('AFTER STEP')

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/lr',
                                    self.get_lr()[0],
                                    self.global_samples)]
            if self.fp16_enabled() and hasattr(self.optimizer, 'cur_scale'):
                self.summary_events.append((f'Train/Samples/loss_scale',
                                            self.optimizer.cur_scale,
                                            self.global_samples))
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown():
            self.timers('step_microstep').stop()
            self.timers('step').stop()
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'batch_input',
                    'forward_microstep',
                    'backward_microstep',
                    'backward_inner_microstep',
                    'backward_allreduce_microstep',
                    'backward_tied_allreduce_microstep',
                    'step_microstep'
                ])
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'forward',
                    'backward',
                    'backward_inner',
                    'backward_allreduce',
                    'step'
                ])

    






# Export version information
from deepspeed.git_version_info import version, git_hash, git_branch

def _parse_version(version_str):
    '''Parse a version string and extract the major, minor, and patch versions.'''
    ver = pkg_version.parse(version_str)
    return ver.major, ver.minor, ver.micro

__version__ = version
__version_major__, __version_minor__, __version_patch__ = _parse_version(__version__)
__git_hash__ = git_hash
__git_branch__ = git_branch

def initialize(args=None,
               model: torch.nn.Module = None,
               optimizer: Optional[Union[Optimizer,
                                         DeepSpeedOptimizerCallable]] = None,
               model_parameters: Optional[torch.nn.Module] = None,
               training_data: Optional[torch.utils.data.Dataset] = None,
               lr_scheduler: Optional[Union[_LRScheduler,
                                            DeepSpeedSchedulerCallable]] = None,
               mpu=None,
               dist_init_required: Optional[bool] = None,
               collate_fn=None,
               model_ckpt_list=None,
               copilot_train=False,
               config=None,
               config_params=None):
    """Initialize the DeepSpeed Engine.

    Arguments:
        args: an object containing local_rank and deepspeed_config fields.
            This is optional if `config` is passed.

        model: Required: nn.module class before apply any wrappers

        optimizer: Optional: a user defined Optimizer or Callable that returns an Optimizer object.
            This overrides any optimizer definition in the DeepSpeed json config.

        model_parameters: Optional: An iterable of torch.Tensors or dicts.
            Specifies what Tensors should be optimized.

        training_data: Optional: Dataset of type torch.utils.data.Dataset

        lr_scheduler: Optional: Learning Rate Scheduler Object or a Callable that takes an Optimizer and returns a Scheduler object.
            The scheduler object should define a get_lr(), step(), state_dict(), and load_state_dict() methods

        mpu: Optional: A model parallelism unit object that implements
            get_{model,data}_parallel_{rank,group,world_size}()

        dist_init_required: Optional: None will auto-initialize torch distributed if needed,
            otherwise the user can force it to be initialized or not via boolean.

        collate_fn: Optional: Merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.

        config: Optional: Instead of requiring args.deepspeed_config you can pass your deepspeed config
            as an argument instead, as a path or a dictionary.

        config_params: Optional: Same as `config`, kept for backwards compatibility.

    Returns:
        A tuple of ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``

        * ``engine``: DeepSpeed runtime engine which wraps the client model for distributed training.

        * ``optimizer``: Wrapped optimizer if a user defined ``optimizer`` is supplied, or if
          optimizer is specified in json config else ``None``.

        * ``training_dataloader``: DeepSpeed dataloader if ``training_data`` was supplied,
          otherwise ``None``.

        * ``lr_scheduler``: Wrapped lr scheduler if user ``lr_scheduler`` is passed, or
          if ``lr_scheduler`` specified in JSON configuration. Otherwise ``None``.
    """
    log_dist("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(__version__, __git_hash__,
                                                                             __git_branch__),
             ranks=[0])

    # Disable zero.Init context if it's currently enabled
    zero.partition_parameters.shutdown_init_context()

    assert model is not None, "deepspeed.initialize requires a model"

    global dist
    from deepspeed import comm as dist
    dist_backend = get_accelerator().communication_backend_name()
    dist.init_distributed(dist_backend=dist_backend, dist_init_required=dist_init_required)

    # Set config using config_params for backwards compat
    if config is None and config_params is not None:
        config = config_params

    # Check for deepscale_config for backwards compat
    if hasattr(args, "deepscale_config") and args.deepscale_config is not None:
        logger.warning("************ --deepscale_config is deprecated, please use --deepspeed_config ************")
        if hasattr(args, "deepspeed_config"):
            assert (args.deepspeed_config is
                    None), "Not sure how to proceed, we were given both a deepscale_config and deepspeed_config"
        args.deepspeed_config = args.deepscale_config
        args.deepscale_config = None

    # Check that we have only one config passed
    if hasattr(args, "deepspeed_config") and args.deepspeed_config is not None:
        assert config is None, "Not sure how to proceed, we were given deepspeed configs in the deepspeed arguments and deepspeed.initialize() function call"
        config = args.deepspeed_config
    assert config != None, "DeepSpeed requires --deepspeed_config to specify configuration file"

    assert model is not None, "deepspeed.initialize requires a model"
    assert mpu is None, "mpu must be None with pipeline parallelism"
    mpu = model.mpu()
    
    config_class = DeepSpeedConfig(config, mpu)
    engine = CopilotPipeEngine(args=args,
                                model=model,
                                optimizer=optimizer,
                                model_parameters=model_parameters,
                                training_data=training_data,
                                lr_scheduler=lr_scheduler,
                                mpu=model.mpu(),
                                dist_init_required=dist_init_required,
                                collate_fn=collate_fn,
                                config=config,
                                config_class=config_class,
                                model_ckpt_list=model_ckpt_list,
                                copilot_train=copilot_train,
                                )

    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler
    ]
    return tuple(return_items)