# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch


class TrainStrategy(str, Enum):
    """
    TrainStrategy provides a convenient way to choose between different training strategies supported by the A4SFramework.

    Args:
        Single: The training is performed on a single device without any parallelism or distribution.

        DDP: Distributed Data Parallel (DDP) training, where the model is replicated across multiple devices and gradients are synchronized.

        Zero0: DeepSpeed Training without using ZeRO (Zero Redundancy Optimizer) parallelism strategy.

        Zero1: The first level of the ZeRO parallelism strategy. The optimizer states are partitioned

        Zero2: The second level of the ZeRO parallelism strategy. Zero1 + reduced gradients for updating the model weights are partitioned.

        Zero3: The third level of the ZeRO parallelism strategy. Zero1 + Zero2 + model parameters are partitioned.

        ZeroInf: Extension to the third level of the ZeRO parallelism strategy by enabling NVMe offloading.

        Pipeline: Pipeline parallelism, where the model is divided into stages and processed on
        different devices in a pipeline fashion.

        ThreeD: 3D parallelism, which combines data, model, and pipeline parallelism for large-scale training.
    """

    DDP = "DDP"
    Zero0 = "Zero0"
    Zero1 = "Zero1"
    Zero2 = "Zero2"
    Zero3 = "Zero3"
    ZeroInf = "ZeroInf"
    Single = "Single"
    Pipeline = "Pipeline"
    ThreeD = "ThreeD"
    NNScaler = "NNScaler"


# torch.serialization.add_safe_globals([TrainStrategy])


@dataclass
class TrainerConfig:
    """
    This TrainerConfig class makes it easier to manage and modify training-related parameters.

    Args:
        seed: Seed for random number generation, ensuring reproducible results.

        fp16, auto_cast, bf16: Flags for mixed precision training and related settings.

        grad_scaler_init: Initial scaling factor for gradient scaling in mixed precision training.

        gradient_accumulation_steps: Number of steps for accumulating gradients before updating weights.

        max_tokens, train_batch_size, val_batch_size: Batch-related parameters for training and validation.

        val_batch_interval, val_batch_log_interval, val_batch_log_all_metric, val_epoch_interval: Settings for validation intervals.

        save_dir, save_batch_interval, save_epoch_interval: Settings for saving checkpoints and their intervals.

        log_interval: Interval for logging training progress.

        strategy: A TrainStrategy enumeration indicating the selected training strategy.

        pp_partition_layer_name, pp_part_list: Settings related to pipeline parallelism.

        cpu: Flag to indicate whether to use CPU for training.

        ifresume, load_ckpt: Flags for resuming training from a checkpoint.

        unfreeze_param_list, finetune_from_checkpoint_dir, finetune_from_checkpoint_id: Settings for fine-tuning from a checkpoint.

        daliLoader, dynamic_loader: Flags for using DALI and dynamic data loaders.

        gradient_clipping: Gradient clipping value.

        total_num_steps, warmup_num_steps: Settings for the total number of training steps and warm-up steps.

        warmup_factor, warmup_lr, warmup_num_epochs: Settings for the warm-up phase of training.
        max_lr, init_lr, min_lr, weight_decay: Settings related to learning rates and weight decay.

        total_num_epochs: The total number of training epochs.

        wandb, wandb_team, wandb_group, wandb_project: Settings for Weights & Biases integration.

        beta1, beta2, eps: Hyperparameters for the optimizer.

    """

    # training parameters
    seed: int = 46
    total_num_steps: int = 1000
    total_num_epochs: int = 100
    max_tokens: int = 2048
    train_batch_size: int = 1
    val_batch_size: int = 1
    val_batch_interval: int = 0
    val_batch_log_interval: int = 1000
    val_batch_log_all_metric: bool = False
    val_epoch_interval: int = 1
    freeze_param_list: str = ""
    unfreeze_param_list: str = ""
    reset_act_each_step: bool = False
    use_unified_batch_sampler: bool = False
    activation_checkpoint_interval: int = 0
    checkpointable_layers: Optional[list[str]] = None
    gradient_clipping: float = 1.0

    # parallelisation parameters
    strategy: TrainStrategy = TrainStrategy.Single
    pp_partition_layer_name: str = ""
    pp_part_list: Optional[List[int]] = None
    cpu: bool = False

    # IO parameters
    save_dir: str = "./checkpoints"
    save_batch_interval: int = 0
    save_epoch_interval: int = 1
    log_interval: int = 10000
    finetune_from_checkpoint_dir: Optional[str] = None
    finetune_from_checkpoint_id: Optional[str] = None
    ifresume: bool = False
    load_ckpt: bool = False

    # optimizer hyperparameters
    max_lr: float = 0.0001
    init_lr: float = 8e-5
    min_lr: float = 8e-6
    weight_decay: float = 0.0
    beta1: float = 0.9  # Adam
    beta2: float = 0.999  # Adam
    eps: float = 1e-8  # Adam

    # lr scheduler hyperparameters
    warmup_num_steps: int = 60
    warmup_factor: float = 0.06
    warmup_lr: float = 1e-6
    warmup_num_epochs: int = 10

    # performance parameters
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    auto_cast: bool = False
    grad_scaler_init: float = 1.0
    bf16: bool = False
    fp8: bool = False
    mm_tensorcore: str = "fp32"  # reduce matrix multiplication precision
    # (bf16, tf32, or fp32)
    compile: bool = False  # compile CUDA kernels with torch.compile
    zero_offload: bool = (
        False  # offload parameters to CPU/NVMe if Zero optimizer is used
    )
    zero_offload_dir: str = "./"
    find_unused_parameters: bool = True

    # dataloader strategy
    dynamic_loader: bool = False
    ifstack: bool = False
    use_dali_pipeline: bool = False
    unified_data_num_workers: int = 0

    # wandb
    wandb: bool = False
    wandb_team: str = ""
    wandb_group: str = ""
    wandb_project: str = ""
    wandb_run_name: str = ""

    # early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_metric: str = "valid_loss"
    early_stopping_mode: str = "min"

    # validate
    calculate_metrics: bool = False

    # profiler
    profiling: bool = False
    prof_dir: str = "./prof"
    ptensorboard: bool = False
    allreduce_log_path: str = "/tmp/stragglers"

    # debugger
    debug: bool = False

    # AutoGradForce
    AutoGradForce: bool = False

    def __str__(self):
        return (
            "Config[\n"
            + "\n".join([f"  {k}: {v}" for k, v in asdict(self).items()])
            + "\n]"
        )


@dataclass
class DistributedTrainConfig(TrainerConfig):
    """
    DistributedTrainConfig is a combination of TrainerConfig and DistributedConfig.

    Args:
        local_rank (int, default: -1): The local rank of the process within a node.

        world_size (int, default: 1): The total number of processes involved in the distributed training.

        node_rank (int, default: 0): The rank of the current node within the cluster.

        rank (int, default: 0): The global rank of the process.

        pipeline_model_parallel_size (int, default: 0): The size of the pipeline model parallel group.

        tensor_model_parallel_size (int, default: 1): The size of the tensor model parallel group.

        deepspeed_config (str, default: ''): The path to the DeepSpeed configuration file.

        dist_backend (str, default: 'nccl'): The distributed backend to use for communication among processes.
    """

    local_rank: int = -1
    world_size: int = 1
    node_rank: int = 0
    rank: int = 0
    pipeline_model_parallel_size: int = 0
    tensor_model_parallel_size: int = 1
    deepspeed_config_path: str = ""
    deepspeed_config: Any = None
    dist_backend: str = "nccl"


@dataclass
class NNScalerTrainConfig(DistributedTrainConfig):
    """
    NNScalerTrainConfig is a combination of DistributedTrainConfig and NNScaler needed config.
    """

    plan_ngpus: int = -1
    runtime_ngpus: int = -1

    # whether to use dynamic shape to generate code
    dynamic_shape: bool = True

    use_zero: bool = False
    zero_ngroups: int = 1

    # whether the generated code is for inference only
    inference_only: bool = False

    # end2end means,
    #  1. the first argument of `module.forward` must be the data sample
    #  2. the first return value of `module.forward` must be the loss
    #  which must be a scalar tensor
    use_end2end: bool = False

    # current only end2end module supports in pipeline mode.
    # so be sure to set use_end2end=True when use_pipeline=True
    use_pipeline: bool = False
    # number of micro-batches
    pipeline_nmicros: int = -1
    # number of stages
    pipeline_nstages: int = -1
    # it is pas's responsibility to apply the scheduler
    pipeline_scheduler: str = "1f1b"

    # 1. 'match': Reuse if match, error if not match, generate if no previous gerenated code exists.
    # 2. 'override': Nothing will be reused. Everything will be regenerated.
    # 3. 'moo': MOO is short for 'match or override'. It will reuse if match, generate if not match or no previous generated code exists.
    # 4. 'graph': Reuse graph only if match, generate otherwise.
    file_reuse: str = "match"

    # the folder to save gencode and model init state dict
    cache_savedir: str = "./.cube"

    # please view the doc of init_module_params in MagicCube/docs/source/parallel_module.md
    init_module_params: bool = False

    # please view the doc of BroadcastGenFilesStrategy in MagicCube/docs/source/parallel_module.md
    broadcast_strategy: str = "all"


@dataclass
class TrainerState:
    """
    The TrainerState class helps manage various training-related attributes, making it easier to monitor and control the progress of the training.

    Args:
        args: A TrainerConfig object that stores the configuration settings for the training process.

        global_step (int, default: 0): The current global step of the training process, which is a count of the number of gradient updates performed.

        epoch (int, default: 0): The current epoch of the training process, representing the number of times the entire dataset has been processed.

        batch (int, default: 0): The current batch number within the current epoch.
    """

    args: TrainerConfig
    global_step: int = 0
    epoch: int = 0
    batch: int = 0
    sample: int = 0
    total_acc_sample: int = 0


@dataclass
class ModelOutput:
    loss: torch.Tensor
    num_examples: Optional[int] = None
    log_output: Optional[Dict] = None
    logits: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None


def format_extra_output(raw_extra_output):
    if raw_extra_output is None:
        return ""

    extra_output = []
    for k, v in raw_extra_output.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                v = v.item()
                extra_output.append(f"{k}: {v:.4g}")
            else:
                v = v.detach().cpu().numpy()
                extra_output.append(f"{k}: {v}")
        elif isinstance(v, float):
            extra_output.append(f"{k}: {v:.4g}")
        else:
            extra_output.append(f"{k}: {v}")
    extra_output = " | ".join(extra_output)
    return extra_output


@dataclass
class TrainLogOutput:
    loss: float
    grad_scale: float
    lr: float
    epoch: int
    batch: int
    global_step: int
    total_samples: int
    extra_output: Dict

    def __str__(self) -> str:
        extra_output = format_extra_output(self.extra_output)
        return (
            f"Step: {self.global_step} (Epoch {self.epoch} Iter {self.batch+1}) | Loss: {self.loss:.4g} | LR: {self.lr:.4g} | Grad Scale: {self.grad_scale:.4g} | "
            + extra_output
        )


@dataclass
class ValidLogOutput:
    valid_loss: float
    epoch: int
    num_examples: Optional[int] = None
    extra_output: Optional[Dict] = None
    logits: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None

    def __str__(self):
        extra_output = format_extra_output(self.extra_output)
        return (
            f"Valid Loss: {self.valid_loss:.4g} | Num Examples: {self.num_examples} | "
            + extra_output
        )
