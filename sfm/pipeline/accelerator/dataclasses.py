# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Optional, Union

import torch


class TrainStrategy(str, Enum):
    DDP = "DDP"
    Zero1 = "Zero1"
    Zero2 = "Zero2"
    Zero3 = "Zero3"
    Single = "Single"
    Pipeline = "Pipeline"


# Will be removed, use DistributedTrainConfig instead
@dataclass
class DistributedConfig:
    local_rank: int = -1
    world_size: int = 1
    node_rank: int = 0
    rank: int = 0
    pipeline_parallelism: int = 0
    tensor_parallelism: int = 1
    deepspeed_config: str = ""
    dist_backend: str = "nccl"


@dataclass
class TrainerConfig:
    epochs: int = 1
    seed: int = 46
    fp16: bool = False
    auto_cast: bool = False
    bf16: bool = False
    grad_scaler_init: float = 1.0
    gradient_accumulation_steps: int = 1
    train_batch_size: int = 1
    val_batch_size: int = 1
    val_batch_interval: int = 0
    val_batch_log_interval: int = 1000
    val_epoch_interval: int = 1
    save_dir: str = "./checkpoints"
    save_batch_interval: int = 0
    save_epoch_interval: int = 1
    log_interval: int = 10000
    strategy: TrainStrategy = TrainStrategy.Single
    cpu: bool = False
    ifresume: bool = False

    gradient_clipping: float = 1.0
    total_num_steps: int = 1000
    warmup_num_steps: int = 60
    warmup_factor: float = 0.06
    warmup_lr: float = 1e-6
    warmup_num_epochs: int = 10
    max_lr: float = 0.0001
    init_lr: float = 8e-5
    min_lr: float = 8e-6
    weight_decay: float = 0.0
    total_num_epochs: int = 100

    # adam
    beta1: float = 0.9
    beta2: float = 0.999

    def __str__(self):
        return (
            "Config[\n"
            + "\n".join([f"  {k}: {v}" for k, v in asdict(self).items()])
            + "\n]"
        )


@dataclass
class DistributedTrainConfig(TrainerConfig):
    local_rank: int = -1
    world_size: int = 1
    node_rank: int = 0
    rank: int = 0
    pipeline_parallelism: int = 0
    tensor_parallelism: int = 1
    deepspeed_config: str = ""
    dist_backend: str = "nccl"


@dataclass
class TrainerState:
    args: TrainerConfig
    global_step: int = 0
    epoch: int = 0
    batch: int = 0


@dataclass
class ModelOutput:
    loss: torch.Tensor
    num_examples: Optional[int] = None
    log_output: Optional[Dict] = None


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
    num_examples: Optional[int] = None
    extra_output: Optional[Dict] = None

    def __str__(self):
        extra_output = format_extra_output(self.extra_output)
        return (
            f"Valid Loss: {self.valid_loss:.4g} | Num Examples: {self.num_examples} | "
            + extra_output
        )
