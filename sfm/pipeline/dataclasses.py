import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict

import torch


class TraingStrategy(Enum):
    Single = 0
    Zero1 = 1
    Zero2 = 2
    Zero3 = 3
    DDP = 4

@dataclass
class TrainerConfig:
    epochs: int = 1
    seed: int = 46
    fp16: bool = False
    grad_scaler_init: float = 1.0
    update_freq: int = 1
    batch_size: int = 32
    save_dir: str = "./checkpoints"
    save_batch_interval: int = 0
    save_epoch_interval: int = 1
    log_interval: int = 100
    strategy: TraingStrategy = TraingStrategy.Single
    cpu: bool = False

@dataclass
class TrainerState:
    args: TrainerConfig
    global_step: int = 0
    epoch: int = 0
    batch: int = 0

@dataclass
class ModelOutput:
    loss: torch.Tensor
    log_output: Dict


@dataclass
class LogOutput:
    loss: float
    grad_scale: float
    lr: float
    epoch: int
    batch: int
    global_step: int
    time: str
    extra_output: Dict
    
    def __str__(self) -> str:
        return json.dumps(asdict(self))
