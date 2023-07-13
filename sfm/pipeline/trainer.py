# -*- coding: utf-8 -*-
from collections import defaultdict
import json
import time
from typing import Dict, Optional, Callable, Sized, Union
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, dataloader
from abc import ABC, abstractmethod
import numpy as np
import random
from dataclasses import dataclass, field, asdict
from enum import Enum

from tqdm import tqdm
from sfm.pipeline.accelerator import Accelerator, SingleNodeAccelerator, DdpAccelerator, DeepSpeedAccelerator



class TraingStrategy(Enum):
    Single = 0
    Zero1 = 1
    Zero2 = 2
    Zero3 = 3
    DDP = 4

@dataclass
class TrainerConfig:
    # common
    strategy: TraingStrategy = TraingStrategy.Zero1
    epochs: int = 1
    seed: int = 46
    init_loss_scale: float = 1.0
    batch_size: int = 32
    save_dir: str = "./checkpoints"
    resume_checkpoint: str = "checkpoint_last.pt"
    save_batch_interval: int = 0
    save_epoch_interval: int = 1
    log_interval: int = 100
    strategy: TraingStrategy = TraingStrategy.Single

@dataclass
class TrainerState:
    args: TrainerConfig
    global_step: int = 0
    epoch: int = 0
    batch: int = 0
    loss_scale: float = 0

@dataclass
class ModelOutput:
    loss: torch.Tensor
    log_output: Dict


@dataclass
class LogOutput:
    loss: float
    loss_scale: float
    lr: float
    epoch: int
    batch: int
    global_step: int
    time: float
    extra_output: Dict
    
    def __str__(self) -> str:
        return json.dumps(asdict(self))


class Model(nn.Module, ABC):    
    @abstractmethod
    def compute_loss(self, pred, batch) -> ModelOutput:
        pass
    
    def load_pretrained(self):
        """
        Load all pretrained weights, e.g., pretrained encoders or decoders.
        """
        pass
    
    @abstractmethod
    def config_optimizer(self) -> tuple[Optimizer, LRScheduler]:
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            tuple[Optimizer, LRScheduler]: _description_
        """
        pass

def seed_everything(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class Trainer(object):
    def __init__(
        self,
        args: TrainerConfig,
        /,
        model: Model,
        train_data: Dataset,
        collater: Callable,
        valid_data: Optional[Dataset] = None,
        test_data: Optional[Dataset] = None,
    ):
        super().__init__()
        self.args = args

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        
        self.optimizer, self.lr_scheduler = model.config_optimizer()
        
        self.train_data_loader = self.build_data_loader(train_data, shuffle=True, collater=collater)
        self.valid_data_loader = self.build_data_loader(valid_data, shuffle=False, collater=collater)

        self.state = TrainerState(args=args)
        self.accelerator = self.build_accelerator()
        
        self.save_dir = Path(self.args.save_dir)
        self.model.load_pretrained()
    
    def save_checkpoint(self, name: str):
        self.accelerator.save_checkpoint(name)
        
            
    def load_checkpoint(self, ckpt_id: int|str):
        self.state = self.accelerator.load_checkpoint(ckpt_id)
    
    def resume(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_list_path = self.save_dir / 'checkpoint_list.txt'
        if checkpoint_list_path.exists():
            with open(checkpoint_list_path, 'r') as f:
                checkpoint_list = f.read().splitlines()
            if len(checkpoint_list) > 0:
                checkpoint_last = checkpoint_list[-1]
                checkpoint_path = self.save_dir / checkpoint_last
                if checkpoint_path.exists():
                    print("Resume from checkpoint: ", checkpoint_path)
                    self.load_checkpoint(checkpoint_last)
                else:
                    print("Not resume from checkpoint")
        else:
            with open(checkpoint_list_path, 'w') as f:
                f.write('')
    
    def build_accelerator(self) -> Accelerator:
        if self.args.strategy == TraingStrategy.Single:
            return SingleNodeAccelerator(
                self.args,
                self.model,
                self.optimizer,
                self.lr_scheduler,
            )
        elif self.args.strategy in [TraingStrategy.Zero1, TraingStrategy.Zero2, TraingStrategy.Zero3]:
            return DeepSpeedAccelerator(
                self.args,
                self.model,
                self.optimizer,
                self.lr_scheduler
            )
        elif self.args.strategy == TraingStrategy.DDP:
            return DdpAccelerator(
                self.args,
                self.model,
                self.optimizer,
                self.lr_scheduler,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.args.strategy}")
    
    def build_data_loader(self, data: Optional[Dataset], shuffle: bool, collater: dataloader._collate_fn_t) -> Optional[DataLoader]:
        if data is None:
            return None
        
        if self.is_distibuated:
            sampler = DistributedSampler(data, num_replicas=self.args.world_size, shuffle=shuffle)
        else:
            sampler = RandomSampler(data) if shuffle else None
        
        return DataLoader(
            data,
            sampler=sampler,
            batch_size=self.args.batch_size,
            collate_fn=collater,
            drop_last=True
        )
    
    def build_log_output(self, model_output: ModelOutput) -> LogOutput:
        return LogOutput(
            loss=model_output.loss.item(),
            loss_scale=self.state.loss_scale,
            lr=self.lr_scheduler.get_last_lr()[0],
            epoch=self.state.epoch,
            batch=self.state.batch,
            global_step=self.state.global_step,
            time=time.time(),
            extra_output=model_output.log_output
        )

    def train(self):
        print("start training")
        print(self.model)
        seed_everything(self.args.seed)
        
        assert self.train_data_loader is not None
        
        self.resume()
        
        # TODO: add more logging
        for epoch in range(self.args.epochs):
            print("epoch: ", epoch)
            self.state.epoch = epoch

            for i, batch_data in tqdm(enumerate(self.train_data_loader)):
                self.state.batch = i
                # TODO: change to accelerator
                pred = self.model(batch_data)
                model_output = self.model.compute_loss(pred, batch_data)
                loss = model_output.loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                
                self.state.global_step += 1
                
                self.state.loss_scale = 1.0 # todo: FP16 support
                
                if self.args.save_batch_interval > 0 and self.state.global_step % self.args.save_batch_interval == 0:
                    checkpoint_name = f'checkpoint_E{epoch}_B{i}.pt'
                    self.save_checkpoint(checkpoint_name)
                
                if self.args.log_interval > 0 and self.state.global_step % self.args.log_interval == 0:
                    log_output = self.build_log_output(model_output)
                    print(log_output)
            
            if self.args.save_epoch_interval > 0 and epoch % self.args.save_epoch_interval == 0:
                checkpoint_name = f'checkpoint_E{epoch}.pt'
                self.save_checkpoint(checkpoint_name)
        
        print('Finished Training')

    def validate(self):
        if self.valid_data_loader is None:
            print("No validation data, skip validation")
            return
        
        print("start validation")

        #TODO: support multiple losses
        for i, batch_data in tqdm(enumerate(self.valid_data_loader)):
            loss = self.model(batch_data)
            print("loss: ", loss) 
