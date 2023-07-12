# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional

import deepspeed
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
import numpy as np
import random
from dataclasses import dataclass
from enum import Enum

from tqdm import tqdm


class TraingStrategy(Enum):
    Zero1 = 1
    Zero2 = 2
    Zero3 = 3
    DDP = 4

@dataclass
class TrainerConfig:
    strategy: TraingStrategy = TraingStrategy.Zero1
    epochs: int = 1
    seed: int = 46
    loss_scale: float = 1.0
    batch_size: int = 32
    
    ## Distibuated training
    global_rank: int = 0
    local_rank: int = 0
    node_rank: int = 0
    world_size: int = 1
    num_nodes: int = 1


class Model(nn.Module, ABC):
    @abstractmethod
    def compute_loss(self, pred, batch) -> torch.Tensor:
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
        model: Model,
        train_data: Dataset,
        valid_data: Optional[Dataset] = None,
        test_data: Optional[Dataset] = None,
    ):
        super().__init__()
        self.args = args

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        
        self.optimizer = self.build_optimizer(self.model.parameters())
        self.lr_scheduler = self.build_lr_scheduler(self.optimizer)
        
        self.train_data_loader = self.build_data_loader(train_data, shuffle=True, collater=train_data.collater)
        self.valid_data_loader = self.build_data_loader(valid_data, shuffle=False, collater=valid_data.collater)
    
    def build_optimizer(self, parameters) -> Optimizer:
        return torch.optim.Adam(parameters, lr=1e-3) # todo: add more options
    
    def build_lr_scheduler(self, optimizer) -> LRScheduler:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # todo: add more options
        
    def load_checkpoint(self, checkpoint_path=None):
        pass
    
    def build_data_loader(self, data: Optional[Dataset], shuffle: bool, collater) -> Optional[DataLoader]:
        if data is None:
            return None
        
        if self.args.world_size == 1:
            sampler = torch.utils.data.RandomSampler(data) if shuffle else None
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                data, num_replicas=self.args.world_size, shuffle=shuffle
            )
        
        return DataLoader(
            data,
            sampler=sampler,
            batch_size=self.args.batch_size,
            collate_fn=collater,
            drop_last=True
        )  

    def train(self):
        print("start training")
        seed_everything(self.args.seed)
        
        assert self.train_data_loader is not None
        
        # TODO: add more logging
        for epoch in range(self.args.epochs):
            print("epoch: ", epoch)
            for i, batch_data in tqdm(enumerate(self.train_data_loader)):
                # TODO: change to accelerator
                pred = self.model(batch_data)
                loss = self.model.compute_loss(pred, batch_data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

    def validate(self):
        if self.valid_data_loader is None:
            print("No validation data, skip validation")
            return
        
        print("start validation")

        #TODO: support multiple losses
        for i, batch_data in tqdm(enumerate(self.valid_data_loader)):
            loss = self.model(batch_data)
            print("loss: ", loss)
        
    
    
