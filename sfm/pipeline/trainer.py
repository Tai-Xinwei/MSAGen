# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional

import deepspeed
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset

import numpy as np
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class TraingStrategy(Enum):
    Zero1 = 1
    Zero2 = 2
    Zero3 = 3
    DDP = 4

@dataclass
class TrainerConfig:
    strategy: TraingStrategy = TraingStrategy.Zero1
    epochs: int = 100
    seed: int = 46
    loss_scale: float = 1.0

class EntityType(Enum):
    Text = 0
    Mol = 1
    Protein = 2

@dataclass
class DataBatch:
    # Batch size * sequence length * hidden size
    x: Optional[torch.Tensor] = None
    
    # the entity mask dict form entity type (e.g., moleclue) to a list of masks
    # Each encoder can only process the entities they care about
    mask: Optional[Dict[EntityType, List[torch.Tensor]]] = None
    
    # the label of the batch, Batch size * sequence length
    label: Optional[torch.Tensor] = None


class Trainer(ABC):
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def build_model(self) -> nn.Module:
        pass
    
    @abstractmethod
    def save_checkpoint(self):
        pass
    
    @abstractmethod
    def build_optimizer(self, parameters: Iterator[nn.Parameter]) -> Optimizer:
        pass
    
    @abstractmethod
    def build_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        pass
    
    @abstractmethod
    def build_criterion(self):
        pass
    
    @abstractmethod
    def load_checkpoint(self):
        pass  

    @abstractmethod
    def build_trainig_data_loader(self):
        pass
    
    @abstractmethod
    def build_validation_data_loader(self):
        pass
    

class BaseTrainer(Trainer):
    def __init__(
        self,
        args: TrainerConfig
    ):
        super().__init__()
        self.args = args

        self.net = self.build_model()
        self.creterion = self.build_criterion()
        self.optimizer = self.build_optimizer(self.net.parameters())
        self.scheduler = self.build_scheduler(self.optimizer)
        
        
        self.train_data_loader = self.build_trainig_data_loader()
        self.valid_data_loader = self.build_validation_data_loader()
        
        train_data = self.train_data_loader.load_data()
        
        self.model_engine, _, self.train_loader, _ = deepspeed.initialize(
            args=args,
            model=self.net,
            model_parameters=self.net,
            training_data=train_data,
            collate_fn=train_data.collater2,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler
        )
        
        # define model
        # net = GraphormerModel(args)
        # count_paranum(net)

        # define criterion
        # self.L1loss = L1_criterions(args, reduction='mean', data_mean=data_mean, data_std=data_std)

        # define optimizer
        # parameters = filter(lambda p: p.requires_grad, net.parameters())
        # self.optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

        # define scheduler
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=0.0)

        # define model engine
        # self.model_engine, _, self.train_loader, _ = deepspeed.initialize(args=args,
        #                                                                   model=net,
        #                                                                   model_parameters=parameters,
        #                                                                   training_data=train_data,
        #                                                                   collate_fn=train_data.collater2,
        #                                                                   optimizer=self.optimizer,
        #                                                                   lr_scheduler=self.scheduler,
        #                                                                   loss_scale=args.loss_scale,
        #                                                                   loss_fn=self.L1loss,
        #                                                                   )

        # define dataloader
        # len_val, self.val_dataloader = self.set_dataloader(data=val_data)
        # len_test, self.test_dataloader = self.set_dataloader(data=test_data)

        # load checkpoints
        # self.load_checkpoint()
        
    def seed_everything(self, seed):
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        
    def load_checkpoint(self, checkpoint_path=None):
        pass

    def set_dataloader(self, data=None):
        len_data = 0
        if data is not None:
            len_data = len(data)
            validsampler = torch.utils.data.distributed.DistributedSampler(
                data, num_replicas=self.model_engine.dp_world_size, shuffle=True
            )
            dataloader = torch.utils.data.DataLoader(
                data,
                sampler=validsampler,
                batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                collate_fn=data.collater,
            )

        return len_data, dataloader

    def train(self):
        print("start training")
        self.seed_everything(self.args.seed)
        
        # TODO: add more logging
        for epoch in range(self.args.epochs):
            for i, batch_data in enumerate(self.train_data_loader):
                model_output = self.model_engine.train_batch(batch_data)
                loss = self.creterion(model_output, batch_data.label)
                self.model_engine.backward(loss)
                self.model_engine.step()
                
                
                
                
            
