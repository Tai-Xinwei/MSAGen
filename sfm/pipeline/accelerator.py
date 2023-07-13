from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union
from sfm.pipeline.trainer import Trainer, TrainerConfig, Model, ModelOutput, TrainerState
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel

import torch
import os

try:
    import deepspeed
except ImportError:
    deepspeed = None

class Accelerator(ABC):
    args: TrainerConfig
    model: Union[Model, DistributedDataParallel]
    optimizer: Optimizer
    lr_scheduler: LRScheduler
    
    @abstractmethod
    def set_up():
        pass
    
    @abstractmethod
    def train_step(self, batch_data) -> ModelOutput:
        pass
    
    @abstractmethod
    def save_checkpoint(self, ckpt_id: int|str, extra_state: Optional[dict] = None):
        pass
    
    @abstractmethod
    def load_checkpoint(self, ckpt_id: int|str) -> TrainerState:
        pass
    
    @abstractmethod
    def should_log() -> bool:
        pass
    

class SingleNodeAccelerator(Accelerator):
    device: Union[int, str, torch.device]
    
    def __init__(self, args, model, optimizer, lr_scheduler) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def set_up(self):
        pass
    
    def train_step(self, batch_data):
        self.model.train()
        self.model.to(self.device)
        
        pred = self.model(batch_data)
        model_output = self.model.compute_loss(pred, batch_data)
        loss = model_output.loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        
        return model_output
    
    def save_checkpoint(self, ckpt_id: str, extra_state: Optional[dict] = None):
        save_dir = Path(self.args.save_dir)
        
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }
        
        if extra_state is not None:
            checkpoint.update(extra_state)
        print('save checkpoint: ', ckpt_id)
        torch.save(checkpoint, save_dir / chpt_id)
        
        with open(save_dir / 'checkpoint_list.txt', 'a') as f:
            f.write(ckpt_id + '\n')
    

    def load_checkpoint(self, ckpt_id: str) -> TrainerState:
        save_dir = Path(self.args.save_dir)
        checkpoint_path = save_dir / ckpt_id
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        state = TrainerState(args=self.args)
        for k, v in checkpoint.items():
            if k not in ['model', 'optimizer', 'lr_scheduler']:
                setattr(state, k, v)
        return state

    def should_log(self) -> bool:
        return True

class DdpAccelerator(SingleNodeAccelerator):
    dist_backend: str = "nccl"
    
    def __init__(self, args, model, optimizer, lr_scheduler) -> None:
        super().__init__(args, model, optimizer, lr_scheduler)
    
    def set_up(self):
        assert "WORLD_SIZE" in os.environ, "WORLD_SIZE must be set to use DDP"
        assert "RANK" in os.environ, "RANK must be set to use DDP"
        assert "LOCAL_RANK" in os.environ, "LOCAL_RANK must be set to use DDP"
        
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
            
        print(f'Initializing DDP by env: word size: {self.world_size}, rank: {self.rank}, local_rank{self.local_rank} ', flush=True)
        
        torch.distributed.init_process_group(
            backend=self.dist_backend, init_method="env://", world_size=self.world_size, rank=self.rank

        )
        
        torch.distributed.barrier()

        print('DDP initialized.', flush=True)


        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
        )

    def save_checkpoint(self, ckpt_id: str, extra_state: dict | None = None):
        if self.rank == 0:
            super().save_checkpoint(ckpt_id, extra_state)

    def should_log(self) -> bool:
        return self.local_rank == 0

class DeepSpeedAccelerator(Accelerator):
    dist_backend: str = "nccl"
    
    def __init__(self, args, model, optimizer, lr_scheduler) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
    def set_up(self):
        assert deepspeed is not None, "Deepspeed is not installed."
        
        deepspeed.init_distributed(dist_backend=self.dist_backend)
        
        
        self.model_engine, self.optimizer, self.lr_scheduler = deepspeed.initialize(
            args=self.args,
            model=self.model,
            model_parameters=self.model.parameters(),
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler
        )
        
    def train_step(self, batch_data):
        pred = self.model_engine(batch_data)
        model_output = self.model.compute_loss(pred, batch_data)
        loss = model_output.loss
        
        self.model_engine.backward(loss)
        self.model_engine.step()
        
        return model_output
    
    def save_checkpoint(self, ckpt_id: int, extra_state: TrainerState):
        self.model_engine.save_checkpoint(
            self.args.save_dir,
            ckpt_id=ckpt_id,
            client_sd=asdict(extra_state),
        )
    
    def load_checkpoint(self, ckpt_id: int) -> TrainerState:
        _, cliend_sd = self.model_engine.load_checkpoint(
            self.args.save_dir,
            ckpt_id=ckpt_id,
        )
        
        return TrainerState(**cliend_sd)
    
    def should_log(self) -> bool:
        return deepspeed.local_rank() == 0


        
        