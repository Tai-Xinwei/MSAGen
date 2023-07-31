# -*- coding: utf-8 -*-
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.data.sampler import Sampler

from sfm.data.dataset import FoundationModelDataset
from sfm.logging import logger
from sfm.pipeline.accelerator.dataclasses import ModelOutput, TrainerState
from sfm.utils.data_utils import move_to_device


class Accelerator(ABC):
    @abstractmethod
    def set_up():
        pass

    @abstractmethod
    def train_step(self, grouped_batch_data: list) -> ModelOutput:
        pass

    @abstractmethod
    def save_checkpoint(
        self, ckpt_id: Union[int, str], extra_state: Optional[dict] = None
    ):
        pass

    @abstractmethod
    def load_checkpoint(self, ckpt_id: Union[int, str]) -> TrainerState:
        pass

    @abstractmethod
    def build_data_loader(self, train_data, val_data):
        pass

    @property
    @abstractmethod
    def grad_scale(self) -> float:
        pass

    def before_epoch(self, epoch: int):
        pass


class SingleNodeAccelerator(Accelerator):
    def __init__(self, args, model, optimizer, lr_scheduler, device: str) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        if not torch.cuda.is_available():
            self.device = "cpu"
        self.scaler = GradScaler(
            init_scale=self.args.grad_scaler_init, enabled=self.args.fp16
        )

    @property
    def grad_scale(self) -> float:
        return self.scaler.get_scale()

    def set_up(self):
        pass

    def build_data_loader(
        self, train_data: FoundationModelDataset, val_data: FoundationModelDataset
    ):
        self.train_sampler = RandomSampler(train_data)
        self.train_data_loader = DataLoader(
            train_data,
            sampler=self.train_sampler,
            batch_size=self.args.train_batch_size,
            collate_fn=train_data.collate,
            drop_last=True,
        )

        if val_data:
            self.val_data_loader = DataLoader(
                val_data,
                sampler=None,
                batch_size=self.args.val_batch_size,
                collate_fn=val_data.collate,
                drop_last=False,
            )

    def train_step(self, grouped_batch_data: list) -> ModelOutput:
        assert grouped_batch_data, "grouped_batch_data is empty"

        self.model.train()
        self.model.to(self.device)

        self.optimizer.zero_grad()
        for batch_data in grouped_batch_data:
            self.model.before_batch()
            batch_data = move_to_device(batch_data, self.device)

            with autocast(enabled=self.args.fp16):
                pred = self.model(batch_data)
                model_output = self.model.compute_loss(pred, batch_data)
                loss = model_output.loss / len(grouped_batch_data)
                self.scaler.scale(loss).backward()

            self.model.after_batch()

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()

        return model_output

    def save_checkpoint(self, ckpt_id: str, extra_state: Optional[dict] = None):
        save_dir = Path(self.args.save_dir)

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }

        if extra_state is not None:
            checkpoint.update(extra_state)
        logger.info("save checkpoint: {}", ckpt_id)
        torch.save(checkpoint, save_dir / ckpt_id)

        with open(save_dir / "checkpoint_list.txt", "a") as f:
            f.write(ckpt_id + "\n")

    def load_checkpoint(self, ckpt_id: str) -> TrainerState:
        save_dir = Path(self.args.save_dir)
        checkpoint_path = save_dir / ckpt_id
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        state = TrainerState(args=self.args)
        for k, v in checkpoint.items():
            if k not in ["model", "optimizer", "lr_scheduler"]:
                setattr(state, k, v)
        return state


class DdpAccelerator(SingleNodeAccelerator):
    def __init__(self, args, model, optimizer, lr_scheduler) -> None:
        super().__init__(args, model, optimizer, lr_scheduler, device="cuda")

    def set_up(self):
        assert "WORLD_SIZE" in os.environ, "WORLD_SIZE must be set to use DDP"
        assert "RANK" in os.environ, "RANK must be set to use DDP"
        assert "LOCAL_RANK" in os.environ, "LOCAL_RANK must be set to use DDP"

        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        logger.info(
            f"Initializing DDP by env: word size: {self.world_size}, rank: {self.rank}, local_rank{self.local_rank} ",
        )

        torch.distributed.init_process_group(
            backend=self.dist_backend,
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )

        torch.distributed.barrier()

        logger.info("DDP initialized.")

        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
        )

    def build_data_loader(
        self, train_data: FoundationModelDataset, val_data: FoundationModelDataset
    ):
        self.train_sampler = DistributedSampler(
            train_data, num_replicas=self.world_size, rank=self.rank
        )
        self.train_data_loader = DataLoader(
            train_data,
            sampler=self.train_sampler,
            batch_size=self.args.train_batch_size,
            collate_fn=train_data.collate,
            drop_last=True,
        )

        if val_data:
            self.val_data_loader = DataLoader(
                val_data,
                sampler=None,
                batch_size=self.args.eval_batch_size,
                collate_fn=val_data.collate,
                drop_last=False,
            )

    def before_epoch(self, epoch: int):
        self.train_sampler.set_epoch(epoch)

    def save_checkpoint(self, ckpt_id: str, extra_state: Optional[dict] = None):
        if self.rank == 0:
            super().save_checkpoint(ckpt_id, extra_state)

    def buld_sampler(self, data) -> Sampler:
        return DistributedSampler(data, num_replicas=self.world_size, rank=self.rank)


class DeepSpeedAccelerator(Accelerator):
    def __init__(self, args, model, optimizer, lr_scheduler) -> None:
        try:
            import deepspeed
        except ImportError:
            raise ImportError("Deepspeed is not installed.")

        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @property
    def grad_scale(self) -> float:
        return self.scaler.get_scale()

    def set_up(self):
        import deepspeed

        deepspeed.init_distributed(dist_backend=self.dist_backend)

        self.model_engine, self.optimizer, self.lr_scheduler = deepspeed.initialize(
            args=self.args,
            model=self.model,
            model_parameters=self.model.parameters(),
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )

    def build_data_loader(
        self, train_data: FoundationModelDataset, val_data: FoundationModelDataset
    ):
        raise NotImplementedError

    def train_step(self, grouped_batch_data):
        pred = self.model_engine(grouped_batch_data)
        model_output = self.model.compute_loss(pred, grouped_batch_data)
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
