# -*- coding: utf-8 -*-
import copy
import random
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sfm.logging import logger
from sfm.pipeline.accelerator.accelerator import (
    Accelerator,
    DdpAccelerator,
    DeepSpeedAccelerator,
    SingleNodeAccelerator,
)
from sfm.pipeline.accelerator.dataclasses import (
    ModelOutput,
    TrainerConfig,
    TrainerState,
    TraingStrategy,
    TrainLogOutput,
    ValidLogOutput,
)
from sfm.pipeline.accelerator.model import Model


def seed_everything(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class GroupedBatchIter(object):
    """
    This class is used to group batches into a larger batch. i.e., gradient accumulation.
    """

    def __init__(self, it, group_size, drop_last=False):
        self.it = it
        self.group_size = group_size
        self.drop_last = drop_last

    def __iter__(self):
        chunk = []
        for item in self.it:
            chunk.append(item)
            if len(chunk) == self.group_size:
                yield chunk
                chunk = []
        if not self.drop_last and chunk:
            yield chunk

    def __len__(self):
        if self.drop_last:
            return len(self.it) // self.group_size
        else:
            return (len(self.it) + self.group_size - 1) // self.group_size


class Trainer(object):
    def __init__(
        self,
        args: TrainerConfig,
        model: Model,
        train_data: Dataset,
        valid_data: Optional[Dataset] = None,
        test_data: Optional[Dataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        super().__init__()
        self.args = args

        logger.info("Trainer args: {}", args)

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data

        if optimizer is None:
            self.optimizer, self.lr_scheduler = model.config_optimizer()
        else:
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        self.accelerator = self.build_accelerator()
        self.accelerator.set_up()

        if args.strategy.find("zero") == -1:
            self.accelerator.build_data_loader(train_data, valid_data)
        else:
            self.train_data = train_data
            self.valid_data = valid_data

        self.state = TrainerState(args=args)

        self.save_dir = Path(self.args.save_dir)

    def save_checkpoint(self, name: str):
        self.accelerator.save_checkpoint(name)

    def load_checkpoint(self, ckpt_id: Union[int, str]):
        self.state = self.accelerator.load_checkpoint(ckpt_id)

    def resume(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_list_path = self.save_dir / "checkpoint_list.txt"
        if checkpoint_list_path.exists():
            with open(checkpoint_list_path, "r") as f:
                checkpoint_list = f.read().splitlines()
            if len(checkpoint_list) > 0:
                checkpoint_last = checkpoint_list[-1]
                checkpoint_path = self.save_dir / checkpoint_last
                if checkpoint_path.exists():
                    logger.info("Resume from checkpoint: ", checkpoint_path)
                    self.load_checkpoint(checkpoint_last)
                else:
                    logger.info("Not resume from checkpoint")
        else:
            with open(checkpoint_list_path, "w") as f:
                f.write("")

    def build_accelerator(self) -> Accelerator:
        if self.args.strategy == TraingStrategy.Single:
            return SingleNodeAccelerator(
                self.args,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                "cpu" if self.args.cpu else "cuda",
            )
        elif self.args.strategy in [
            TraingStrategy.Zero1,
            TraingStrategy.Zero2,
            TraingStrategy.Zero3,
        ]:
            return DeepSpeedAccelerator(
                self.args,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.train_data,
                self.valid_data,
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

    def build_log_output(self, model_output: ModelOutput) -> TrainLogOutput:
        return TrainLogOutput(
            loss=model_output.loss.item(),
            grad_scale=self.accelerator.grad_scale,
            lr=self.lr_scheduler.get_last_lr()[0],
            epoch=self.state.epoch,
            batch=self.state.batch,
            global_step=self.state.global_step,
            extra_output=model_output.log_output,
        )

    def should_save_batch_checkpoint(self) -> bool:
        return (
            self.args.save_batch_interval > 0
            and self.state.global_step % self.args.save_interval == 0
        )

    def should_save_epoch_checkpoint(self) -> bool:
        return (
            self.args.save_epoch_interval > 0
            and (self.state.epoch + 1) % self.args.save_epoch_interval == 0
        )

    def should_log(self) -> bool:
        return (
            self.args.log_interval > 0
            and self.state.global_step % self.args.log_interval == 0
        )

    def should_do_batch_validate(self) -> bool:
        return (
            self.args.val_batch_interval > 0
            and self.state.global_step % self.args.val_batch_interval == 0
        )

    def should_do_epoch_validate(self) -> bool:
        return (
            self.args.val_epoch_interval > 0
            and self.state.epoch % self.args.val_epoch_interval == 0
        )

    @property
    def train_data_loader(self) -> DataLoader:
        return self.accelerator.train_data_loader

    @property
    def valid_data_loader(self) -> DataLoader:
        return self.accelerator.valid_data_loader

    def train(self):
        logger.info("Start training")
        logger.info(self.model)

        seed_everything(self.args.seed)

        assert self.train_data_loader is not None

        self.model.before_training()
        self.resume()

        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            "Total number of parameters: {:,}, trainable: {:,}",
            total_num,
            trainable_num,
        )

        for epoch in range(self.args.epochs):
            self.state.epoch = epoch
            self.accelerator.before_epoch(epoch)

            if (
                self.args.strategy == TraingStrategy.DDP
                or self.args.strategy == TraingStrategy.Single
            ):
                _iter = GroupedBatchIter(
                    self.train_data_loader, self.args.update_freq, drop_last=True
                )
            else:
                _iter = iter(copy.deepcopy(self.train_data_loader))

            for i, grouped_batch_data in enumerate(_iter):
                model_output = self.accelerator.train_step(grouped_batch_data)

                self.state.batch = i
                self.state.global_step += 1

                if self.should_log():
                    log_output = self.build_log_output(model_output)
                    with logger.contextualize(wandb_log=log_output):
                        logger.info(log_output)

                if self.should_do_batch_validate():
                    self.validate()

                if self.should_save_batch_checkpoint():
                    checkpoint_name = f"checkpoint_E{epoch}_B{i}.pt"
                    self.save_checkpoint(checkpoint_name)

            if self.should_do_epoch_validate():
                self.validate()

            if self.should_save_epoch_checkpoint():
                checkpoint_name = f"checkpoint_E{epoch}.pt"
                self.save_checkpoint(checkpoint_name)

        self.model.after_training()

        logger.info("Finished Training")

    def validate(self):
        if self.valid_data_loader is None:
            logger.warning("No validation data, skip validation")
            return

        logger.info("Start validation for epoch: {}", self.state.epoch)

        # TODO: add other metrics
        total_loss = 0.0
        num_examples = 0
        _iter = iter(copy.deepcopy(self.valid_data_loader))

        for idx, batch_data in enumerate(_iter):
            output = self.accelerator.valid_step(batch_data)

            total_loss += output.valid_loss * output.num_examples
            num_examples += output.num_examples

            if (idx + 1) % self.args.val_batch_log_interval == 0:
                logger.info(
                    "Validtion batch: {} / {}, loss: {}",
                    idx + 1,
                    len(self.valid_data_loader),
                    output.valid_loss,
                )

        avg_loss = total_loss / num_examples
        with logger.contextualize(
            wandb_log=ValidLogOutput(
                valid_loss=avg_loss, num_examples=num_examples, extra_output={}
            )
        ):
            logger.info("Validation loss: {}", avg_loss)
