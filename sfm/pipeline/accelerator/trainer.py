# -*- coding: utf-8 -*-
import copy
import random
import time
from pathlib import Path
from typing import Optional, Union

import deepspeed
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sfm.logging import logger, metric_logger
from sfm.pipeline.accelerator.accelerator import (
    Accelerator,
    DdpAccelerator,
    DeepSpeedAccelerator,
    GroupedBatchIter,
    SingleNodeAccelerator,
)
from sfm.pipeline.accelerator.dataclasses import (
    TrainerConfig,
    TrainerState,
    TrainLogOutput,
    TrainStrategy,
    ValidLogOutput,
)
from sfm.pipeline.accelerator.model import Model


def seed_everything(seed):
    deepspeed.runtime.utils.set_random_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class LossAccumulator(object):
    def __init__(self):
        self.sum = 0
        self.num_examples = 0

    def add(self, loss, num_examples):
        if loss is None:
            return

        if type(loss) == torch.Tensor:
            loss = loss.item()

        if type(num_examples) == torch.Tensor:
            num_examples = num_examples.item()

        if num_examples is None or num_examples <= 0:
            return

        if np.isnan(loss) or np.isinf(loss):
            return

        self.sum += loss * num_examples
        self.num_examples += num_examples

    def reset(self):
        self.sum = 0.0
        self.num_examples = 0

    @property
    def averge_loss(self):
        if self.num_examples == 0:
            return 0
        return self.sum / self.num_examples


class LogAccumulator(object):
    def __init__(self, world_size=1, allreduce_fn=None):
        self.sum = 0
        self.num_examples = 0
        self.extra_log = {}
        self.extra_log_num = {}
        self.start_time = time.time()
        self.allreduce_fn = allreduce_fn
        self.world_size = world_size

    def add(self, loss, num_examples, extra_log=None):
        if loss is None:
            return

        if type(loss) == torch.Tensor:
            loss = loss.item()

        if type(num_examples) == torch.Tensor:
            num_examples = num_examples.item()

        if num_examples is None or num_examples <= 0:
            return

        if np.isnan(loss) or np.isinf(loss):
            return

        self.sum += loss * num_examples
        self.num_examples += num_examples

        if extra_log is not None:
            for k, v in extra_log.items():
                if k not in self.extra_log and isinstance(v, (torch.Tensor, float)):
                    if isinstance(v, torch.Tensor):
                        self.extra_log[k] = v.item() * num_examples
                    else:
                        self.extra_log[k] = v * num_examples
                    self.extra_log_num[k] = 1 * num_examples
                elif k in self.extra_log and isinstance(v, (torch.Tensor, float)):
                    if isinstance(v, torch.Tensor):
                        self.extra_log[k] += v.item() * num_examples
                    else:
                        self.extra_log[k] += v * num_examples
                    self.extra_log_num[k] += 1 * num_examples

    def reset(self):
        self.sum = 0.0
        self.num_examples = 0
        self.start_time = time.time()
        for k, v in self.extra_log.items():
            self.extra_log[k] = 0.0
            self.extra_log_num[k] = 0

    @property
    def averge_loss(self):
        if self.num_examples == 0:
            return 0
        return self.sum / self.num_examples

    def _allreducelog(self, log_dict: dict = {}, log_num_dict: dict = {}):
        return self.allreduce_fn(log_dict, log_num_dict)

    @property
    def averge_log(self):
        self.extra_log["SamplePerSec"] = self.num_examples / (
            time.time() - self.start_time
        )
        self.extra_log_num["SamplePerSec"] = 1.0 / self.world_size
        if self.world_size == 1 or self.allreduce_fn is None:
            return {k: v / self.extra_log_num[k] for k, v in self.extra_log.items()}
        else:
            return self._allreducelog(self.extra_log, self.extra_log_num)


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
        loss_log_dict: Optional[dict] = {},
    ):
        super().__init__()
        self.args = args

        logger.info("Trainer args: {}", args)

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        if optimizer is None and args.strategy not in [
            TrainStrategy.Pipeline,
            TrainStrategy.ThreeD,
        ]:
            self.optimizer, self.lr_scheduler = model.config_optimizer()
        else:
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        self.accelerator = self.build_accelerator(loss_log_dict=loss_log_dict)
        self.accelerator.set_up()

        if args.strategy.find("Zero") == -1:
            self.accelerator.build_data_loader(train_data, valid_data)

        self.state = TrainerState(args=args)

        self.save_dir = Path(self.args.save_dir)

        if self.args.finetune_from_checkpoint_dir is not None:
            self.finetune_from_checkpoint_dir = Path(
                self.args.finetune_from_checkpoint_dir
            )
        else:
            self.finetune_from_checkpoint_dir = None

        self.world_size = self.accelerator.world_size

    def save_checkpoint(self, name: str):
        self.accelerator.save_checkpoint(name)

    def _load_checkpoint(self, path: Path, model_states_only: bool = False):
        checkpoint_list_path = path / "checkpoint_list.txt"
        latest_path = path / "latest"  # latest path for DeepSpeed

        checkpoint_last = None
        if model_states_only and self.args.finetune_from_checkpoint_id is not None:
            checkpoint_last = self.args.finetune_from_checkpoint_id
        elif checkpoint_list_path.exists():
            with open(checkpoint_list_path, "r") as f:
                checkpoint_list = f.read().splitlines()
            if len(checkpoint_list) > 0:
                checkpoint_last = checkpoint_list[-1]
        elif latest_path.exists():
            with open(latest_path, "r") as f:
                latest_list = f.read().splitlines()
            if len(latest_list) > 0:
                checkpoint_last = latest_list[-1]

        if checkpoint_last is not None:
            checkpoint_path = path / checkpoint_last
            if checkpoint_path.exists():
                if not model_states_only:
                    logger.info(f"Resume from checkpoint: {checkpoint_path}")
                else:
                    logger.info(f"Finetune from checkpoint: {checkpoint_path}")
                self.state = self.accelerator.load_checkpoint(
                    path,
                    checkpoint_last,
                    self.state,
                    model_states_only=model_states_only,
                )
            else:
                logger.warning(f"Checkpoint path {checkpoint_path} does not exist.")
        else:
            logger.warning(
                f"Non-empty checkpoint_list.txt or latest file is not present in {path}, or finetune_from_checkpoint_id is not provided. No checkpoint is loaded."
            )

    def resume(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._load_checkpoint(self.save_dir)

    def finetune_from_checkpoint(self):
        if self.finetune_from_checkpoint_dir is not None:
            self._load_checkpoint(
                self.finetune_from_checkpoint_dir, model_states_only=True
            )
        else:
            logger.warning("No finetune_from_checkpoint_dir is provided.")

    def build_accelerator(self, loss_log_dict: Optional[dict] = {}) -> Accelerator:
        if self.args.strategy == TrainStrategy.Single:
            return SingleNodeAccelerator(
                self.args,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                "cpu" if self.args.cpu else "cuda",
            )
        elif self.args.strategy in [
            TrainStrategy.Zero1,
            TrainStrategy.Zero2,
            TrainStrategy.Zero3,
            TrainStrategy.Pipeline,
            TrainStrategy.ThreeD,
        ]:
            return DeepSpeedAccelerator(
                self.args,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.train_data,
                self.valid_data,
                loss_log_dict=loss_log_dict,
            )
        elif self.args.strategy == TrainStrategy.DDP:
            return DdpAccelerator(
                self.args,
                self.model,
                self.optimizer,
                self.lr_scheduler,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.args.strategy}")

    def build_log_output(self, loss, extra_output=None) -> TrainLogOutput:
        try:
            lr = self.accelerator.lr_scheduler.get_last_lr()[0]
        except:
            lr = 0.0

        if type(loss) == torch.Tensor:
            loss = loss.item()

        return TrainLogOutput(
            loss=loss,
            grad_scale=self.accelerator.grad_scale,
            lr=lr,
            epoch=self.state.epoch,
            batch=self.state.batch,
            global_step=self.state.global_step,
            extra_output=extra_output,
        )

    def should_save_batch_checkpoint(self) -> bool:
        return (
            self.args.save_batch_interval > 0
            and (self.state.global_step + 1) % self.args.save_batch_interval == 0
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
            and (self.state.epoch + 1) % self.args.val_epoch_interval == 0
        )

    @property
    def train_data_loader(self) -> DataLoader:
        return GroupedBatchIter(
            self.accelerator.train_data_loader,
            self.args.gradient_accumulation_steps,
            drop_last=True,
        )

    @property
    def valid_data_loader(self) -> DataLoader:
        return self.accelerator.valid_data_loader

    def train(self):
        logger.info("Start training")
        logger.info(self.model)

        assert self.train_data_loader is not None

        self.model.before_training()
        if self.args.ifresume:
            self.resume()
        elif self.args.finetune_from_checkpoint_dir is not None:
            self.finetune_from_checkpoint()

        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            "Total number of parameters: {:,}, trainable: {:,}",
            total_num,
            trainable_num,
        )

        for epoch in range(self.args.total_num_epochs):
            self.state.epoch = epoch
            self.accelerator.before_epoch(epoch)

            logger.info("Start Training for epoch: {}", self.state.epoch)

            loss_accumulator = LossAccumulator()
            interval_loss_accumulator = LogAccumulator(
                self.accelerator.world_size, self.accelerator._allreducelog
            )
            for i, grouped_batch_data in enumerate(self.train_data_loader):
                model_output = self.accelerator.train_step(grouped_batch_data)
                loss_accumulator.add(model_output.loss, model_output.num_examples)
                interval_loss_accumulator.add(
                    model_output.loss,
                    model_output.num_examples,
                    model_output.log_output,
                )

                # Log and save checkpoint
                self.state.batch = i
                self.state.global_step += 1

                if self.should_do_batch_validate():
                    self.validate()

                if self.should_log():
                    log_output = self.build_log_output(
                        # model_output.loss, model_output.log_output
                        interval_loss_accumulator.averge_loss,
                        interval_loss_accumulator.averge_log,
                    )
                    interval_loss_accumulator.reset()
                    metric_logger.log(log_output, "train_inner")

                if self.should_save_batch_checkpoint():
                    checkpoint_name = f"checkpoint_E{epoch}_B{i}.pt"
                    self.save_checkpoint(checkpoint_name)

            log_output = self.build_log_output(loss_accumulator.averge_loss)
            metric_logger.log(log_output, "train")

            if self.should_do_epoch_validate():
                self.validate()

            self.accelerator.barrier()
            if self.should_save_epoch_checkpoint():
                checkpoint_name = f"checkpoint_E{epoch}.pt"
                self.save_checkpoint(checkpoint_name)

        self.model.after_training()

        logger.info("Finished Training")

    def validate(self):
        if self.valid_data_loader is None:
            logger.warning("No validation data, skip validation")
            return

        logger.info(
            "Start validation for epoch: {}, global step: {}",
            self.state.epoch,
            self.state.global_step,
        )

        # TODO: add other metrics
        loss_accumulator = LossAccumulator()
        interval_loss_accumulator = LogAccumulator(
            self.accelerator.world_size, self.accelerator._allreducelog
        )

        for idx, batch_data in enumerate(self.valid_data_loader):
            output = self.accelerator.valid_step(batch_data)
            loss_accumulator.add(output.valid_loss, output.num_examples)
            interval_loss_accumulator.add(
                output.valid_loss,
                output.num_examples,
                output.extra_output,
            )

            if (idx + 1) % self.args.val_batch_log_interval == 0:
                logger.info(
                    "Validtion batch: {} / {}, loss: {}",
                    idx + 1,
                    len(self.valid_data_loader),
                    output.valid_loss,
                )

        # DDP and Zero need to sync loss and num_examples at validation
        total_loss, num_examples = self.accelerator.sync_valid_loss(
            loss_accumulator.sum, loss_accumulator.num_examples
        )
        valid_log = ValidLogOutput(
            valid_loss=total_loss / num_examples,
            num_examples=num_examples,
            extra_output=interval_loss_accumulator.averge_log,
        )

        metric_logger.log(valid_log, "valid")
