# -*- coding: utf-8 -*-
import math
import multiprocessing
import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Union

import deepspeed
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from sfm.data.data_utils import batch_by_size
from sfm.data.dataset import Batch, Data, FoundationModelDataset
from sfm.data.dynamics_loader import DynamicBatchSampler, DynamicDataLoader
from sfm.logging import logger
from sfm.pipeline.accelerator.dataclasses import (
    ModelOutput,
    TrainerState,
    TrainStrategy,
    ValidLogOutput,
)
from sfm.pipeline.accelerator.fp16_scaler import FP16Scaler
from sfm.utils.move_to_device import move_to_device
from sfm.utils.PPEngine import initialize as initialize_pp_engine

from .pipeline_module import SFMPipelineModule


class Accelerator(ABC):
    @abstractmethod
    def set_up():
        pass

    @abstractmethod
    def train_step(self, grouped_batch_data: List[Batch]) -> ModelOutput:
        pass

    @abstractmethod
    def valid_step(self, batch_data: Batch) -> ValidLogOutput:
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

    @abstractmethod
    def barrier(self):
        pass

    @abstractmethod
    def sync_valid_loss(self, total_loss, num_examples):
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
        self.world_size = 1
        if not torch.cuda.is_available():
            self.device = "cpu"
        self.scaler = FP16Scaler(
            init_scale=self.args.grad_scaler_init, enabled=self.args.fp16
        )

        if args.fp16:
            self.model = self.model.half()

    @property
    def grad_scale(self) -> float:
        return self.scaler.scale

    def set_up(self):
        pass

    def barrier(self):
        pass

    def build_data_loader(
        self, train_data: FoundationModelDataset, valid_data: FoundationModelDataset
    ):
        self.train_sampler = RandomSampler(train_data)
        self.train_data_loader = DataLoader(
            train_data,
            sampler=self.train_sampler,
            batch_size=self.args.train_batch_size,
            collate_fn=train_data.collate,
            drop_last=True,
        )

        if valid_data:
            self.valid_data_loader = DataLoader(
                valid_data,
                sampler=None,
                batch_size=self.args.val_batch_size,
                collate_fn=valid_data.collate,
                drop_last=False,
            )
        else:
            self.valid_data_loader = None

    def train_step(self, grouped_batch_data: List[Batch]) -> ModelOutput:
        assert grouped_batch_data, "grouped_batch_data is empty"

        self.model.train()
        self.model.to(self.device)

        self.optimizer.zero_grad()
        success_batch_count = 0
        for batch_data in grouped_batch_data:
            self.model.before_batch()
            batch_data = move_to_device(batch_data, self.device)

            pred = self.model(batch_data)
            model_output = self.model.compute_loss(pred, batch_data)
            loss = model_output.loss / len(grouped_batch_data)

            if torch.isnan(loss).item() or torch.isinf(loss).item():
                logger.info("loss is nan or inf. skip this batch")
                continue
            else:
                success_batch_count += 1
                self.scaler.backward(loss)

            self.model.after_batch()

        if success_batch_count > 0:
            self.scaler.step(self.model, self.optimizer, self.args.gradient_clipping)

        self.lr_scheduler.step()

        return model_output

    def valid_step(self, batch_data: Batch) -> ValidLogOutput:
        self.model.eval()
        self.model.to(self.device)

        batch_data = move_to_device(batch_data, self.device)
        with torch.no_grad():
            pred = self.model(batch_data)
            model_output = self.model.compute_loss(pred, batch_data)

        if hasattr(batch_data, "batch_size"):
            num_examples = batch_data.batch_size
        elif hasattr(model_output, "num_examples"):
            num_examples = model_output.num_examples
        else:
            logger.info("num_examples is not found. set to None")
            num_examples = None

        return ValidLogOutput(
            valid_loss=model_output.loss.item(),
            num_examples=num_examples,
            extra_output=model_output.log_output,
        )

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

    def sync_valid_loss(self, total_loss, num_examples):
        return total_loss, num_examples

    @staticmethod
    def _allreducelog(log_dict: dict = {}, log_num_dict: dict = {}):
        return None


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

        master_addr = os.environ.get("MASTER_ADDR", "")
        master_port = os.environ.get("MASTER_PORT", "")

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        multiprocessing.set_start_method("spawn", force=True)

        logger.critical(
            f"Initializing DDP by env://. word size: {self.world_size}, rank: {self.rank}, local_rank: {self.local_rank}, master_addr: {master_addr}, master_port: {master_port}"
        )
        torch.distributed.init_process_group(
            backend=self.args.dist_backend,
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )

        torch.distributed.barrier()

        logger.success("DDP initialized.")

        self.model.to(self.device)
        self.ddp_model = DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
        )

    def barrier(self):
        torch.distributed.barrier()

    def train_step(self, grouped_batch_data: List[Batch]) -> ModelOutput:
        assert grouped_batch_data, "grouped_batch_data is empty"

        self.ddp_model.train()
        self.optimizer.zero_grad()

        success_batch_count = 0
        for idx, batch_data in enumerate(grouped_batch_data):
            self.model.before_batch()
            batch_data = move_to_device(batch_data, self.device)

            # No sync for gradient accumulation
            maybe_no_sync = (
                self.ddp_model.no_sync()
                if idx != len(grouped_batch_data) - 1
                else nullcontext()
            )

            with maybe_no_sync:
                pred = self.ddp_model(batch_data)
                model_output = self.model.compute_loss(pred, batch_data)
                loss = model_output.loss / len(grouped_batch_data)

                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    logger.info("loss is nan or inf. skip this batch")
                    continue
                else:
                    success_batch_count += 1
                    self.scaler.backward(loss)

            self.model.after_batch()

        if success_batch_count > 0:
            self.scaler.step(self.model, self.optimizer, self.args.gradient_clipping)

        self.lr_scheduler.step()

        return model_output

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
            validsampler = torch.utils.data.distributed.DistributedSampler(
                val_data, num_replicas=self.world_size, shuffle=False
            )
            self.valid_data_loader = DataLoader(
                val_data,
                sampler=validsampler,
                batch_size=self.args.val_batch_size,
                collate_fn=val_data.collate,
                drop_last=False,
            )
        else:
            self.valid_data_loader = None

    def before_epoch(self, epoch: int):
        self.train_sampler.set_epoch(epoch)

    def save_checkpoint(self, ckpt_id: str, extra_state: Optional[dict] = None):
        if self.rank == 0:
            super().save_checkpoint(ckpt_id, extra_state)

        torch.distributed.barrier()

    def sync_valid_loss(self, total_loss, num_examples):
        total_loss = torch.Tensor([total_loss]).cuda(self.device)
        num_examples = torch.Tensor([num_examples * 1.0]).cuda(self.device)
        torch.distributed.all_reduce(total_loss)
        torch.distributed.all_reduce(num_examples)
        total_loss = total_loss.item()
        num_examples = num_examples.item()

        return total_loss, num_examples

    @staticmethod
    def _allreducelog(log_dict: dict = {}, log_num_dict: dict = {}):
        for k, v in log_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            log_dict[k] = v.item()

        for k, v in log_num_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            log_num_dict[k] = v.item()

        return {k: v / log_num_dict[k] for k, v in log_dict.items()}


class DeepSpeedAccelerator(Accelerator):
    def __init__(
        self, args, model, optimizer, lr_scheduler, train_data, valid_data
    ) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_data = train_data
        self.valid_data = valid_data

    @property
    def grad_scale(self) -> float:
        return self.optimizer.cur_scale

    def set_ds_config(self):
        if (
            isinstance(self.args.deepspeed_config, str)
            and len(self.args.deepspeed_config) > 0
        ):
            import json

            try:
                with open(self.args.deepspeed_config) as deepspeed_config_file:
                    self.args.deepspeed_config = json.load(deepspeed_config_file)
            except Exception as e:
                logger.warning(
                    f"Failed to load deepspeed config from {self.args.deepspeed_config}, using default config instead. Error message: {e}"
                )
                from sfm.utils.defaultdsconfig import DEFAULT_DS_CONFIG

                self.args.deepspeed_config = DEFAULT_DS_CONFIG
        elif (
            isinstance(self.args.deepspeed_config, str)
            and len(self.args.deepspeed_config) == 0
        ):
            from sfm.utils.defaultdsconfig import DEFAULT_DS_CONFIG

            self.args.deepspeed_config = DEFAULT_DS_CONFIG

            self.args.deepspeed_config["train_batch_size"] = self.args.train_batch_size
            self.args.deepspeed_config[
                "gradient_accumulation_steps"
            ] = self.args.gradient_accumulation_steps

            self.args.deepspeed_config["fp16"]["enabled"] = self.args.fp16
            self.args.deepspeed_config["fp16"]["auto_cast"] = self.args.auto_cast
            self.args.deepspeed_config["fp16"]["initial_scale_power"] = round(
                math.log2(self.args.grad_scaler_init)
            )

            if (
                self.args.strategy == TrainStrategy.Zero1
                or self.args.strategy == TrainStrategy.Pipeline
            ):
                self.args.deepspeed_config["zero_optimization"]["stage"] = 1
            elif self.args.strategy == TrainStrategy.Zero2:
                self.args.deepspeed_config["zero_optimization"]["stage"] = 2
            elif self.args.strategy == TrainStrategy.Zero3:
                self.args.deepspeed_config["zero_optimization"]["stage"] = 3
            else:
                raise ValueError(
                    f"Unsupported accelerator strategy: {self.args.strategy}"
                )

            self.args.deepspeed_config["optimizer"]["params"]["lr"] = self.args.max_lr

            self.args.deepspeed_config["scheduler"]["params"][
                "total_num_steps"
            ] = self.args.total_num_steps
            self.args.deepspeed_config["scheduler"]["params"][
                "warmup_num_steps"
            ] = self.args.warmup_num_steps
            self.args.deepspeed_config["scheduler"]["params"][
                "warmup_max_lr"
            ] = self.args.max_lr

            self.args.deepspeed_config[
                "gradient_clipping"
            ] = self.args.gradient_clipping
            self.args.deepspeed_config["steps_per_print"] = self.args.log_interval

            self.args.deepspeed_config["wandb"]["enabled"] = self.args.wandb
            self.args.deepspeed_config["wandb"]["team"] = self.args.wandb_team
            self.args.deepspeed_config["wandb"]["group"] = self.args.wandb_group
            self.args.deepspeed_config["wandb"]["project"] = self.args.wandb_project
            return

    def get_unfreeze_param_list(self, unfreeze_param_name_list: str):
        if unfreeze_param_name_list == "":
            return None
        unfreeze_param = []
        unfreeze_param_name_list = list(
            filter(lambda x: x != "", unfreeze_param_name_list.split(","))
        ) + ["dummy"]
        for name, param in self.model.named_parameters():
            for param_name in unfreeze_param_name_list:
                if name.find(param_name) != -1:
                    unfreeze_param.append(param)
                    break
        return unfreeze_param

    def set_up(self):
        deepspeed.init_distributed(dist_backend=self.args.dist_backend)
        self.set_ds_config()

        if self.args.strategy == TrainStrategy.Pipeline:
            assert (
                self.args.pipeline_model_parallel_size > 0
            ), f"invalid model parallel size: {self.args.pipeline_model_parallel_size}"

            pp_partition_layer_name = self.args.deepspeed_config.get(
                "pp_partition_layer_name", self.args.pp_partition_layer_name
            )

            if pp_partition_layer_name not in ["parameters", "uniform", "manual"]:
                pp_partition_layer_name = "type:" + pp_partition_layer_name

            self.model = SFMPipelineModule(
                self.model,
                loss_fn=lambda pred, label: self.model.compute_loss(pred, label).loss,
                num_stages=self.args.deepspeed_config.get(
                    "num_pp_stages", self.args.pipeline_model_parallel_size
                ),
                partition_method=pp_partition_layer_name,
                part_list=self.args.pp_part_list,
            )
            unfreeze_params = self.get_unfreeze_param_list(
                self.args.unfreeze_param_list
            )
            model_parameters = (
                unfreeze_params
                if unfreeze_params is not None
                else self.model.parameters()
            )
            (
                self.model_engine,
                self.optimizer,
                self.train_data_loader,
                self.lr_scheduler,
            ) = initialize_pp_engine(
                args=self.args,
                model=self.model,
                model_parameters=model_parameters,
                training_data=self.train_data,
                collate_fn=self.train_data.collate,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
            )
        else:
            (
                self.model_engine,
                self.optimizer,
                self.train_data_loader,
                self.lr_scheduler,
            ) = deepspeed.initialize(
                args=self.args,
                model=self.model,
                model_parameters=self.model.parameters(),
                training_data=self.train_data,
                collate_fn=self.train_data.collate,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
            )
        self.build_data_loader(self.train_data, self.valid_data)

    @property
    def world_size(self) -> int:
        return self.model_engine.dp_world_size

    def barrier(self):
        deepspeed.comm.barrier()

    def build_data_loader(
        self, train_data: FoundationModelDataset, val_data: FoundationModelDataset
    ):
        if self.args.dynamic_loader:
            assert (
                self.args.strategy is not TrainStrategy.Pipeline
            ), "dyanmic loader is not supported in pipeline mode"

            train_batch_size_per_gpu = self.args.train_batch_size // (
                self.model_engine.dp_world_size * self.args.gradient_accumulation_steps
            )
            assert (
                train_batch_size_per_gpu > 0
            ), "train_batch_size_per_gpu should be greater than 0"

            # sampler = torch.utils.data.distributed.DistributedSampler(
            #     self.train_data,
            #     num_replicas=self.model_engine.dp_world_size,
            #     rank=self.model_engine.global_rank,
            #     shuffle=False,
            # )
            # dynamic_sampler = DynamicBatchSampler(
            #     sampler,
            #     num_tokens_fn=self.train_data.num_tokens,
            #     max_size=self.args.max_num_aa,
            #     max_tokens=self.args.max_tokens,
            # )
            # self.train_data_loader = DataLoader(
            #     dataset=self.train_data,
            #     collate_fn=self.train_data.collate,
            #     batch_sampler=dynamic_sampler,
            # )

            self.train_data_loader = DynamicDataLoader(
                dataset=self.train_data,
                batch_by_size_fn=batch_by_size,
                max_tokens=self.args.max_tokens,
                max_length=self.args.max_length,
                num_tokens_fn=self.train_data.num_tokens,
                collate_fn=self.train_data.collate,
                shuffle=True,
                drop_last=False,
                num_replicas=self.model_engine.dp_world_size,
                rank=self.model_engine.global_rank,
            )

        if self.valid_data:
            validsampler = torch.utils.data.distributed.DistributedSampler(
                self.valid_data,
                num_replicas=self.model_engine.dp_world_size,
                rank=self.model_engine.global_rank,
                shuffle=False,
            )
            valid_batch_size_per_gpu = self.args.val_batch_size // (
                self.model_engine.dp_world_size * self.args.gradient_accumulation_steps
            )
            assert (
                valid_batch_size_per_gpu > 0
            ), "valid_batch_size_per_gpu should be greater than 0"

            self.valid_data_loader = DataLoader(
                self.valid_data,
                sampler=validsampler,
                batch_size=valid_batch_size_per_gpu,
                collate_fn=self.valid_data.collate,
                drop_last=False,
            )
        else:
            self.valid_data_loader = None

    def train_step(self, grouped_batch_data) -> ModelOutput:
        self.model_engine.module.train()
        if self.args.strategy == TrainStrategy.Pipeline:
            loss = self.model_engine.train_batch()
            model_output = ModelOutput(
                loss=loss,
                num_examples=self.args.deepspeed_config["train_batch_size"],
                log_output={"loss": loss},
            )
        else:
            for idx, batch_data in enumerate(grouped_batch_data):
                self.model_engine.tput_timer.start()
                batch_data = move_to_device(
                    batch_data, device=self.args.local_rank, non_blocking=True
                )
                self.model.before_batch()
                pred = self.model_engine(batch_data)

                model_output = self.model.compute_loss(pred, batch_data)
                loss = model_output.loss

                self.model.after_batch()

                self.model_engine.backward(loss)
                self.model_engine.step()

        torch.cuda.empty_cache()
        return model_output

    def valid_step(self, batch_data: Data) -> ValidLogOutput:
        self.model_engine.module.eval()
        batch_data = move_to_device(
            batch_data, device=self.args.local_rank, non_blocking=True
        )

        pred = self.model_engine(batch_data)
        model_output = self.model.compute_loss(pred, batch_data)

        torch.cuda.empty_cache()
        return ValidLogOutput(
            valid_loss=model_output.loss.detach().item(),
            num_examples=model_output.num_examples,
            extra_output=model_output.log_output,
        )

    def save_checkpoint(self, ckpt_id: str, extra_state: Optional[dict] = None):
        self.model_engine.save_checkpoint(
            self.args.save_dir,
            client_state={"ckpt_id": ckpt_id},
        )

    def load_checkpoint(self, ckpt_id: int) -> TrainerState:
        _, cliend_sd = self.model_engine.load_checkpoint(
            self.args.save_dir,
            # ckpt_id=ckpt_id,
        )

        return TrainerState(**cliend_sd)

    def sync_valid_loss(self, total_loss, num_examples):
        total_loss = torch.Tensor([total_loss]).cuda()
        num_examples = torch.Tensor([num_examples * 1.0]).cuda()
        deepspeed.comm.all_reduce(total_loss)
        deepspeed.comm.all_reduce(num_examples)
        total_loss = total_loss.item()
        num_examples = num_examples.item()

        return total_loss, num_examples

    @staticmethod
    def _allreducelog(log_dict: dict = {}, log_num_dict: dict = {}):
        for k, v in log_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            deepspeed.comm.all_reduce(v)
            log_dict[k] = v.item()

        for k, v in log_num_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            deepspeed.comm.all_reduce(v)
            log_num_dict[k] = v.item()

        return {k: v / log_num_dict[k] for k, v in log_dict.items()}
