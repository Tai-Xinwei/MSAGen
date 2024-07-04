# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Type, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset

try:
    from nnscaler.parallel import (
        ComputeConfig,
        build_optimizer,
        deduped_state_dict,
        load_deduped_state_dict,
        merge_state_dicts,
        parallelize,
    )
    from nnscaler.runtime.device import DeviceGroup

    ENABLE_NNSCALER = True
except:
    ENABLE_NNSCALER = False

import logging

from sfm.data.dataset import Batch, FoundationModelDataset
from sfm.pipeline.accelerator.dataclasses import (
    ModelOutput,
    TrainerState,
    ValidLogOutput,
)
from sfm.utils.move_to_device import move_to_device

from .accelerator import Accelerator, safe_div
from .dataclasses import NNScalerTrainConfig

logger = logging.getLogger(__name__)


class NNScalerAccelerator(Accelerator):
    def __init__(
        self,
        args: NNScalerTrainConfig,
        model_cls: Type[torch.nn.Module],
        model_init: Callable[[], torch.nn.Module],
        optimizer_init: Callable[
            [Iterator[torch.nn.Parameter], NNScalerTrainConfig], Optimizer
        ],
        lr_scheduler_init: Callable[[Optimizer, NNScalerTrainConfig], LRScheduler],
        train_data,
        valid_data,
    ) -> None:
        super().__init__()
        self.args = args
        self.model_cls = model_cls
        self.model_init = model_init
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init

        self.nnscaler_config = ComputeConfig(
            plan_ngpus=args.plan_ngpus,
            runtime_ngpus=args.runtime_ngpus,
            dynamic_shape=args.dynamic_shape,
            use_zero=args.use_zero,
            zero_ngroups=args.zero_ngroups,
            inference_only=args.inference_only,
            use_end2end=args.use_end2end,
            use_pipeline=args.use_pipeline,
            pipeline_nmicros=args.pipeline_nmicros,
            pipeline_nstages=args.pipeline_nstages,
            pipeline_scheduler=args.pipeline_scheduler,
            pas_config={"update_freq": self.args.gradient_accumulation_steps},
        )

        self.world_size = self.nnscaler_config.runtime_ngpus

        torch.distributed.init_process_group(
            backend="nccl",
        )

        # DeviceGroup().local_rank is int(os.environ.get('LOCAL_RANK'))
        self.local_rank = DeviceGroup().local_rank
        self.build_data_loader(train_data, valid_data)

        self.save_dir = Path(self.args.save_dir)

    def set_up(self):
        # need a dummy input to trace the model graph
        dummy_input = {"batched_data": next(iter(self.train_data_loader))}

        # create a ParallelModule type class from the original model class,
        # the created ParallelModule is a distributed version of the original model
        self.nnscaler_model_cls = parallelize(
            self.model_cls,
            dummy_input,
            "autodist",
            self.nnscaler_config,
            cube_savedir=self.args.cache_savedir,
            reuse=self.args.file_reuse,
            module_fn=self.model_init,
            init_module_params=self.args.init_module_params,
            broadcast_strategy=self.args.broadcast_strategy,
        )
        self.model = self.nnscaler_model_cls(True).cuda()
        self.optimizer = build_optimizer(self.model, self.optimizer_init, self.args)
        self.lr_scheduler = self.lr_scheduler_init(self.optimizer, self.args)

    def barrier(self):
        torch.distributed.barrier()

    def build_data_loader(
        self, train_data: FoundationModelDataset, val_data: FoundationModelDataset
    ):
        scale_unit_num = (
            self.nnscaler_config.runtime_ngpus // self.nnscaler_config.plan_ngpus
        )

        if self.args.dynamic_loader:
            raise NotImplementedError("not support dynamic_loader now")
        elif self.args.ifstack:
            raise NotImplementedError("not support ifstack now")
        else:
            train_batch_size_per_nnsacler_scale_unit = self.args.train_batch_size // (
                scale_unit_num * self.args.gradient_accumulation_steps
            )
            assert (
                train_batch_size_per_nnsacler_scale_unit > 0
            ), "train_batch_size_per_nnsacler_scale_unit should be greater than 0"

            if not isinstance(train_data, IterableDataset):
                self.train_sampler = DistributedSampler(
                    train_data,
                    num_replicas=scale_unit_num,
                    rank=torch.distributed.get_rank()
                    // self.nnscaler_config.plan_ngpus,
                )
                self.train_data_loader = DataLoader(
                    train_data,
                    sampler=self.train_sampler,
                    batch_size=train_batch_size_per_nnsacler_scale_unit,
                    collate_fn=train_data.collate,
                    drop_last=True,
                )
            else:
                raise NotImplementedError("not support IterableDataset now")

        if val_data:
            valid_batch_size_per_scale_unit = self.args.val_batch_size // (
                scale_unit_num * self.args.gradient_accumulation_steps
            )
            assert (
                valid_batch_size_per_scale_unit > 0
            ), "valid_batch_size_per_scale_unit should be greater than 0"

            validsampler = DistributedSampler(
                val_data,
                num_replicas=scale_unit_num,
                rank=torch.distributed.get_rank() // self.nnscaler_config.plan_ngpus,
                shuffle=False,
            )
            self.valid_data_loader = DataLoader(
                val_data,
                sampler=validsampler,
                batch_size=valid_batch_size_per_scale_unit,
                collate_fn=val_data.collate,
                drop_last=False,
            )
        else:
            self.valid_data_loader = None

    def before_epoch(self, epoch: int):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

    def train_step(self, grouped_batch_data: List[Batch]) -> ModelOutput:
        assert grouped_batch_data, "grouped_batch_data is empty"

        self.model.train()
        self.optimizer.zero_grad()

        success_batch_count = 0
        sample_count = 0
        total_loss = 0.0
        total_log_output = {}

        data_len = len(grouped_batch_data)

        def scale_fn(loss: torch.Tensor):
            if torch.isnan(loss).item() or torch.isinf(loss).item():
                logger.info("loss is nan or inf. skip this batch")
                mask = torch.isnan(loss) | torch.isinf(loss)
                loss[mask] = 0.0
            return loss * self.grad_scale / data_len

        for idx, batch_data in enumerate(grouped_batch_data):
            batch_data = move_to_device(batch_data, self.local_rank)

            model_outputs = self.model.train_step(
                [batch_data], is_dummy_batch=None, scale_fn=scale_fn
            )
            model_output = ModelOutput(*model_outputs[0])

            success_batch_count += 1

            self._accumulate_log_output(
                total_log_output,
                model_output.log_output,
                sample_count,
                model_output.num_examples,
            )
            sample_count += model_output.num_examples
            total_loss += model_output.loss * model_output.num_examples

        if success_batch_count > 0:
            self.optimizer.step()

        self.lr_scheduler.step()
        model_output.num_examples = sample_count
        model_output.loss = safe_div(total_loss, sample_count)
        model_output.log_output = total_log_output
        return model_output

    def valid_step(self, batch_data: Batch, epoch: int = 0) -> ValidLogOutput:
        raise NotImplementedError

    def save_checkpoint(
        self, ckpt_id: Union[int, str], extra_state: Optional[dict] = None
    ):
        model_state_dict, opt_state_dict = deduped_state_dict(
            self.model, self.optimizer
        )
        checkpoint = {
            "model": model_state_dict,
            "optimizer": opt_state_dict,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "extra_state": extra_state,
        }
        torch.save(checkpoint, str(self.save_dir / f"{ckpt_id}.{DeviceGroup().rank}"))
        torch.save(checkpoint, str(self.save_dir / f"latest.pt.{DeviceGroup().rank}"))

    def load_checkpoint(
        self,
        ckpt_dir: Path,
        ckpt_id: Union[int, str],
        trainer_state: TrainerState,
        model_states_only: bool = False,
    ) -> TrainerState:
        ckpt_id = ".".join(ckpt_id.split(".")[:-1])
        checkpoint = torch.load(f"{ckpt_dir / ckpt_id}.{DeviceGroup().rank}")
        if model_states_only:
            load_deduped_state_dict(self.model, checkpoint["model"])
        else:
            load_deduped_state_dict(
                self.model, checkpoint["model"], self.optimizer, checkpoint["optimizer"]
            )
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if checkpoint["extra_state"] is not None:
                for k, v in checkpoint["extra_state"].items():
                    setattr(trainer_state, k, v)
        return trainer_state

    def sync_valid_loss(self, total_loss, num_examples):
        raise NotImplementedError

    @property
    def grad_scale(self) -> float:
        return 1.0

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

        return {k: safe_div(v, log_num_dict[k]) for k, v in log_dict.items()}


def merge_nnscaler_checkpoint(
    ckpt_dir: Path, ckpt_id_prefix: str = "latest.pt.", model_states_only: bool = True
):
    """
    Args:
        ckpt_dir: The path of the checkpoints saved, note that all the checkpoints of all ranks should in this folder.
        ckpt_id_prefix: The common prefix of the checkpoints file, for example, `latest.pt.`.
        model_states_only: if only merge the model state dict.
    Returns:
        Tuple[Dict, Dict]: model_state_dict, optimizer_state_dict
    """
    import os
    import re

    pattern = re.compile(f"{ckpt_id_prefix}\\d+")
    model_state_dicts, opt_state_dict = [], []
    for filename in os.listdir(ckpt_dir):
        if pattern.match(filename):
            checkpoint = torch.load(filename)
            model_state_dicts.append(checkpoint["model"])
            if not model_states_only:
                opt_state_dict.append(checkpoint["optimizer"])
    return merge_state_dicts(
        model_state_dicts, None if model_states_only else opt_state_dict
    )
