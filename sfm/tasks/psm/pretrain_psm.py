# -*- coding: utf-8 -*-
import os
import sys

import wandb  # isort:skip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from dataclasses import dataclass
from typing import Any, Dict, Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from sfm.data.psm_data.pipeline import UnifiedBatchedIterableDataset
from sfm.data.psm_data.unifieddataset import (
    BatchedDataDataset,
    BatchedDataDatasetForUnifiedSampler,
    UnifiedPSMDataset,
)
from sfm.logging import logger
from sfm.models.psm.loss.mae3ddiff import DiffMAE3dCriterions
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psm_optimizer import DECAY_COSINE_RATE, WarmupDecayLR

try:
    from apex.optimizers import FusedAdam as AdamW
except:
    from torch.optim.adamw import AdamW

from sfm.models.psm.psm_optimizer import AdamFP16
from sfm.models.psm.psmmodel import PSMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer, seed_everything
from sfm.tasks.psm.ft_modules import PSM_FT_REGISTER
from sfm.utils import env_init
from sfm.utils.cli_utils import wandb_init


@dataclass
class Config(DistributedTrainConfig, PSMConfig):
    backbone_config: Dict[str, Any] = MISSING
    backbone: str = "graphormer"
    ode_mode: bool = False
    finetune_module: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="config_psm_schema", node=Config)


@hydra.main(
    version_base=None, config_path="../../../config_file", config_name="config_psm"
)
def main(args: DictConfig) -> None:
    args = OmegaConf.to_object(args)
    assert isinstance(
        args, Config
    ), f"args must be an instance of Config! But it is {type(args)}"

    wandb_init(args)
    seed_everything(args.seed)
    env_init.set_env(args)

    ### define psm dataset here
    dataset = UnifiedPSMDataset(
        args.data_path, args.data_path_list, args.dataset_name_list, args
    )
    train_data, valid_data = dataset.split_dataset()

    finetune_module = None
    extra_collate_fn = None
    if args.psm_finetune_mode:
        finetune_module = PSM_FT_REGISTER[args.finetune_module](args)
        extra_collate_fn = finetune_module.update_batched_data

    if args.ifstack:
        raise NotImplementedError("ifstack is not finished yet!")
        # train_data = StackedIterableDataset(train_data, args, dataset.sizes)
    elif args.use_unified_batch_sampler:
        train_data = BatchedDataDatasetForUnifiedSampler(
            args, train_data, dataset.train_len, extra_collate_fn=extra_collate_fn
        )
        valid_data = BatchedDataDatasetForUnifiedSampler(
            args, valid_data, dataset.valid_len, extra_collate_fn=extra_collate_fn
        )
    elif args.use_dali_pipeline:
        train_data = UnifiedBatchedIterableDataset(args, train_data, dataset.train_len)
        valid_data = BatchedDataDatasetForUnifiedSampler(
            args, valid_data, dataset.valid_len, extra_collate_fn=extra_collate_fn
        )
    else:
        train_data = BatchedDataDataset(
            args, train_data, dataset.train_len, extra_collate_fn=extra_collate_fn
        )
        valid_data = BatchedDataDataset(
            args, valid_data, dataset.valid_len, extra_collate_fn=extra_collate_fn
        )

    # define psm models here, define the diff loss in DiffMAE3dCriterions
    if args.rescale_loss_with_std:

        def loss_fn(args):
            return DiffMAE3dCriterions(
                args,
                dataset.molecule_energy_mean,
                dataset.molecule_energy_std,
                dataset.periodic_energy_mean,
                dataset.periodic_energy_std,
                dataset.molecule_energy_per_atom_mean,
                dataset.molecule_energy_per_atom_std,
                dataset.periodic_energy_per_atom_mean,
                dataset.periodic_energy_per_atom_std,
                dataset.molecule_force_mean,
                dataset.molecule_force_std,
                dataset.periodic_force_mean,
                dataset.periodic_force_std,
            )

    else:
        loss_fn = DiffMAE3dCriterions

    model = PSMModel(args, loss_fn, psm_finetune_head=finetune_module)
    # define optimizer here
    if args.fp16:
        optimizer = AdamFP16(
            model.parameters(),
            distributed_strategy=args.strategy,
            lr=args.max_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=args.max_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
    lr_scheduler = WarmupDecayLR(
        optimizer,
        total_num_steps=args.total_num_steps,
        warmup_max_lr=args.max_lr,
        warmup_num_steps=args.warmup_num_steps,
        decay_type=DECAY_COSINE_RATE,
    )

    trainer = Trainer(
        args,
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data=train_data,
        valid_data=valid_data,
    )
    if args.psm_validation_mode:
        trainer.validate()
    else:
        trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
    finally:
        wandb.finish()  # support to finish wandb logging
        logger.info("wandb finish logging!")
