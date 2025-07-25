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

from sfm.data.psm_data.unifieddataset import (
    BatchedDataDataset,
    BatchedDataDatasetForUnifiedSampler,
    UnifiedPSMDataset,
)
from sfm.logging import logger
from sfm.models.psm.loss.mae3ddiff import DiffMAE3dCriterions, DiffProteaCriterions
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psm_optimizer import (
    DECAY_COSINE_RATE,
    WarmupDecayLR,
    groupWarmupDecayLR,
    myAdam,
)

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
    if args.psm_finetune_mode and args.finetune_module:
        finetune_module = PSM_FT_REGISTER[args.finetune_module](args)
        extra_collate_fn = finetune_module.update_batched_data

    if args.ifstack:
        raise NotImplementedError("ifstack is not finished yet!")
        # train_data = StackedIterableDataset(train_data, args, dataset.sizes)
        # valid_data = BatchedDataDataset(
        #     args, valid_data, dataset.valid_len, extra_collate_fn=extra_collate_fn
        # )
    elif args.use_unified_batch_sampler:
        train_data = BatchedDataDatasetForUnifiedSampler(
            args, train_data, dataset.train_len, extra_collate_fn=extra_collate_fn
        )
        valid_data = BatchedDataDatasetForUnifiedSampler(
            args, valid_data, dataset.valid_len, extra_collate_fn=extra_collate_fn
        )
    elif args.use_dali_pipeline:
        from sfm.data.psm_data.pipeline import UnifiedBatchedIterableDataset

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
        molecule_energy_per_atom_std = dataset.molecule_energy_per_atom_std
        if hasattr(args, "molecule_energy_per_atom_std_override"):
            # A special fix for models where molecule_energy_per_atom_std was overrided
            # Should not be used unless you know what you are doing
            logger.warning(
                "=== N O T E === dataset.molecule_energy_per_atom_std={} is overrided by {}",
                molecule_energy_per_atom_std,
                args.molecule_energy_per_atom_std_override,
            )
            molecule_energy_per_atom_std = args.molecule_energy_per_atom_std_override

        def loss_fn(args):
            if args.diffusion_mode == "protea":
                return DiffProteaCriterions(
                    args,
                    dataset.molecule_energy_mean,
                    dataset.molecule_energy_std,
                    dataset.periodic_energy_mean,
                    dataset.periodic_energy_std,
                    dataset.molecule_energy_per_atom_mean,
                    molecule_energy_per_atom_std,
                    dataset.periodic_energy_per_atom_mean,
                    dataset.periodic_energy_per_atom_std,
                    dataset.molecule_force_mean,
                    dataset.molecule_force_std,
                    dataset.periodic_force_mean,
                    dataset.periodic_force_std,
                )
            else:
                return DiffMAE3dCriterions(
                    args,
                    dataset.molecule_energy_mean,
                    dataset.molecule_energy_std,
                    dataset.periodic_energy_mean,
                    dataset.periodic_energy_std,
                    dataset.molecule_energy_per_atom_mean,
                    molecule_energy_per_atom_std,
                    dataset.periodic_energy_per_atom_mean,
                    dataset.periodic_energy_per_atom_std,
                    dataset.molecule_force_mean,
                    dataset.molecule_force_std,
                    dataset.periodic_force_mean,
                    dataset.periodic_force_std,
                    dataset.periodic_stress_mean,
                    dataset.periodic_stress_std,
                )

    else:
        if args.diffusion_mode == "protea":
            loss_fn = DiffProteaCriterions
        else:
            loss_fn = DiffMAE3dCriterions

    model = PSMModel(
        args,
        loss_fn,
        psm_finetune_head=finetune_module,
        molecule_energy_per_atom_std=dataset.molecule_energy_per_atom_std,
        periodic_energy_per_atom_std=dataset.periodic_energy_per_atom_std,
        molecule_force_std=dataset.molecule_force_std,
        periodic_force_std=dataset.periodic_force_std,
        periodic_stress_mean=dataset.periodic_stress_mean,
        periodic_stress_std=dataset.periodic_stress_std,
    )
    # define optimizer here
    if args.group_optimizer:
        optimizer = myAdam(
            model,
            lr=args.max_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
    elif args.fp16:
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

    if args.group_optimizer:
        lr_scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=args.total_num_steps,
            warmup_max_lr=args.max_lr,
            warmup_num_steps=args.warmup_num_steps,
            decay_type=DECAY_COSINE_RATE,
            d_tilde=args.group_lr_ratio,
        )
    else:
        lr_scheduler = WarmupDecayLR(
            optimizer,
            total_num_steps=args.total_num_steps,
            warmup_max_lr=args.max_lr,
            warmup_num_steps=args.warmup_num_steps,
            decay_type=DECAY_COSINE_RATE,
        )

    if args.psm_validate_for_train_set and args.psm_validation_mode:
        valid_data = train_data

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
        wandb.finish()  # support to finish wandb logging
        logger.info("KeyboardInterrupt!")
    finally:
        wandb.finish()  # support to finish wandb logging
        logger.info("wandb finish logging!")
