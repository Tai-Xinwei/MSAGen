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

from sfm.data.psm_data.dataset import ComplexDataset
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

import torch

from sfm.models.psm.complexmodel import ComplexModel
from sfm.models.psm.psm_optimizer import AdamFP16
from sfm.models.psm.psmmodel import PSMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig, ModelOutput
from sfm.pipeline.accelerator.trainer import Model, Trainer, seed_everything
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


class ComplexMAE3dCriterions(DiffMAE3dCriterions):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def forward(self, model_output, batched_data):
        # noise_pred = model_output["noise_pred"]
        # noise_label = model_output["noise"]
        # protein_len = batched_data["protein_len"]
        # num_atoms = batched_data["num_atoms"]
        # ligand_mask = torch.zeros_like(noise_label)
        # for i in range(ligand_mask.size()[0]):
        #     ligand_mask[i, protein_len[i] : num_atoms[i]] = 1

        # noise_pred = noise_pred * ligand_mask
        # noise_label = noise_label * ligand_mask
        # model_output["noise_pred"] = noise_pred
        # model_output["noise"] = noise_label
        # model_output["clean_mask"] = torch.ones(
        #     noise_label.size()[0],
        #     noise_label.size()[1],
        #     dtype=torch.bool,
        #     device=noise_label.device,
        # )
        # for i in range(noise_label.size()[0]):
        #     model_output["clean_mask"][i, protein_len[i] : num_atoms[i]] = 0
        return super().forward(model_output, batched_data)


class PSMComplexModel(Model):
    def __init__(self, args, base):
        super().__init__()

        self.args = args
        self.base = base
        PSMConfig(args)
        self.loss = ComplexMAE3dCriterions(args)

    def forward(self, batch_data):
        return self.base(batch_data)

    def compute_loss(self, model_output, batch_data):
        model_output["noise_pred"]
        y_true = model_output["noise"]
        size = y_true.shape[0]

        loss, output = self.loss(model_output, batch_data)

        return ModelOutput(loss=loss, num_examples=size, log_output=output)

    def config_optimizer(self):
        return (None, None)


def load_data(args):
    dataset = ComplexDataset(args, args.data_path)
    train_data, valid_data = dataset.split_dataset()
    return train_data, valid_data


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
    train_data, valid_data = load_data(args)
    logger.info(f"train_data: {len(train_data)}")
    logger.info(f"valid_data: {len(valid_data)}")

    # if args.psm_finetune_mode:
    #     PSM_FT_REGISTER[args.finetune_module](args)

    base = ComplexModel(args, loss_fn=ComplexMAE3dCriterions, psm_finetune_head=None)
    model = PSMComplexModel(args, base)
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
