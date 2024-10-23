# -*- coding: utf-8 -*-
"""This script is used to fine-tune the PSM model on small molecules dataset, such MD17, MD22, PubChem50K, etc."""
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

import hydra
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from sfm.data.psm_data.unifieddataset import (
    BatchedDataDataset,
    BatchedDataDatasetForUnifiedSampler,
    SmallMolDataset,
    UnifiedPSMDataset,
)
from sfm.logging import logger
from sfm.models.psm.equivariant.geomformer import EquivariantVectorOutput
from sfm.models.psm.loss.mae3ddiff import DiffMAE3dCriterions
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psm_optimizer import DECAY_COSINE_RATE, groupWarmupDecayLR, myAdam
from sfm.models.psm.psmmodel import PSMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig, ModelOutput
from sfm.pipeline.accelerator.trainer import Model, Trainer, seed_everything
from sfm.tasks.psm.ft_modules import PSM_FT_REGISTER, MDEnergyForceHead
from sfm.utils import env_init
from sfm.utils.cli_utils import cli, wandb_init

import wandb  # isort: skip


kcalmol_to_ev = 0.0433634


@dataclass
class SmallMolConfig(DistributedTrainConfig, PSMConfig):
    backbone_config: Dict[str, Any] = MISSING
    backbone: str = "graphormer"
    train_val_test_split: List[float] = field(
        default_factory=lambda: [0.2, 0.7, 0.1]
        # default_factory=lambda: [0.97, 0.03, 0.0]
    )  # NOTE: This is only for MD data
    shuffle: bool = True
    vsc_debug: bool = False
    energy_loss_weight: float = 0.01
    force_loss_weight: float = 0.99
    finetune_module: str = "md_energy_force_head"  # if "", skip reset ft head
    loss_unit: str = "ev"


cs = ConfigStore.instance()
cs.store(name="config_psm_schema", node=SmallMolConfig)


def load_data(args, extra_collate_fn=None):
    # Dataset will automatically load based on the args
    sub_data_path_list = args.data_path_list.split(",")
    dataset_name_list = args.dataset_name_list.split(",")
    file_list = []

    for sub_data_path in sub_data_path_list:
        file_list.append(os.path.join(args.data_path, sub_data_path))

    train_dataset_list = []
    valid_dataset_list = []
    test_dataset_list = []
    train_len = 0
    valid_len = 0
    test_len = 0
    for data_path, dataset_name in zip(file_list, dataset_name_list):
        dataset = SmallMolDataset(
            args,
            data_path,
            data_name=dataset_name,
        )
        shuffle = args.shuffle
        train_dataset, valid_dataset, test_dataset = dataset.split_train_valid_test(
            args.train_val_test_split, sort=False, shuffle=shuffle
        )

        train_dataset_list.append(train_dataset)
        valid_dataset_list.append(valid_dataset)
        test_dataset_list.append(test_dataset)

        train_len += len(train_dataset)
        valid_len += len(valid_dataset)
        test_len += len(test_dataset)

    BatchedDataset = (
        BatchedDataDatasetForUnifiedSampler
        if args.use_unified_batch_sampler
        else BatchedDataDataset
    )

    train_data = BatchedDataset(
        args, train_dataset_list, train_len, extra_collate_fn=extra_collate_fn
    )
    valid_data = BatchedDataset(
        args, valid_dataset_list, valid_len, extra_collate_fn=extra_collate_fn
    )
    test_data = BatchedDataset(
        args, test_dataset_list, test_len, extra_collate_fn=extra_collate_fn
    )

    return train_data, valid_data, test_data


# class PSMSmallMoleculeModel(Model):
#     def __init__(self, args, base):
#         """
#         Initialize the FinetunePSMSmallMol class.

#         Args:
#             args: The arguments passed to the class.
#             base: The foundation model used.

#         Attributes:
#             args: The arguments passed to the class.
#             base: The base object.
#             config: The PSMConfig object.
#             net: The neural network model.

#         """
#         super().__init__()

#         self.args = args
#         self.base = base
#         config = PSMConfig(args)

#         self.energy_head = nn.Sequential(
#             nn.Linear(
#                 config.embedding_dim,
#                 config.embedding_dim,
#                 bias=True,
#             ),
#             nn.SiLU(),
#             nn.Linear(config.embedding_dim, 1, bias=True),
#         )

#         self.force_head = EquivariantVectorOutput(config.embedding_dim)

#     def forward(self, batch_data):
#         base_output = self.base.forward(batch_data)
#         # print("**********************************base:", base_output)
#         # energy = self.energy_head(base_output)
#         # force = self.force_head(base_output)

#         result_dict = {
#             "energy": base_output["energy_per_atom"] * batch_data["num_atoms"],
#             "forces": base_output["forces"],
#         }

#         # return result_dict
#         return result_dict

#     def compute_loss(self, model_output, batch_data):
#         """Compute the MAE loss for the energy and forces."""
#         e_pred = model_output["energy"]
#         e_true = batch_data["energy"]
#         size = e_true.shape[0]

#         f_pred = model_output["forces"]
#         f_true = batch_data["forces"]

#         e_loss = torch.mean(torch.abs(e_pred - e_true))

#         f_loss = torch.mean(torch.abs(f_pred - f_true))

#         if self.args.loss_unit == "kcal/mol":
#             e_loss /= kcalmol_to_ev
#             f_loss /= kcalmol_to_ev

#         loss = (
#             self.args.energy_loss_weight * e_loss + self.args.force_loss_weight * f_loss
#         )
#         log_output = {
#             "loss": loss,
#             "energy_loss": (e_loss, size),
#             "force_loss": (f_loss, size),
#         }
#         return ModelOutput(loss=loss, num_examples=size, log_output=log_output)

#     def config_optimizer(self):
#         return (None, None)


@hydra.main(
    version_base="1.3", config_path="../../../config_file", config_name="config_psm"
)
def finetune(cfg: DictConfig) -> None:
    args = OmegaConf.to_object(cfg)
    assert isinstance(args, SmallMolConfig)
    assert args.clean_sample_ratio >= 1, "clean_sample_ratio should be set to 1.0"

    if args.vsc_debug:
        import debugpy

        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached!")

    wandb_init(args)
    seed_everything(args.seed)
    env_init.set_env(args)

    finetune_module = None
    extra_collate_fn = None
    if len(args.finetune_module) > 0:
        if args.psm_finetune_mode and args.finetune_module is not None:
            finetune_module = PSM_FT_REGISTER[args.finetune_module](args)
            extra_collate_fn = finetune_module.update_batched_data

    train_data, valid_data, test_data = load_data(
        args, extra_collate_fn=extra_collate_fn
    )

    # Define model
    model = PSMModel(
        args, loss_fn=DiffMAE3dCriterions, psm_finetune_head=finetune_module
    )
    # base = PSMModel(args, loss_fn=DiffMAE3dCriterions)
    # model = PSMSmallMoleculeModel(args, base)

    # Define optimizer
    optimizer = myAdam(
        model,
        lr=args.max_lr,
        betas=[0.9, 0.999],
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    lr_scheduler = groupWarmupDecayLR(
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
        test_data=test_data,
    )

    trainer.train()


if __name__ == "__main__":
    try:
        finetune()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
    finally:
        wandb.finish()
