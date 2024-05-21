# -*- coding: utf-8 -*-
import os
import sys
from dataclasses import asdict, dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

import torch

from sfm.data.psm_data.dataset import PCQM4Mv2LMDBDataset
from sfm.data.psm_data.unifieddataset import BatchedDataDataset
from sfm.logging import logger
from sfm.models.psm.loss.mae3ddiff import DiffMAE3dCriterions
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psm_optimizer import DECAY_COSINE_RATE, groupWarmupDecayLR, myAdam
from sfm.models.psm.psmmodel import PSMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig, ModelOutput
from sfm.pipeline.accelerator.trainer import Model, Trainer
from sfm.utils.cli_utils import cli

import wandb  # isort: skip


@dataclass
class PSMFTConfig:
    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(args)
        for k, v in asdict(self).items():
            if hasattr(args, k):
                setattr(self, k, getattr(args, k))


def load_data(args):
    if args.dataset_names == "pcqm4mv2":
        dataset = PCQM4Mv2LMDBDataset(args, args.data_path)
        train_data, valid_data = dataset.split_dataset()
    else:
        raise ValueError("invalid dataset name")

    train_data = BatchedDataDataset(args, [train_data], len(train_data))
    valid_data = BatchedDataDataset(args, [valid_data], len(valid_data))

    return train_data, valid_data


class PSMFTModel(Model):
    def __init__(self, args, base):
        super().__init__()

        self.args = args
        self.base = base
        config = PSMConfig(args)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(config.embedding_dim, config.embedding_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(config.embedding_dim, 1, bias=True),
        )

    def forward(self, batch_data):
        output = self.net(self.base.ft_forward(batch_data))  # (B, L, 1)
        output = output.squeeze(-1).sum(dim=-1)

        result_dict = {
            "homo_lumo_gap": output,
        }

        return result_dict

    def compute_loss(self, model_output, batch_data):
        y_pred = model_output["homo_lumo_gap"]
        y_true = batch_data["energy"]

        loss = torch.nn.L1Loss()(y_pred, y_true)
        size = y_true.shape[0]

        return ModelOutput(loss=loss, num_examples=size)

    def config_optimizer(self):
        return (None, None)


@cli(DistributedTrainConfig, PSMConfig, PSMFTConfig)
def finetune(args):
    train_data, valid_data = load_data(args)

    # define model
    base = PSMModel(args, load_ckpt=True, loss_fn=DiffMAE3dCriterions)
    model = PSMFTModel(args, base)

    # define optimizer
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
    )

    trainer.train()


if __name__ == "__main__":
    try:
        finetune()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
    finally:
        wandb.finish()
