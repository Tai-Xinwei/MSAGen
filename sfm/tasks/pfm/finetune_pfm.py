# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from sfm.criterions.mae3d import ProteinPMLM
from sfm.data.prot_data.dataset import BatchedDataDataset, DownstreamLMDBDataset
from sfm.logging import logger, metric_logger
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.models.pfm.pfm_optimizer import DECAY_COSINE_RATE, groupWarmupDecayLR, myAdam
from sfm.models.pfm.pfmmodel import PFMModel
from sfm.pipeline.accelerator.dataclasses import (
    DistributedTrainConfig,
    ModelOutput,
    ValidLogOutput,
)
from sfm.pipeline.accelerator.trainer import Model, Trainer
from sfm.utils.cli_utils import cli
from sfm.utils.move_to_device import move_to_device


@dataclass
class DownstreamConfig:
    task_name: str
    data_basepath: str
    head_dropout: float = 0.1


class SingleSequenceModel(Model):
    def __init__(self, args, model, n_classes):
        super().__init__()
        self.args = args
        self.model = model
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(args.head_dropout),
            torch.nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(args.head_dropout),
            torch.nn.Linear(args.encoder_embed_dim, n_classes),
        )

    def forward(self, batch_data):
        # x: (B, L, C)
        x = self.model.ft_forward(batch_data)
        x = x[:, 0, :].squeeze(1)
        logits = self.head(x)
        return logits

    def load_pretrained_weights(self, args, pretrained_model_path):
        self.model.load_pretrained_weights(args, pretrained_model_path)

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        seq_aa = batch_data["x"]
        if DownstreamLMDBDataset.TASKINFO[self.args.task_name]["type"] == "regression":
            mean, std = DownstreamLMDBDataset.TASKINFO[self.args.task_name]["mean_std"]
            target = (batch_data["target"].unsqueeze(1) - mean) / std
            lossfn = torch.nn.MSELoss()
        elif DownstreamLMDBDataset.TASKINFO[self.args.task_name]["type"] == "binary":
            target = batch_data["target"].unsqueeze(1)
            lossfn = torch.nn.BCEWithLogitsLoss()
        elif (
            DownstreamLMDBDataset.TASKINFO[self.args.task_name]["type"]
            == "classification"
        ):
            target = batch_data["target"].unsqueeze(1)
            lossfn = torch.nn.CrossEntropyLoss()
        elif (
            DownstreamLMDBDataset.TASKINFO[self.args.task_name]["type"]
            == "multi_classification"
        ):
            target = batch_data["target"].unsqueeze(1)
            lossfn = torch.nn.BCEWithLogitsLoss()

        # B x n_classes
        logits = model_output
        bs = seq_aa.shape[0]
        loss = lossfn(logits.to(torch.float32), target)
        return ModelOutput(loss=loss, num_examples=bs)

    def config_optimizer(self):
        # Why can I just pass, like the way in PFMModel?
        # pass
        optimizer, _ = myAdam(
            self,
            lr=self.args.max_lr,
            betas=[0.9, 0.999],
            weight_decay=self.args.weight_decay,
            eps=1e-8,
        )

        # may be here we should use a grouped lr scheduler
        # eg: set the lr of the head to be 10x of the pretrained model
        # cosine scheduler!!!
        lr_scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=self.args.total_num_steps,
            warmup_max_lr=self.args.max_lr,
            warmup_num_steps=self.args.warmup_num_steps,
            d_tilde=0.1,  # this is the ratio of the lr of the encoder to the head
            decay_type=DECAY_COSINE_RATE,
        )
        return optimizer, lr_scheduler


def load_batched_dataset(args):
    # required args:
    # args.task_name, args.data_basepath
    dataset_dict = DownstreamLMDBDataset.load_dataset(args)
    trainset = dataset_dict["train"]
    valset = dataset_dict["valid"]
    # others are test sets
    testset_dict = {
        k: v for k, v in dataset_dict.items() if k not in ["train", "valid"]
    }

    train_data = BatchedDataDataset(
        trainset,
        args=args,
        vocab=trainset.vocab,
    )
    val_data = BatchedDataDataset(
        valset,
        args=args,
        vocab=trainset.vocab,
    )
    testset_dict = {
        k: BatchedDataDataset(
            v,
            args=args,
            vocab=trainset.vocab,
        )
        for k, v in testset_dict.items()
    }
    logger.info(f"Got test dataset: {testset_dict.keys()}")
    return train_data, val_data, testset_dict


def init_model(args):
    # seems model loading require this parameter
    args.ft = True
    basemodel = PFMModel(args, loss_fn=ProteinPMLM)

    if DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "regression":
        model = SingleSequenceModel(args, basemodel, n_classes=1)
    elif DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "binary":
        model = SingleSequenceModel(args, basemodel, n_classes=1)
    elif DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "classification":
        n_classes = len(DownstreamLMDBDataset.TASKINFO[args.task_name]["classes"])
        model = SingleSequenceModel(args, basemodel, n_classes=n_classes)
    elif (
        DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "multi_classification"
    ):
        n_classes = len(DownstreamLMDBDataset.TASKINFO[args.task_name]["classes"])
        model = SingleSequenceModel(args, basemodel, n_classes=n_classes)
    else:
        raise NotImplementedError()
    return model


@cli(DistributedTrainConfig, PFMConfig, DownstreamConfig)
def finetune(args) -> None:
    train_data, val_data, testset_dict = load_batched_dataset(args)

    model = init_model(args)
    logger.info(f"Finetuning on task {args.task_name}")

    # any important settings to keep in mind?
    trainer = Trainer(
        args,
        model,
        train_data=train_data,
        valid_data=val_data,
    )
    trainer.train()


if __name__ == "__main__":
    finetune()
