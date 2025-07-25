# -*- coding: utf-8 -*-
import os

# import ogb
import sys

import deepspeed
import torch

# import pytorch_forecasting
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

import subprocess
from argparse import ArgumentParser

# subprocess.check_call([sys.executable, "-m", "pip", "install", "PyTDC"])
from tdc.benchmark_group import admet_group

from sfm.criterions.l1ft import L1Criterions
from sfm.data.mol_data.dataset import BatchedDataDataset
from sfm.data.mol_data.tdc import TDCDataset
from sfm.logging import logger
from sfm.models.graphormer.graphormer import GraphormerModel
from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli

# from sfm.pipeline.graphormer.graphormer_fter_bk import Finetuner
from sfm.utils.optim.optimizer import myAdam
from sfm.utils.optim.set_lr import groupWarmupDecayLR


@cli(DistributedTrainConfig, GraphormerConfig)
def main(args):
    # Define dataset
    group = admet_group(path=args.data_path)
    benchmark = group.get("Caco2_Wang")

    # benchmark = group.get('Lipophilicity_AstraZeneca')
    # benchmark = group.get('LD50_Zhu')
    # benchmark = group.get('Solubility_AqSolDB')
    # benchmark = group.get('PPBR_AZ')

    # benchmark = group.get('Bioavailability_Ma')

    # all benchmark names in a benchmark group are stored in group.dataset_names
    name = benchmark["name"]
    train_val, test = benchmark["train_val"], benchmark["test"]
    train, valid = group.get_train_valid_split(
        benchmark=name, split_type="default", seed=6
    )

    dataset = TDCDataset(train_val)
    dataset_name = dataset.dataset_name
    data_mean = dataset.mean
    data_std = dataset.std

    trainset = TDCDataset(train)
    valset = TDCDataset(valid)
    testset = TDCDataset(test)

    # dataset_name = 'PCQM4M-LSC-V2'
    # dataset = PCQPreprocessedData(args, dataset_name, dataset_path = args.data_path)
    # trainset = dataset.dataset_train
    # valset = dataset.dataset_val
    # testset = dataset.dataset_test
    # data_mean = 0.0
    # data_std = 1.0

    logger.info(
        f"length of trainset {len(trainset)}, length of validset {len(valset)}, length of testset {len(testset)}"
    )
    logger.info(f"std is {data_std}, , mean is {data_mean}")

    max_node = 512
    train_data = BatchedDataDataset(
        trainset,
        dataset_version="3D" if dataset_name == "PCQM4M-LSC-V2-3D" else "2D",
        max_node=max_node,
        args=args,
        ft=True,
    )

    valid_data = BatchedDataDataset(
        valset,
        dataset_version="3D" if dataset_name == "PCQM4M-LSC-V2-2D" else "2D",
        max_node=max_node,
        args=args,
        ft=True,
    )

    test_data = BatchedDataDataset(
        testset,
        dataset_version="3D" if dataset_name == "PCQM4M-LSC-V2-2D" else "2D",
        max_node=max_node,
        args=args,
        ft=True,
    )

    args.ft = True
    args.add_3d = False

    model = GraphormerModel(
        args, loss_fn=L1Criterions, data_mean=data_mean, data_std=data_std
    )

    optimizer, _ = myAdam(
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
    )

    logger.info(
        f"finetune: {args.ft}, add_3d: {args.add_3d}, infer: {args.infer}, no_2d: {args.no_2d}"
    )

    trainer = Trainer(
        args,
        model,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    main()
