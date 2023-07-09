import os

# import ogb
import sys

import deepspeed
import numpy as np
import torch
import torch.nn as nn

# import pytorch_forecasting


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
from functools import lru_cache
from pathlib import Path

from data.mol_data.dataset import BatchedDataDataset, PCQPreprocessedData

subprocess.check_call([sys.executable, "-m", "pip", "install", "PyTDC"])

import logging

from tdc.benchmark_group import admet_group

from data.mol_data.tdc import TDCDataset
from pipeline.graphormer_fter import Finetuner
from utils.add_argument import add_argument


def main():
    torch.set_flush_denormal(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    args = add_argument()

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)
    os.environ["NCCL_BLOCKING_WAIT"] = "0"

    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed()

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

    print(len(trainset), len(valset), len(testset))
    print("std", data_std, ", mean", data_mean)

    max_node = 512
    train_data = BatchedDataDataset(
        trainset,
        dataset_version="3D" if dataset_name == "PCQM4M-LSC-V2-3D" else "2D",
        max_node=max_node,
        args=args,
        ft=True,
    )

    val_data = BatchedDataDataset(
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
    if args.pipeline_parallelism == 0:
        fter_pp = Finetuner(
            args,
            train_data,
            val_data=val_data,
            test_data=test_data,
            data_mean=data_mean,
            data_std=data_std,
        )
        fter_pp(iftrain=True)


if __name__ == "__main__":
    main()
