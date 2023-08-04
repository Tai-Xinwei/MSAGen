# -*- coding: utf-8 -*-
import os
import sys

import deepspeed
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])
from argparse import ArgumentParser

from sfm.criterions.mae3ddiff import DiffMAE3dCriterions
from sfm.data.mol_data.dataset import BatchedDataDataset, PCQPreprocessedData
from sfm.logging import logger
from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.models.graphormer.graphormerdiff import GraphormerDiffModel
from sfm.pipeline.accelerator.dataclasses import DistributedConfig, TrainerConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils import arg_utils
from sfm.utils.optimizer import myAdam
from sfm.utils.set_lr import groupWarmupDecayLR


def main() -> None:
    parser = ArgumentParser()
    parser = arg_utils.add_dataclass_to_parser(
        [TrainerConfig, DistributedConfig, GraphormerConfig], parser
    )
    args = parser.parse_args()

    ## Init distributed
    torch.set_flush_denormal(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    os.environ["NCCL_BLOCKING_WAIT"] = "0"
    torch.cuda.set_device(args.local_rank)

    logger.success(
        "Print os.environ:--- RANK: {}, WORLD_SIZE: {}, LOCAL_RANK: {}".format(
            os.environ["RANK"], os.environ["WORLD_SIZE"], os.environ["LOCAL_RANK"]
        )
    )

    dataset = PCQPreprocessedData(
        args, dataset_name=args.dataset_names, dataset_path=args.data_path
    )

    trainset = dataset.dataset_train

    train_data = BatchedDataDataset(
        trainset,
        dataset_version="2D" if dataset.dataset_name == "PCQM4M-LSC-V2" else "3D",
        min_node=dataset.max_node,
        max_node=dataset.max_node2,
        multi_hop_max_dist=dataset.multi_hop_max_dist,
        spatial_pos_max=dataset.spatial_pos_max,
        args=args,
    )

    # valset = dataset.dataset_val

    # valid_data = BatchedDataDataset(
    #     valset,
    #     dataset_version="2D" if dataset.dataset_name == "PCQM4M-LSC-V2" else "3D",
    #     min_node=dataset.max_node,
    #     max_node=dataset.max_node2,
    #     multi_hop_max_dist=dataset.multi_hop_max_dist,
    #     spatial_pos_max=dataset.spatial_pos_max,
    #     args=args,
    # )

    model = GraphormerDiffModel(args, loss_fn=DiffMAE3dCriterions)

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
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()

    # trainer = Trainer(args, train_data)
    # trainer = DiffTrainer(args, train_data)
    # trainer()


if __name__ == "__main__":
    main()
