# -*- coding: utf-8 -*-
import logging
import os
import sys

import deepspeed
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mol_data.dataset import BatchedDataDataset, PCQPreprocessedData

# import torch.distributed as dist
from deepspeed import comm as dist
from pipeline.graphormer_pretrainer import DiffTrainer, Trainer
from utils.add_argument import add_argument

logging.getLogger().setLevel(logging.ERROR)
# from graphormer.pipeline.trainer_pp import Trainer_pp


def main() -> None:
    torch.set_flush_denormal(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    args = add_argument()

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)
    # os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    # os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ['LOCAL_RANK']
    os.environ["NCCL_BLOCKING_WAIT"] = "0"

    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed()

    print(
        "Print os.environ:--- RANK: {}, WORLD_SIZE: {}, LOCAL_RANK: {}".format(
            os.environ["RANK"], os.environ["WORLD_SIZE"], os.environ["LOCAL_RANK"]
        )
    )

    dataset = PCQPreprocessedData(
        args, dataset_name=args.dataset_name, dataset_path=args.data_path
    )

    trainset = dataset.dataset_train
    valset = dataset.dataset_val

    train_data = BatchedDataDataset(
        trainset,
        dataset_version="2D" if dataset.dataset_name == "PCQM4M-LSC-V2" else "3D",
        max_node=dataset.max_node,
        max_node2=dataset.max_node2,
        multi_hop_max_dist=dataset.multi_hop_max_dist,
        spatial_pos_max=dataset.spatial_pos_max,
        args=args,
    )

    BatchedDataDataset(
        valset,
        dataset_version="2D" if dataset.dataset_name == "PCQM4M-LSC-V2" else "3D",
        max_node=dataset.max_node,
        max_node2=dataset.max_node2,
        multi_hop_max_dist=dataset.multi_hop_max_dist,
        spatial_pos_max=dataset.spatial_pos_max,
        args=args,
    )

    print("add-3d", args.add_3d, "no-2d", args.no_2d)
    # trainer = Trainer(args, train_data)
    trainer = DiffTrainer(args, train_data)
    trainer()


if __name__ == "__main__":
    main()
