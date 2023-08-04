# -*- coding: utf-8 -*-
import os
import sys
from typing import Dict

import deepspeed
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from argparse import ArgumentParser

from transformers import AutoTokenizer

from sfm.data.mol_data.moltext_dataset import SupervisedProcessedDataWithSmiles
from sfm.logging import logger
from sfm.models.generalist.generalist_config import GeneralistConfig
from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.pipeline.accelerator.dataclasses import (
    DistributedConfig,
    TrainerConfig,
    TrainStrategy,
)
from sfm.pipeline.generalist.graphormerllama_trainer import Trainer
from sfm.utils import arg_utils
from sfm.utils.chemical_tokens import CHEMICAL_TOKENS

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def make_supervised_data_module(args, mode="train") -> Dict:
    assert mode in [
        "train",
        "eval",
        "test",
    ], f"Invalid mode: {mode}, must be train, eval, or test."

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # special_tokens_dict["additional_special_tokens"] = CHEMICAL_TOKENS
    tokenizer.add_special_tokens(special_tokens_dict)

    """ Make dataset and collator for supervised fine-tuning. """
    dataset = SupervisedProcessedDataWithSmiles(
        data_path=args.data_path,
        dataset_names=args.dataset_names,
        dataset_splits=args.dataset_splits,
        in_memory=False,
        model_max_length=args.model_max_length,
        mol_embed_type="atoms",
        molecule_max_size=512,
        pad_token_id=tokenizer.pad_token_id,
        # dataset_ratios=args.dataset_ratios,
        pool_mode=args.pool_mode,
        embedding_length=args.embedding_length,
    )

    return dict(train_dataset=dataset, eval_dataset=None, vocab_size=len(tokenizer))


def main() -> None:
    # init args
    parser = ArgumentParser()
    parser = arg_utils.add_dataclass_to_parser(
        [TrainerConfig, DistributedConfig, GraphormerConfig, GeneralistConfig], parser
    )
    args = parser.parse_args()

    ## Init distributed
    torch.set_flush_denormal(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)
    os.environ["NCCL_BLOCKING_WAIT"] = "0"

    torch.cuda.set_device(args.local_rank)
    if args.strategy == TrainStrategy.DDP:
        torch.distributed.init_process_group(backend="nccl")
    elif args.strategy in [
        TrainStrategy.Zero1,
        TrainStrategy.Zero2,
        TrainStrategy.Zero3,
    ]:
        deepspeed.init_distributed()

    logger.success(
        "Print os.environ:--- RANK: {}, WORLD_SIZE: {}, LOCAL_RANK: {}".format(
            os.environ["RANK"], os.environ["WORLD_SIZE"], os.environ["LOCAL_RANK"]
        )
    )

    args.add_3d = False

    logger.info(
        {
            "add-3d": args.add_3d,
            "no-2d": args.no_2d,
            "mfm_lora": args.mfm_lora,
            "pool_mode": args.pool_mode,
            "embedding_length": args.embedding_length,
            "btn_adaptor": args.btn_adaptor,
        }
    )

    data_module = make_supervised_data_module(args, mode="train")
    logger.info("length of dataset", len(data_module["train_dataset"]))

    freeze_list = []
    unfreeze_list = ["adaptor", "dummy", "0.layers.22", "0.layers.23"]

    trainer = Trainer(
        args,
        data_module["train_dataset"],
        vocab_size=data_module["vocab_size"],
        freeze_list=freeze_list,
        unfreeze_list=unfreeze_list,
    )

    if args.pipeline_parallelism == 0:
        trainer.train()
    else:
        trainer.train_pipeline()


if __name__ == "__main__":
    main()
