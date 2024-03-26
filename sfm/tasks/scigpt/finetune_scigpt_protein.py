# -*- coding: utf-8 -*-

from dataclasses import dataclass

import torch

from sfm.data.prot_data.dataset import DownstreamLMDBDataset
from sfm.data.sci_data.dataset import ProteinLmdbDataset
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from sfm.logging import logger
from sfm.models.scigpt.config import (
    ScigptConfig,
    scigpt_7b_config,
    scigpt_13b_config,
    scigpt_350m_config,
    scigpt_shallow_config,
    scigpt_tiny_config,
)
from sfm.models.scigpt.scigpt import ScigptModel
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli

config_registry = {
    "scigpt_tiny": scigpt_tiny_config,
    "scigpt_shallow": scigpt_shallow_config,
    "scigpt_350m": scigpt_350m_config,
    "scigpt": scigpt_shallow_config,
    "scigpt_7b": scigpt_7b_config,
    "scigpt_13b": scigpt_13b_config,
}


@dataclass
class ProtFinetuneConfig(ScigptConfig):
    task_name: str = ""
    data_basepath: str = ""
    max_length: int = 2048


@cli(ProtFinetuneConfig)
def main(args):
    tokenizer = SFMDecTokenizer.from_pretrained(
        args.dict_path,
        prot_spm_path=args.prot_spm_path,
        dna_spm_path=args.dna_spm_path,
    )
    args.vocab_size = len(tokenizer)
    args.pad_token_id = tokenizer.pad_token_id

    config = arg_utils.from_args(args, ProtFinetuneConfig)
    config = config_registry.get(config.model_type, scigpt_tiny_config)(config)

    logger.info(f"config: {config}")

    model = ScigptModel(config)

    dataset_dict = DownstreamLMDBDataset.load_dataset(args)
    trainset = dataset_dict["train"]
    valset = dataset_dict["valid"]

    train_dataset = ProteinLmdbDataset(
        task_name=config.task_name,
        lmdb_dataset=trainset,
        lmdb_vocab=trainset.vocab,
        tokenizer=tokenizer,
    )

    valid_dataset = ProteinLmdbDataset(
        task_name=config.task_name,
        lmdb_dataset=valset,
        lmdb_vocab=valset.vocab,
        tokenizer=tokenizer,
    )

    train_data_len = len(train_dataset)
    total_steps = (train_data_len // config.train_batch_size) * config.total_num_epochs
    config.total_num_steps = total_steps
    config.warmup_num_steps = int(total_steps * 0.1)
    logger.info(f"train data len: {train_data_len}")
    logger.info(f"override total steps to {total_steps}")
    logger.info(f"override warmup steps to {config.warmup_num_steps}")

    trainer = Trainer(
        config,
        model=model,
        train_data=train_dataset,
        valid_data=valid_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
