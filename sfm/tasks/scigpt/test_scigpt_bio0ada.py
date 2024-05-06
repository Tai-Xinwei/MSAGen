# -*- coding: utf-8 -*-
from collections import namedtuple

import torch

from sfm.data.sci_data.dataset import ProcessedSciDataset
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
from sfm.models.scigpt.scigptbio0ada import Scigptbio0adaModel
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

SciTokenIdxAndMask = namedtuple("SciTokenIdxAndMask", ["input_ids", "padding_mask"])
SciDataTuple = namedtuple("SciDataTuple", ["input", "labels"])


class inferdataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer):
        with open(args.train_data_path, "r") as f:
            data = [line.strip() for line in f.readlines()]
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.data[idx], return_tensors="pt")
        input_ids = encodings.input_ids
        return input_ids

    def collate(self, samples):
        input_ids = torch.stack(samples, dim=0)[0]
        padding_mask = input_ids.ne(self.tokenizer.pad_token_id)
        input = SciTokenIdxAndMask(input_ids, padding_mask)
        labels = input
        return SciDataTuple(input, labels)


@cli(ScigptConfig)
def main(args) -> None:
    assert (
        args.train_data_path is not None and len(args.train_data_path) > 0
    ), f"train_dataset is {args.train_data_path} it should not be None or empty"

    assert (
        args.valid_data_path is not None and len(args.valid_data_path) > 0
    ), f"valid_dataset is {args.valid_data_path} it should not be None or empty"

    tokenizer = SFMDecTokenizer.from_pretrained(
        args.dict_path,
        prot_spm_path="/data/peiran/blob/msralaphilly2/ml-la/shufxi/data/scigpt/ur50bpe/bpe",
        dna_spm_path="/data/peiran/blob/msralaphilly2/ml-la/shufxi/data/scigpt/dnabpe/bpe",
        rna_spm_path="/data/peiran/blob/msralaphilly2/ml-la/shufxi/data/scigpt/rnabpe/bpe",
    )
    args.vocab_size = len(tokenizer)  # now we have new tokens
    args.pad_token_id = tokenizer.pad_token_id

    config = arg_utils.from_args(args, ScigptConfig)
    config = config_registry.get(config.model_type, scigpt_tiny_config)(config)

    logger.info(f"config: {config}")

    model = Scigptbio0adaModel(config)

    train_dataset = inferdataset(args, tokenizer)
    valid_dataset = train_dataset

    trainer = Trainer(
        config,
        model=model,
        train_data=train_dataset,
        valid_data=valid_dataset,
    )
    trainer.validate()


if __name__ == "__main__":
    main()
