# -*- coding: utf-8 -*-

from dataclasses import dataclass

import torch

from sfm.data.prot_data.dataset import DownstreamLMDBDataset
from sfm.data.sci_data.dataset import RawTextSciDataset
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
class InstructionConfig(ScigptConfig):
    conditional_generation: bool = False
    use_template: bool = False
    max_length: int = 1024


@cli(InstructionConfig)
def main(args):
    tokenizer = SFMDecTokenizer.from_pretrained(
        args.dict_path,
        prot_spm_path=args.prot_spm_path,
        dna_spm_path=args.dna_spm_path,
        rna_spm_path=args.rna_spm_path,
    )
    args.vocab_size = len(tokenizer)
    args.pad_token_id = tokenizer.pad_token_id

    config = arg_utils.from_args(args, InstructionConfig)
    config = config_registry.get(config.model_type, scigpt_tiny_config)(config)
    config.ft = True

    logger.info(f"config: {config}")

    model = ScigptModel(config)

    train_dataset = RawTextSciDataset(
        path=config.train_data_path,
        tokenizer=tokenizer,
        conditional_generation=config.conditional_generation,
        use_template=config.use_template,
        max_len=config.max_length,
    )

    if config.valid_data_path:
        valid_dataset = RawTextSciDataset(
            path=config.valid_data_path,
            tokenizer=tokenizer,
            conditional_generation=config.conditional_generation,
            use_template=config.use_template,
            max_len=config.max_length,
        )
    else:
        valid_dataset = None
        logger.info("No validation dataset provided")

    trainer = Trainer(
        config,
        model=model,
        train_data=train_dataset,
        valid_data=valid_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
