# -*- coding: utf-8 -*-

from dataclasses import dataclass

from transformers import AutoTokenizer

from sfm.data.sci_data.dataset import RawTextSciDataset
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
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli
from sfm.utils.science_tokens import SCIENCE_TAG_TOKENS

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

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
    conditional_generation: bool = True
    use_template: bool = True
    max_length: int = 1024


@cli(InstructionConfig)
def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_ckpt_path,
        model_max_length=2048,
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

    special_tokens_dict["additional_special_tokens"] = SCIENCE_TAG_TOKENS
    tokenizer.add_special_tokens(special_tokens_dict)
    args.vocab_size = len(tokenizer)
    args.pad_token_id = tokenizer.pad_token_id
    args.eos_token_id = tokenizer.eos_token_id
    args.bos_token_id = tokenizer.bos_token_id
    args.unk_token_id = tokenizer.unk_token_id

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
