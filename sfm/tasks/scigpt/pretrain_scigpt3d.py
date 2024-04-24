# -*- coding: utf-8 -*-

from megatron.arguments import parse_megatron_args
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.initialize import initialize_megatron
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
from sfm.models.scigpt.scigpt import ScigptModel
from sfm.pipeline.accelerator.dataclasses import TrainStrategy
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


@cli(ScigptConfig, TransformerConfig, parse_megatron_args)
def main(args) -> None:
    assert (
        args.train_data_path is not None and len(args.train_data_path) > 0
    ), f"train_dataset is {args.train_data_path} it should not be None or empty"

    assert (
        args.valid_data_path is not None and len(args.valid_data_path) > 0
    ), f"valid_dataset is {args.valid_data_path} it should not be None or empty"

    if not args.vocab_size:
        tokenizer = SFMDecTokenizer.from_pretrained(args.dict_path)
        args.vocab_size = len(tokenizer)  # now we have new tokens
        args.pad_token_id = tokenizer.pad_token_id

    config = arg_utils.from_args(args, ScigptConfig)
    config = config_registry.get(config.model_type, scigpt_tiny_config)(config)

    logger.info(f"config: {config}")

    if args.strategy == TrainStrategy.Pipeline:
        model = ScigptModel(config)
    elif args.strategy == TrainStrategy.ThreeD:
        initialize_megatron(args, tokenizer=tokenizer)
        model = ScigptModel(args, len(tokenizer))

    train_dataset = ProcessedSciDataset(
        config.train_data_path, args.pad_token_id, config.max_position_embeddings
    )
    valid_dataset = ProcessedSciDataset(
        config.valid_data_path, args.pad_token_id, config.max_position_embeddings
    )

    trainer = Trainer(
        config,
        model=model,
        train_data=train_dataset,
        valid_data=valid_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
