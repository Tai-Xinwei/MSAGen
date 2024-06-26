# -*- coding: utf-8 -*-
import os
import sys

import wandb  # isort:skip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from megatron.arguments import parse_megatron_args
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.initialize import initialize_megatron
from sfm.data.sci_data.dataset import (
    ProcessedSciDatasetLmdb,
    ProcessedSciWeightedDatasetLmdb,
)
from sfm.data.sci_data.NlmTokenizer import NlmLlama3Tokenizer
from sfm.logging import logger
from sfm.models.nlm.moe_config import MoeModelConfig, sfm_nlm_1b_base_config
from sfm.models.nlm.nlm_base import NLMBaseModel
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli


@cli(MoeModelConfig, TransformerConfig, parse_megatron_args)
def main(args) -> None:
    assert (
        args.train_data_path is not None and len(args.train_data_path) > 0
    ), f"train_dataset is {args.train_data_path} it should not be None or empty"

    assert (
        args.valid_data_path is not None and len(args.valid_data_path) > 0
    ), f"valid_dataset is {args.valid_data_path} it should not be None or empty"

    # if not args.vocab_size:
    tokenizer = NlmLlama3Tokenizer.from_pretrained(args.dict_path)
    args.vocab_size = len(tokenizer)  # now we have new tokens
    args.pad_token_id = tokenizer.pad_token_id
    args = sfm_nlm_1b_base_config(args)
    initialize_megatron(args, tokenizer=tokenizer)

    model = NLMBaseModel(args, len(tokenizer))

    if args.weighted_dataset:
        train_dataset = ProcessedSciWeightedDatasetLmdb(
            args.data_dir,
            args.train_data_path,
            args.pad_token_id,
            args.max_position_embeddings,
            data_raito=args.data_raito,
        )
        valid_dataset = ProcessedSciDatasetLmdb(
            args.valid_data_path, args.pad_token_id, args.max_position_embeddings
        )
    else:
        train_dataset = ProcessedSciDatasetLmdb(
            args.train_data_path, args.pad_token_id, args.max_position_embeddings
        )
        valid_dataset = ProcessedSciDatasetLmdb(
            args.valid_data_path, args.pad_token_id, args.max_position_embeddings
        )

    logger.info("datasets loaded")

    trainer = Trainer(
        args,
        model=model,
        train_data=train_dataset,
        valid_data=valid_dataset,
        loss_log_dict={"lm_loss": 0.0},
    )
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
    finally:
        wandb.finish()  # support to finish wandb logging
        logger.info("wandb finish logging!")
