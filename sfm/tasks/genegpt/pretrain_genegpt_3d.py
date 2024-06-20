# -*- coding: utf-8 -*-
import os
import sys

import wandb  # isort:skip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from megatron.arguments import parse_megatron_args
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.initialize import initialize_megatron
from sfm.data.gene_data.gene_dataset import GeneDataset
from sfm.data.gene_data.GeneTokenizer import GeneKMerTokenizer
from sfm.logging import logger
from sfm.models.genegpt.genegpt_3d import GenegptModel
from sfm.models.genegpt.genegpt_config import (
    GenegptConfig3D,
    genegpt3D_1b_config,
    genegpt3D_100m_config,
)
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli

config_registry = {
    "genegpt_100m": genegpt3D_100m_config,
    "genegpt_1b": genegpt3D_1b_config,
}


@cli(GenegptConfig3D, TransformerConfig, parse_megatron_args)
def main(args) -> None:
    # assert (
    #     args.train_data_path is not None and len(args.train_data_path) > 0
    # ), f"train_dataset is {args.train_data_path} it should not be None or empty"

    # assert (
    #     args.valid_data_path is not None and len(args.valid_data_path) > 0
    # ), f"valid_dataset is {args.valid_data_path} it should not be None or empty"
    # args.ifresume = True
    config = arg_utils.from_args(args, GenegptConfig3D)
    config = config_registry.get(config.model_type, genegpt3D_1b_config)(config)

    logger.info(f"config: {config}")
    tokenizer = GeneKMerTokenizer()
    initialize_megatron(args, tokenizer=tokenizer)
    logger.info("Initializing megatron for 3D training.")
    print(len(tokenizer))
    print(config)
    model = GenegptModel(args, config, tokenizer.vocab_size())

    train_dataset = GeneDataset(
        config.train_data_path,
        tokenizer.pad_token_id,
        max_len=args.max_position_embeddings,
    )
    valid_dataset = GeneDataset(
        config.valid_data_path,
        tokenizer.pad_token_id,
        max_len=args.max_position_embeddings,
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
