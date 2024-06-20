# -*- coding: utf-8 -*-

from sfm.data.gene_data.gene_dataset import GeneDataset
from sfm.logging import logger
from sfm.models.genegpt.genegpt import GenegptModel
from sfm.models.genegpt.genegpt_config import (
    GenegptConfig,
    genegpt_1b_config,
    genegpt_100m_config,
)
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli

config_registry = {
    "genegpt_100m": genegpt_100m_config,
    "genegpt_1b": genegpt_1b_config,
}


@cli(GenegptConfig, DistributedTrainConfig)
def main(args) -> None:
    # assert (
    #     args.train_data_path is not None and len(args.train_data_path) > 0
    # ), f"train_dataset is {args.train_data_path} it should not be None or empty"

    # assert (
    #     args.valid_data_path is not None and len(args.valid_data_path) > 0
    # ), f"valid_dataset is {args.valid_data_path} it should not be None or empty"
    args.ifresume = True
    config = arg_utils.from_args(args, GenegptConfig)
    config = config_registry.get(config.model_type, genegpt_1b_config)(config)

    logger.info(f"config: {config}")

    model = GenegptModel(config)

    train_dataset = GeneDataset(
        config.train_data_path,
        config.pad_token_id,
        config.max_position_embeddings,
        args.max_tokens,
    )
    valid_dataset = GeneDataset(
        config.valid_data_path,
        config.pad_token_id,
        config.max_position_embeddings,
        args.max_tokens,
    )

    trainer = Trainer(
        args,
        model=model,
        train_data=train_dataset,
        valid_data=valid_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
