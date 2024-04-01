# -*- coding: utf-8 -*-

from sfm.data.sci_data.dataset import ProcessedSciDataset
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from sfm.logging import logger
from sfm.models.scigpt.moe_config import ScigptMoeConfig, scigptmoe_tiny_config
from sfm.models.scigpt.scigpt_moe import ScigptMoeModel
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli

config_registry = {"scigptmoe_tiny": scigptmoe_tiny_config}


@cli(ScigptMoeConfig)
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

    config = arg_utils.from_args(args, ScigptMoeConfig)
    config = config_registry.get(config.model_type, scigptmoe_tiny_config)(config)

    logger.info(f"config: {config}")

    model = ScigptMoeModel(config)

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
        loss_log_dict={"lm_loss": 0.0, "lb_loss": 0.0},
    )
    trainer.train()


if __name__ == "__main__":
    main()
