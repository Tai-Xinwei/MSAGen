# -*- coding: utf-8 -*-

from sfm.data.sci_data.dataset import ProcessedSciDataset
from sfm.data.sci_data.NlmTokenizer import NlmTokenizer
from sfm.logging import logger
from sfm.models.nlm.moe_config import (
    MoeModelConfig,
    sfm_nlm_moe_8x7b_config,
    sfm_nlm_moe_tiny_config,
)
from sfm.models.nlm.moe_model import Model
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli

config_registry = {
    "scigptmoe_tiny": sfm_nlm_moe_tiny_config,
    "scigptmoe_8x7b": sfm_nlm_moe_8x7b_config,
}


@cli(MoeModelConfig)
def main(args) -> None:
    assert (
        args.train_data_path is not None and len(args.train_data_path) > 0
    ), f"train_dataset is {args.train_data_path} it should not be None or empty"

    assert (
        args.valid_data_path is not None and len(args.valid_data_path) > 0
    ), f"valid_dataset is {args.valid_data_path} it should not be None or empty"

    if not args.vocab_size:
        tokenizer = NlmTokenizer.from_pretrained(args.dict_path)
        args.vocab_size = len(tokenizer)  # now we have new tokens
        args.pad_token_id = tokenizer.pad_token_id

    config = arg_utils.from_args(args, MoeModelConfig)
    config = config_registry.get(config.model_type, sfm_nlm_moe_tiny_config)(config)

    logger.info(f"config: {config}")

    model = Model(config)

    train_dataset = ProcessedSciDataset(
        config.train_data_path,
        args.pad_token_id,
        config.max_position_embeddings,
        eos_idx=config.eos_token_id,
    )

    valid_dataset = ProcessedSciDataset(
        config.valid_data_path,
        args.pad_token_id,
        config.max_position_embeddings,
        eos_idx=config.eos_token_id,
    )

    trainer = Trainer(
        args=config,
        model=model,
        train_data=train_dataset,
        valid_data=valid_dataset,
        loss_log_dict={"lm_loss": 0.0, "lb_loss": 0.0},
    )
    trainer.train()


if __name__ == "__main__":
    main()
