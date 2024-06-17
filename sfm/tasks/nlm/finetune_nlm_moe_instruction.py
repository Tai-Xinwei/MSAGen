# -*- coding: utf-8 -*-
from dataclasses import dataclass

from sfm.data.sci_data.dataset import RawTextSciDataset
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
    "sfm_nlm_moe_tiny": sfm_nlm_moe_tiny_config,
    "sfm_nlm_moe_8x7b": sfm_nlm_moe_8x7b_config,
}


@dataclass
class InstructionConfig(MoeModelConfig):
    conditional_generation: bool = False
    use_template: bool = False
    max_length: int = 1024


@cli(InstructionConfig)
def main(args):
    tokenizer = NlmTokenizer.from_pretrained(args.dict_path)
    args.vocab_size = len(tokenizer)
    args.pad_token_id = tokenizer.pad_token_id

    config = arg_utils.from_args(args, InstructionConfig)
    config = config_registry.get(config.model_type, sfm_nlm_moe_tiny_config)(config)
    config.ft = True

    logger.info(f"config: {config}")

    model = Model(config)

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
        logger.warning("No valid data path provided")
        valid_dataset = None

    trainer = Trainer(
        model=model,
        train_data=train_dataset,
        valid_data=valid_dataset,
        args=config,
        loss_log_dict={"lm_loss": 0.0, "lb_loss": 0.0},
    )
    trainer.train()


if __name__ == "__main__":
    main()
