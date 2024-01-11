# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import torch

from sfm.data.prot_data.processed_mlm_dataset import ProcessedMlmDataset
from sfm.logging import logger
from sfm.models.pfm.pfm_mlm_config import (
    PfmMlmConfig,
    pfm_mlm_base_config,
    pfm_mlm_tiny_config,
    pfm_mlm_tiny_h24_config,
)
from sfm.models.pfm.pfm_mlm_model import PfmMlmModel, PfmMlmModelRd
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli

config_registry = {
    "pfm_mlm_tiny": pfm_mlm_tiny_config,
    "pfm_mlm_tiny_h24": pfm_mlm_tiny_h24_config,
    "pfm_mlm_base": pfm_mlm_base_config,
}


@cli(PfmMlmConfig)
def main(args) -> None:
    assert (
        args.train_data_path is not None and len(args.train_data_path) > 0
    ), f"train_dataset is {args.train_data_path} it should not be None or empty"

    assert (
        args.valid_data_path is not None and len(args.valid_data_path) > 0
    ), f"valid_dataset is {args.valid_data_path} it should not be None or empty"

    config = arg_utils.from_args(args, PfmMlmConfig)
    config = config_registry.get(config.model_type, pfm_mlm_tiny_config)(config)

    logger.info(f"config: {config}")

    if config.use_rd:
        model = PfmMlmModelRd(config)
    else:
        model = PfmMlmModel(config)

    train_dataset = ProcessedMlmDataset(
        path=config.train_data_path,
        bos_idx=config.bos_token_id,
        eos_idx=config.eos_token_id,
        pad_idx=config.pad_token_id,
        mask_idx=config.mask_token_id,
        vocab_size=config.vocab_size,
        mask_prob=config.mask_prob,
        leave_unmasked_prob=config.leave_unmasked_prob,
        random_token_prob=config.random_token_prob,
    )
    valid_dataset = ProcessedMlmDataset(
        path=config.valid_data_path,
        bos_idx=config.bos_token_id,
        eos_idx=config.eos_token_id,
        pad_idx=config.pad_token_id,
        mask_idx=config.mask_token_id,
        vocab_size=config.vocab_size,
        mask_prob=config.mask_prob,
        leave_unmasked_prob=config.leave_unmasked_prob,
        random_token_prob=config.random_token_prob,
    )

    print(train_dataset)
    print(valid_dataset)

    if config.compile_model:
        model = torch.compile(model)

    trainer = Trainer(
        config,
        model=model,
        train_data=train_dataset,
        valid_data=valid_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
