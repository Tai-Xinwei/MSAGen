# -*- coding: utf-8 -*-
import math
import os

from sfm.data.threedimargen_data.dataset import MODE, ThreeDimARGenDataset
from sfm.data.threedimargen_data.tokenizer import ThreeDimARGenTokenizer
from sfm.logging import logger
from sfm.models.threedimargen.threedimargen_config import (
    ThreeDimARGenConfig,
    threedimargen_100m_config,
    threedimargen_200m_config,
    threedimargen_base_config,
    threedimargen_tiny_config,
)
from sfm.models.threedimargen.threedimargendiff import ThreeDimARGenDiffModel
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli

config_registry = {
    "threedimargen_tiny": threedimargen_tiny_config,
    "threedimargen": threedimargen_base_config,
    "threedimargen_100m": threedimargen_100m_config,
    "threedimargen_200m": threedimargen_200m_config,
}


@cli(ThreeDimARGenConfig)
def main(args) -> None:
    config = arg_utils.from_args(args, ThreeDimARGenConfig)
    config.tokenizer = "num"
    config = config_registry.get(config.model_type, threedimargen_base_config)(config)

    assert (
        config.train_data_path is not None and len(config.train_data_path) > 0
    ), f"train_dataset is {config.train_data_path} it should not be None or empty"

    # assert (
    #    config.valid_data_path is not None and len(config.valid_data_path) > 0
    # ), f"valid_dataset is {config.valid_data_path} it should not be None or empty"

    tokenizer = ThreeDimARGenTokenizer.from_file(config.dict_path, config)
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.padding_idx

    train_dataset = ThreeDimARGenDataset(
        tokenizer, config.train_data_path, config, mode=MODE.TRAIN
    )
    logger.info(f"loadded {len(train_dataset)} samples from train_dataset")
    if config.valid_data_path is not None and len(config.valid_data_path) > 0:
        valid_dataset = ThreeDimARGenDataset(
            tokenizer, config.valid_data_path, config, mode=MODE.VAL
        )
        logger.info(f"loadded {len(valid_dataset)} samples from valid_dataset")
    else:
        valid_dataset = None

    if config.total_num_epochs is not None and config.total_num_epochs > 0:
        config.total_num_steps = (
            math.ceil((len(train_dataset) // config.train_batch_size))
            * config.total_num_epochs
        )
    if os.path.exists(config.save_dir):
        config.ifresume = True

    logger.info(f"config: {config}")

    model = ThreeDimARGenDiffModel(config)

    trainer = Trainer(
        config,
        model=model,
        train_data=train_dataset,
        valid_data=valid_dataset,
    )
    trainer.train()
    # trainer.validate()


if __name__ == "__main__":
    main()
