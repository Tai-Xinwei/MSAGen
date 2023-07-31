# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser

from sfm.data.tamgent2.datasets import TextToMolData, TextToMolDataset
from sfm.logging import logger
from sfm.models.tamgent.model import Tamgent2, Tamgent2Config
from sfm.pipeline.accelerator.trainer import Model, ModelOutput, Trainer, TrainerConfig
from sfm.utils import arg_utils


def main():
    parser = ArgumentParser()
    parser = arg_utils.add_dataclass_to_parser([TrainerConfig, Tamgent2Config], parser)
    args = parser.parse_args()

    trainer_config = arg_utils.from_args(args, TrainerConfig)
    tamgent2_config = arg_utils.from_args(args, Tamgent2Config)

    train_dataset = TextToMolDataset.from_files(
        tamgent2_config.train_mol_path, tamgent2_config.train_txt_path
    )
    val_dataset = TextToMolDataset.from_files(
        tamgent2_config.val_mol_path, tamgent2_config.val_text_path
    )

    num_gpus = os.environ.get("WORLD_SIZE", 1)
    trainer_config.iters_per_epoch = (
        len(train_dataset)
        // trainer_config.train_batch_size
        // trainer_config.update_freq
        // num_gpus
    )
    logger.info(tamgent2_config)

    model = Tamgent2(tamgent2_config)

    trainer = Trainer(trainer_config, model, train_dataset, val_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
