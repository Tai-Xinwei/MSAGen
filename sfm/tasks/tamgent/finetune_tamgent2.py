# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser

from sfm.data.tamgent2.datasets import TextToMolDataset
from sfm.models.tamgent.model import Tamgent2, Tamgent2Config
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils import arg_utils


def main():
    parser = ArgumentParser()
    parser = arg_utils.add_dataclass_to_parser([Tamgent2Config], parser)
    args = parser.parse_args()

    config = arg_utils.from_args(args, Tamgent2Config)

    config.val_batch_interval = 100

    train_dataset = TextToMolDataset.from_files(
        config.train_mol_path, config.train_txt_path
    )
    val_dataset = TextToMolDataset.from_files(config.val_mol_path, config.val_text_path)

    num_gpus = os.environ.get("WORLD_SIZE", 1)
    config.iters_per_epoch = (
        len(train_dataset) // config.train_batch_size // config.update_freq // num_gpus
    )

    model = Tamgent2(config)

    trainer = Trainer(config, model, train_dataset, val_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
