# -*- coding: utf-8 -*-
import os

from sfm.data.tamgent2.datasets import TextToMolDataset
from sfm.models.tamgent.model import Tamgent2, Tamgent2Config
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli


@cli(Tamgent2Config)
def main(args):
    config = arg_utils.from_args(args, Tamgent2Config)

    train_dataset = TextToMolDataset.from_files(
        config.train_mol_path, config.train_txt_path
    )
    val_dataset = TextToMolDataset.from_files(config.val_mol_path, config.val_text_path)

    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    config.iters_per_epoch = (
        len(train_dataset)
        // config.train_batch_size
        // config.gradient_accumulation_steps
        // num_gpus
    )

    model = Tamgent2(config)

    trainer = Trainer(config, model, train_dataset, val_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
