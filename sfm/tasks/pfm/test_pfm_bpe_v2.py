# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from commons import (
    accuracy,
    area_under_prc,
    binary_accuracy,
    f1_max,
    mae,
    mse,
    pearsonr,
    rmse,
    spearmanr,
)
from finetune_pfm_bpe_v2 import (
    DownstreamConfig,
    init_model,
    load_batched_dataset,
    multi_label_transform,
)

from sfm.data.prot_data.dataset import DownstreamLMDBDataset
from sfm.logging import logger, metric_logger
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.models.pfm.pfm_mlm_config import PfmMlmConfig
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli
from sfm.utils.move_to_device import move_to_device


def test(args, trainer):
    """
    Validate the model on the validation data loader.
    """
    if trainer.valid_data_loader is None:
        logger.warning("No validation data, skip validation")
        return

    logger.info(
        "Start validation for epoch: {}, global step: {}",
        trainer.state.epoch,
        trainer.state.global_step,
    )

    pred, true = [], []

    for idx, batch_data in enumerate(trainer.valid_data_loader):
        trainer.model.eval()
        trainer.model.to(trainer.accelerator.device)
        batch_data = move_to_device(batch_data, trainer.accelerator.device)
        with torch.no_grad():
            output = trainer.model(batch_data)
            pred.append(output.to(torch.float32).squeeze(-1).detach().cpu())
        if (
            DownstreamLMDBDataset.TASKINFO[args.task_name]["type"]
            == "multi_classification"
        ):
            target = batch_data["target"].unsqueeze(1)
            target_offset = batch_data["target_offset"]
            target = multi_label_transform(
                target, target_offset, output.shape[0], trainer.model.n_classes
            )
            true.append(target.cpu())
        else:
            true.append(batch_data["target"].detach().cpu())

    if DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "regression":
        mean, std = DownstreamLMDBDataset.TASKINFO[args.task_name]["mean_std"]
        if args.label_normalize:
            pred = torch.cat(pred, axis=0) * std + mean
        else:
            pred = torch.cat(pred, axis=0)
        true = torch.cat(true, axis=0)
        test_fns = [pearsonr, spearmanr, mae, mse, rmse]
    elif DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "binary":
        pred = torch.cat(pred, axis=0)
        true = torch.cat(true, axis=0)
        test_fns = [binary_accuracy]
    elif DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "classification":
        pred = torch.cat(pred, axis=0)
        true = torch.cat(true, axis=0)
        test_fns = [accuracy]
    elif (
        DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "multi_classification"
    ):
        pred = torch.cat(pred, axis=0)  # (B, N_classes)
        true = torch.cat(true, axis=0)
        test_fns = [f1_max, area_under_prc]
    else:
        raise NotImplementedError()
    # assert pred.shape == true.shape
    results = dict()
    for fn in test_fns:
        results[fn.__name__] = fn(pred, true).item()
    # logger.info(f"Checkpoint: {args.loadcheck_path}; Test results: {results}")
    metric_logger.log(results, "test")
    return results


@cli(DistributedTrainConfig, PFMConfig, PfmMlmConfig, DownstreamConfig)
def test_checkpoint(args) -> None:
    logger.info(f"checkpoint_dir: {args.checkpoint_dir}")
    if args.checkpoint_dir:
        assert args.task_name in args.checkpoint_dir
        assert Path(args.checkpoint_dir).is_dir()
        logger.info(f"Test checkpoints from directory: {args.checkpoint_dir}")
        checkpoints = sorted(
            [str(i) for i in Path(args.checkpoint_dir).glob("checkpoint*.pt")]
        )
        logfile = open(str(Path(args.checkpoint_dir)) + ".txt", "w")
    else:
        assert Path(args.loadcheck_path).is_file()
        checkpoints = [args.loadcheck_path]
        logfile = sys.stdout

    train_data, val_data, testset_dict = load_batched_dataset(args)

    model = init_model(args, load_ckpt=False)

    for idx, checkpoint in enumerate(checkpoints):
        checkpoints_state = torch.load(checkpoint, map_location="cpu")
        if "model" in checkpoints_state:
            checkpoints_state = checkpoints_state["model"]
        elif "module" in checkpoints_state:
            checkpoints_state = checkpoints_state["module"]

        IncompatibleKeys = model.load_state_dict(checkpoints_state, strict=False)
        IncompatibleKeys = IncompatibleKeys._asdict()
        logger.info(f"checkpoint: {checkpoint} is loaded")
        logger.warning(f"Following keys are incompatible: {IncompatibleKeys.keys()}")

        if args.which_set == "valid":
            trainer = Trainer(
                args,
                model,
                train_data=train_data,
                valid_data=val_data,
            )
            logger.info(f"Testing valid set: {len(val_data)}")
            results = sorted(test(args, trainer).items(), key=lambda x: x[0])
            if not idx:
                print(
                    "checkpoint,set," + ",".join([i[0] for i in results]), file=logfile
                )

            print(
                f'"{checkpoint}",{args.which_set},'
                + ",".join([str(i[1]) for i in results]),
                file=logfile,
            )

        elif args.which_set == "test":
            for name, testset in testset_dict.items():
                trainer = Trainer(
                    args,
                    model,
                    train_data=train_data,
                    valid_data=testset,
                )
                logger.info(f"Testing test set: {len(val_data)} with name {name}")

                results = sorted(test(args, trainer).items(), key=lambda x: x[0])
                if not idx:
                    print(
                        "checkpoint,set," + ",".join([i[0] for i in results]),
                        file=logfile,
                    )
                print(
                    f'"{checkpoint}",{name},' + ",".join([str(i[1]) for i in results]),
                    file=logfile,
                )
        logfile.flush()
    if logfile != sys.stdout:
        logfile.close()


if __name__ == "__main__":
    test_checkpoint()
