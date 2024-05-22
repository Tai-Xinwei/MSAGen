# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass

from sfm.data.psm_data.dataset import PCQM4Mv2LMDBDataset
from sfm.data.psm_data.unifieddataset import BatchedDataDataset
from sfm.models.psm.loss.evaluation import EvalPropMetric
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psmmodel import PSMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli


@dataclass
class PSMEvalConfig:
    eval_prop: str = "energy"
    eval_metric: str = "mae"

    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(args)
        for k, v in asdict(self).items():
            if hasattr(args, k):
                setattr(self, k, getattr(args, k))


def load_data(args):
    if args.dataset_names == "pcqm4mv2":
        dataset = PCQM4Mv2LMDBDataset(args, args.data_path)
        data = dataset.get_dataset()
    else:
        raise ValueError("invalid dataset name")

    return BatchedDataDataset(args, [data], len(data))


@cli(DistributedTrainConfig, PSMConfig, PSMEvalConfig)
def evaluate(args):
    data = load_data(args)

    model = PSMModel(args, load_ckpt=True, loss_fn=EvalPropMetric)
    trainer = Trainer(args, model, train_data=data, valid_data=data)

    trainer.validate()


if __name__ == "__main__":
    evaluate()
