# -*- coding: utf-8 -*-
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from tqdm import tqdm

import wandb  # isort:skip

try:
    from apex.optimizers import FusedAdam as AdamW
except:
    from torch.optim.adamw import AdamW

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])
from torch.utils.data import DataLoader, DistributedSampler

from sfm.data.psm_data.ft_prot_dataset import ProteinSamplingDataset
from sfm.logging import logger
from sfm.models.psm.loss.mae3ddiff import DiffMAE3dCriterions
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psm_optimizer import AdamFP16
from sfm.models.psm.psmmodel import PSMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer, seed_everything
from sfm.utils import env_init
from sfm.utils.cli_utils import wandb_init

# Vocabulary defined on sfm/data/psm_data/dataset.py:ProteinDownstreamDataset (self.vocab)
VOCAB2AA = {
    130: "LEU",  # "L"
    131: "ALA",  # "A"
    132: "GLY",  # "G"
    133: "VAL",  # "V"
    134: "SER",  # "S"
    135: "GLU",  # "E"
    136: "ARG",  # "R"
    137: "THR",  # "T"
    138: "ILE",  # "I"
    139: "ASP",  # "D"
    140: "PRO",  # "P"
    141: "LYS",  # "K"
    142: "GLN",  # "Q"
    143: "ASN",  # "N"
    144: "PHE",  # "F"
    145: "TYR",  # "Y"
    146: "MET",  # "M"
    147: "HIS",  # "H"
    148: "TRP",  # "W"
    149: "CYS",  # "C"
    150: "UNK",  # "X"
    # other_code: "UNK",
}


@dataclass
class Config(DistributedTrainConfig, PSMConfig):
    backbone_config: Dict[str, Any] = MISSING
    backbone: str = "graphormer"
    ode_mode: bool = False
    output_dir: str = "/tmp/jianwzhu_output"


cs = ConfigStore.instance()
cs.store(name="config_psm_schema", node=Config)


@hydra.main(
    version_base="1.3",
    config_path="../../../config_file",
    config_name="config_psm",
)
def main(args: DictConfig) -> None:
    args = OmegaConf.to_object(args)
    assert isinstance(
        args, Config
    ), f"args must be an instance of Config! But it is {type(args)}"

    assert os.path.exists(
        args.output_dir
    ), f"ERROR: output_dir {args.output_dir} does not exist!"

    wandb_init(args)
    seed_everything(args.seed)
    env_init.set_env(args)

    args.psm_validation_mode = True
    model = PSMModel(args, loss_fn=DiffMAE3dCriterions).cuda()
    model.eval()

    # args.data_path = "/home/peiranjin/output/sample_result/casp_14and15.lmdb"
    args.data_path = os.path.join(args.data_path, "casp_14and15.lmdb")
    dataset = ProteinSamplingDataset(args, args.data_path)
    # sampler = DistributedSampler(
    # dataset
    # )
    train_data_loader = DataLoader(
        dataset,
        # sampler=sampler,
        batch_size=1,
        collate_fn=dataset.collate,
        drop_last=False,
    )

    print(f"Start to predict protein structure for {args.data_path}.")
    for data in train_data_loader:
        data = {k: v.cuda() for k, v in data.items()}
        target = "T9999"
        sequence = data["token_id"][0]
        print(f"Sequence name is {target} and context is \n{sequence}")

        # predict CA position for sequence
        result = model.sample(data, data)
        pred_pos = result["pred_pos"].tolist()
        assert 1 == len(pred_pos), f"ERROR: batch size of {target} should =1"
        assert len(pred_pos[0]) == len(sequence), f"ERROR: wrong length {target}"
        assert 3 == len(pred_pos[0][0]), f"ERROR: 3D coordinates for {target}"

        # parse target name and chain
        assert len(target) >= 5, f"ERROR: {target} is not a valid target name."
        if len(target) == 6 and target[4] == "_":
            tarname, chain = target[:4], target[5]
        elif target[0] == "T":
            tarname, chain = target, " "
        else:
            print(f"ERROR: {target} may be a wrong name.", file=sys.stderr)
            tarname, chain = target, " "
        print(target, tarname, chain)

        # generate atom lines with atom position
        coords = pred_pos[0]
        atomlines = []
        for i, (x, y, z) in enumerate(coords):
            atomidx = i + 1
            resname = VOCAB2AA.get(sequence[i], "UNK")
            resnumb = i + 1
            atomlines.append(
                f"ATOM  {atomidx:>5d}  CA  {resname} {chain}{resnumb:>4d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C  \n"
            )
        atomlines.append("TER\n")
        atomlines.append("END\n")

        # write results to .pdb
        pdb_file = os.path.join(args.output_dir, f"{target}.pdb")
        with open(pdb_file, "w") as fp:
            fp.writelines(atomlines)
        print(f"Predict structure for {target} and write {pdb_file} done.")

        break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
    finally:
        wandb.finish()  # support to finish wandb logging
        logger.info("wandb finish logging!")
