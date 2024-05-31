# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict

import hydra
import lmdb
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

import wandb  # isort:skip

try:
    from apex.optimizers import FusedAdam as AdamW
except:
    from torch.optim.adamw import AdamW

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])
from torch.utils.data import DataLoader, DistributedSampler

from sfm.data.prot_data.util import bstr2obj, obj2bstr
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


cs = ConfigStore.instance()
cs.store(name="config_psm_schema", node=Config)


def generate_pdbcontext_from_lmdb(lmdbdir):
    pdbcontext = {}
    with lmdb.open(lmdbdir, readonly=True).begin(write=False) as txn:
        metadata = bstr2obj(txn.get("__metadata__".encode()))
        logger.info(f"{len(metadata['keys'])} keys in {lmdbdir}")
        logger.info(f"{len(metadata['sizes'])} sizes in {lmdbdir}")
        for key, length in zip(metadata["keys"], metadata["sizes"]):
            value = txn.get(key.encode())
            assert value, f"Key {key} not found."
            data = bstr2obj(value)
            assert data.keys() == {"seq", "pdb"}, f"Wrong keys {data.keys()}."
            pdbcontext[key] = data["pdb"]
    return pdbcontext


def TMscore4Pair(predicted_pdb, native_pdb):
    """Calculate model score by using TMscore program"""
    # intialization
    score = {
        "PredictedPDB": os.path.basename(predicted_pdb),
        "NativePDB": os.path.basename(native_pdb),
        "PredictedLen": 0,
        "NativeLen": 0,
        "AlignLen": 0,
        "RMSD": 0.0,
        "TMscore": 0.0,
        "MaxSub": 0.0,
        "GDT_TS": 0.0,
        "GDT_HA": 0.0,
    }

    status, _ = subprocess.getstatusoutput("which TMscore")
    if status != 0:
        logger.warning("TMscore does not exist in $PATH")
        return score

    if not os.path.exists(predicted_pdb):
        logger.warning(f"cannot found predicted model {predicted_pdb}")
        return score

    if not os.path.exists(native_pdb):
        logger.warning(f"cannot found native structure {native_pdb}")
        return score

    # execuate command and get output
    cmds = ["TMscore", predicted_pdb, native_pdb]
    logger.info(" ".join(cmds))

    # open a temporary file and write the output to this file
    lines = []
    with tempfile.TemporaryFile() as tmp:
        proc = subprocess.Popen(cmds, stdout=tmp, stderr=tmp)
        proc.wait()
        tmp.seek(0)
        lines = [_.decode("utf-8") for _ in tmp.readlines()]

    # parse model score
    for i, l in enumerate(lines):
        cols = l.split()
        if l.startswith("Structure1:") and len(cols) > 3:
            score["PredictedLen"] = int(cols[3])
        elif l.startswith("Structure2:") and len(cols) > 3:
            score["NativeLen"] = int(cols[3])
        elif l.startswith("Number") and len(cols) > 5:
            score["AlignLen"] = int(cols[5])
        elif l.startswith("RMSD") and len(cols) > 5:
            score["RMSD"] = float(cols[5])
        elif l.startswith("TM-score") and len(cols) > 2:
            score["TMscore"] = float(cols[2])
        elif l.startswith("MaxSub") and len(cols) > 1:
            score["MaxSub"] = float(cols[1])
        elif l.startswith("GDT-TS") and len(cols) > 1:
            score["GDT_TS"] = float(cols[1]) * 100
        elif l.startswith("GDT-HA") and len(cols) > 1:
            score["GDT_HA"] = float(cols[1]) * 100
        elif l.startswith('(":"'):
            i += 1
            break
        else:
            continue
        logger.info(l, end="")

    # check data format and TMscore output
    if len(lines) != i + 5 or lines[-1] != "\n":
        logger.warning(f"wrong TMscore between {predicted_pdb} and {native_pdb}")

    return score


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

    wandb_init(args)
    seed_everything(args.seed)
    env_init.set_env(args)

    model = PSMModel(args, loss_fn=DiffMAE3dCriterions).cuda()
    model.eval()

    dataset = ProteinSamplingDataset(args, args.data_path)
    # sampler = DistributedSampler(
    # dataset
    # )
    train_data_loader = DataLoader(
        dataset,
        # sampler=sampler,
        shuffle=False,
        batch_size=1,
        collate_fn=dataset.collate,
        drop_last=False,
    )

    logger.info(f"Loading pdb atomlines from {args.data_path}.")
    pdbcontext = generate_pdbcontext_from_lmdb(args.data_path)

    logger.info(f"Start to predict protein structure for {args.data_path}.")
    for idx, data in enumerate(train_data_loader):
        target = dataset.keys[idx].split(" ")[0]
        data = {k: v.cuda() for k, v in data.items()}
        sequence = data["token_id"][0].cpu().tolist()
        logger.info(f"Sequence name is {target} and length is {len(sequence)}.")

        # predict CA position for sequence
        result = model.sample(data, data)
        pred_pos = result["pred_pos"].tolist()

        # parse target name and chain
        assert len(target) >= 5, f"ERROR: {target} is not a valid target name."
        if len(target) == 6 and target[4] == "_":
            tarname, chain = target[:4], target[5]
        elif target[0] == "T":
            tarname, chain = target, " "
        else:
            logger.warning(f"ERROR: {target} may be a wrong name.")
            tarname, chain = target, " "
        logger.info(target, tarname, chain)

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

        # write results to *.pdb
        model_num = 1
        pdb_file = os.path.join(args.save_dir, f"{target}-{model_num}.pdb")
        with open(pdb_file, "w") as fp:
            fp.writelines(atomlines)
        logger.info(f"Predicted structure for {target} written to {pdb_file}")

        # evaluate predicted structure by TMalign/TMscore
        print("Calculate TMscore between predicted and native pdb")
        with tempfile.NamedTemporaryFile() as tmp:
            with open(tmp.name, "w") as fp:
                fp.writelines(pdbcontext[target])
            score = TMscore4Pair(pdb_file, tmp.name)
            print(score)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
    finally:
        wandb.finish()  # support to finish wandb logging
        logger.info("wandb finish logging!")
