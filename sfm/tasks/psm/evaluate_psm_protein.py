# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Mapping, Tuple, Union

import hydra
import lmdb
import pandas as pd
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

# Vocabulary defined on sfm/data/psm_data/dataset.py (variable: self.vocab)
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
    max_model_num: int = 1


cs = ConfigStore.instance()
cs.store(name="config_psm_schema", node=Config)


def parse_name_chain_from_target(target: str) -> Tuple[str, str]:
    # parse target name and chain
    if len(target) == 6 and target[4] == "_":
        # e.g. 1ctf_A
        name, chain = target[:4], target[5]
    elif (len(target) == 5 or len(target) == 7) and target[0] == "T":
        # e.g. T1024 or T1106s2
        name, chain = target, " "
    else:
        # test or other names
        logger.warning(f"ERROR: {target} may be a wrong name.")
        name, chain = target, " "
    return name, chain


def TMscore4Pair(predicted_pdb: str, native_pdb: str) -> Mapping[str, Any]:
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

    # check data format and TMscore output
    if len(lines) != i + 5 or lines[-1] != "\n":
        logger.warning(f"wrong TMscore between {predicted_pdb} and {native_pdb}")

    return score


def lddt4Pair(predicted_pdb: str, native_pdb: str) -> Mapping[str, Any]:
    """Calculate model score by using TMscore program"""
    # intialization
    score = {
        "PredictedPDB": os.path.basename(predicted_pdb),
        "NativePDB": os.path.basename(native_pdb),
        "PredictedLen": 0,
        "NativeLen": 0,
        "AlignLen": 0,
        "Radius": 0.0,
        "Coverage": 0.0,
        "LDDT": 0.0,
        "LocalLDDT": [],
    }

    status, _ = subprocess.getstatusoutput("which lddt")
    if status != 0:
        logger.warning("'lddt' does not exist in $PATH")
        return score

    if not os.path.exists(predicted_pdb):
        logger.warning(f"cannot found predicted model {predicted_pdb}")
        return score

    if not os.path.exists(native_pdb):
        logger.warning(f"cannot found native structure {native_pdb}")
        return score

    # execuate command and get output
    cmds = ["lddt", "-c", predicted_pdb, native_pdb]

    # open a temporary file and write the output to this file
    lines = []
    with tempfile.TemporaryFile() as tmp:
        proc = subprocess.Popen(cmds, stdout=tmp, stderr=tmp)
        proc.wait()
        tmp.seek(0)
        lines = [_.decode("utf-8") for _ in tmp.readlines()]

    # parse model score
    start_local = False
    local_lddts = []
    for i, l in enumerate(lines):
        cols = l.split()
        if l.startswith("Inclusion") and len(cols) > 2:
            score["Radius"] = float(cols[2])
        elif l.startswith("Coverage") and len(cols) > 6:
            score["Coverage"] = float(cols[1])
            score["NativeLen"] = int(cols[5])
        elif l.startswith("Global") and len(cols) > 3:
            score["LDDT"] = float(cols[3])
        elif l.startswith("Local"):
            continue
        elif l.startswith("Chain"):
            start_local = True
        elif start_local and len(cols) > 5:
            local_lddts.append(cols)
        else:
            continue
    score["PredictedLen"] = len(local_lddts)
    score["AlignLen"] = sum([_[4] != "-" for _ in local_lddts])
    score["LocalLDDT"] = [
        float("nan") if _[4] == "-" else float(_[4]) for _ in local_lddts
    ]

    # check data format and lddt output
    if len(lines) == i - 1 or lines[-1] != "\n":
        logger.warning(f"wrong LDDT between {predicted_pdb} and {native_pdb}")

    return score


def generate_pdb_atomlines(coords: list, sequence: str, chain: str = "A") -> list:
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
    return atomlines


def calculate_score(predlines: list, natilines: list, residx: set) -> dict:
    """Calculate score between predicted and native structure by TM-score"""

    def _select_residues_by_residx(atomlines: list):
        lines = []
        for line in atomlines:
            if line.startswith("ATOM"):
                resnum = int(line[22:26].strip())
                if resnum in residx:
                    lines.append(line)
        lines.append("TER\n")
        lines.append("END\n")
        return lines

    with (
        tempfile.NamedTemporaryFile() as predpdb,
        tempfile.NamedTemporaryFile() as natipdb,
    ):
        with open(predpdb.name, "w") as fp:
            fp.writelines(_select_residues_by_residx(predlines))
        with open(natipdb.name, "w") as fp:
            fp.writelines(_select_residues_by_residx(natilines))
        score = TMscore4Pair(predpdb.name, natipdb.name)
        score["LDDT"] = lddt4Pair(predpdb.name, natipdb.name)["LDDT"]
        return score


def evaluate_predicted_structure(
    metadata: Mapping[str, Union[list, str]],
    preddir: str,
    max_model_num: int = 1,
) -> pd.DataFrame:
    scores = []
    for target in tqdm(metadata["keys"]):
        taridx = metadata["keys"].index(target)
        # calculate score for each domain
        for domstr, domlen, domgroup in metadata["domains"][taridx]:
            try:
                residx = set()
                domseg = domstr.split(":")[1]
                for seg in domseg.split(","):
                    start, finish = [int(_) for _ in seg.split("-")]
                    residx.update(range(start, finish + 1))
                assert domlen == len(residx), f"domain length!={domlen}"
            except Exception as e:
                logger.error(f"Domain {domstr} parsing error, {e}")
                continue

            # process score for each predicted model
            for num in range(1, max_model_num + 1):
                score = {
                    "Target": domstr.split(":")[0],
                    "Length": domlen,
                    "Group": domgroup,
                    "Type": metadata["types"][taridx],
                    "ModelIndex": num,
                }
                try:
                    pdb_file = os.path.join(preddir, f"{target}-{num}.pdb")
                    with open(pdb_file, "r") as fp:
                        predlines = fp.readlines()
                    assert predlines, f" wrong predicted file {pdb_file}"
                    natilines = metadata["pdbs"][taridx]
                    score.update(calculate_score(predlines, natilines, residx))
                except Exception as e:
                    logger.error(f"Failed to evaluate {domstr}, {e}.")
                    continue
                scores.append(score)
    df = pd.DataFrame(scores)
    return df


def calculate_average_score(df: pd.DataFrame) -> pd.DataFrame:
    CATEGORY = {
        "CAMEO  Easy": ["Easy"],
        "CAMEO  Medi": ["Medium", "Hard"],
        "CASP14 Full": ["MultiDom"],
        "CASP14 Easy": ["TBM-easy", "TBM-hard"],
        "CASP14 Hard": ["FM/TBM", "FM"],
        "CASP15 Full": ["MultiDom"],
        "CASP15 Easy": ["TBM-easy", "TBM-hard"],
        "CASP15 Hard": ["FM/TBM", "FM"],
    }
    # group score by target
    records = []
    for target, gdf in df.groupby("Target"):
        record = {
            "Target": target,
            "Length": gdf["Length"].iloc[0],
            "Group": gdf["Group"].iloc[0],
            "Type": gdf["Type"].iloc[0],
        }
        max_model_num = gdf["ModelIndex"].max()
        for col in ["TMscore", "RMSD", "GDT_TS", "LDDT"]:
            maxscore = float("-inf")
            for num in range(1, max_model_num + 1):
                score = gdf[gdf["ModelIndex"] == num][col].iloc[0]
                record[f"Model{num}_{col}"] = score
                maxscore = max(maxscore, score)
            record[f"ModelMax_{col}"] = maxscore
        records.append(record)
    newdf = pd.DataFrame(records)
    # calculate average score for each category
    scores = []
    for key, groups in CATEGORY.items():
        _type = key.split()[0]
        subdf = newdf[(newdf["Type"] == _type) & newdf["Group"].isin(groups)]
        scores.append(
            {
                "CatAndGroup": key,
                "Number": len(subdf),
                "Top1TMscore": subdf["Model1_TMscore"].mean() * 100,
                "Top5TMscore": subdf["ModelMax_TMscore"].mean() * 100,
                "Top1LDDT": subdf["Model1_LDDT"].mean() * 100,
                "Top5LDDT": subdf["ModelMax_LDDT"].mean() * 100,
            }
        )
    meandf = pd.DataFrame(scores).set_index("CatAndGroup")
    return newdf, meandf


@hydra.main(
    version_base="1.3", config_path="../../../config_file", config_name="config_psm"
)
def main(args: DictConfig) -> None:
    args = OmegaConf.to_object(args)
    assert isinstance(
        args, Config
    ), f"args must be an instance of Config! But it is {type(args)}"

    wandb_init(args)
    seed_everything(args.seed)
    env_init.set_env(args)

    torch.distributed.init_process_group(
        backend="NCCL",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(hours=10),
    )
    torch.distributed.barrier()
    logger.success(
        f"DDP initialized on env:// rank: {args.rank}, "
        f"world_size: {args.world_size}, local_rank: {args.local_rank}."
    )

    logger.info(f"Loading metadata from {args.data_path}.")
    with lmdb.open(args.data_path, readonly=True).begin(write=False) as txn:
        metadata = bstr2obj(txn.get("__metadata__".encode()))
    logger.info(f"Metadata contains {len(metadata['keys'])} keys, pdbs, ...")

    logger.info(f"Loading model from {args.loadcheck_path}.")
    model = PSMModel(args, loss_fn=DiffMAE3dCriterions).cuda()
    model.eval()

    logger.info(f"Loading test dataset from {args.data_path}.")
    dataset = ProteinSamplingDataset(args, args.data_path)
    sampler = DistributedSampler(dataset, shuffle=False)
    train_data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        collate_fn=dataset.collate,
        drop_last=False,
    )

    logger.info(f"Start to evaluate protein structure for {args.data_path}.")
    for data in train_data_loader:
        data = {k: v.cuda() for k, v in data.items()}

        idx = data["idx"][0].cpu()
        target = dataset.keys[idx].split(" ")[0]
        if target not in metadata["keys"]:
            logger.error(f"{target} target name does not match metadata.")
            continue
        taridx = metadata["keys"].index(target)
        logger.success(f"Start to evaluate structure for {target}.")

        sequence = data["token_id"][0].cpu().tolist()
        if len(sequence) != metadata["sizes"][taridx]:
            logger.error(f"{target} sequence length does not match metadata.")
            continue
        logger.success(f"Sequence length for {target} is {len(sequence)}.")

        tarname, chain = parse_name_chain_from_target(target)
        logger.success(f"The {target} name is {tarname} and chain is {chain}.")

        for num in range(1, args.max_model_num + 1):
            # write predicted CA position to pdb file
            pdb_file = os.path.join(args.save_dir, f"{target}-{num}.pdb")
            if os.path.exists(pdb_file):
                logger.warning(f"{pdb_file} already exists, skip generating.")
                continue
            logger.success(f"Predicted {target} structure by SFM model.")
            result = model.sample(data)
            coords = result["pred_pos"][0].tolist()
            logger.success(f"Write predicted {target} structure to {pdb_file}.")
            atomlines = generate_pdb_atomlines(coords, sequence, chain)
            with open(pdb_file, "w") as fp:
                fp.writelines(atomlines)
            residx = set(range(1, len(sequence) + 1))
            sco = calculate_score(atomlines, metadata["pdbs"][taridx], residx)
            logger.success(
                f"Score of {target}: "
                f"TM-score={sco['TMscore']:6.4f}, LDDT={sco['LDDT']:6.4f}, "
                f"RMSD={sco['RMSD']:5.2f}, GDT_TS={sco['GDT_TS']:5.2f}."
            )

    # gather all results from different ranks
    torch.distributed.barrier()
    if args.rank != 0:
        return

    logger.info(f"TMscore between predicted.pdb and native.pdb {args.save_dir}")
    df = evaluate_predicted_structure(metadata, args.save_dir, args.max_model_num)
    print(df)

    logger.info(f"Write TM-score to {args.save_dir} and average it.")
    df.to_csv(os.path.join(args.save_dir, "Score4EachModel.csv"))
    newdf, meandf = calculate_average_score(df)
    newdf.to_csv(os.path.join(args.save_dir, "Score4Target.csv"))
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(meandf)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
    finally:
        wandb.finish()  # support to finish wandb logging
        logger.info("wandb finish logging!")
