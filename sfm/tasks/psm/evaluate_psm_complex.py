# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Tuple

import hydra
import lmdb
import numpy as np
import pandas as pd
import torch
import torch.distributed
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
from sfm.data.psm_data.dataset import ComplexDataset
from sfm.logging import logger
from sfm.models.psm.complexmodel import ComplexModel
from sfm.models.psm.loss.mae3ddiff import DiffMAE3dCriterions
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psm_optimizer import AdamFP16
from sfm.models.psm.psmmodel import PSMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer, seed_everything
from sfm.tasks.psm.finetune_psm_complex import ComplexMAE3dCriterions
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


cs = ConfigStore.instance()
cs.store(name="config_psm_schema", node=Config)


def parse_name_chain_from_target(target: str):
    return target.split("-")


@torch.no_grad()
def kabsch_align(
    P: torch.Tensor, Q: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD, in a batched manner.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Center the data
    P_centroid = P.mean(dim=0)
    Q_centroid = Q.mean(dim=0)

    translation = Q_centroid - P_centroid
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    # Compute the covariance matrix
    H = torch.matmul(P_centered.T, Q_centered)

    # SVD
    U, _, Vt = torch.linalg.svd(H)

    # right-hand rule
    d = torch.det(torch.matmul(Vt.T, U.T))
    if d < 0:
        Vt[-1, :] *= -1

    # Compute the optimal rotation
    R = torch.matmul(Vt.T, U.T)

    # Compute the RMSD
    P_rotated = torch.matmul(P_centered, R.T)
    RMSD = torch.sqrt(torch.sum((P_rotated - Q_centered) ** 2) / P.shape[0])

    return R, translation, RMSD


@torch.no_grad()
def ComplexRMSD(
    P: torch.Tensor, Q: torch.Tensor, protein_len: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the RMSD between two sets of points (P -> Q) after aligning them with protein atoms.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: The RMSD between the two sets of points after alignment.
    """
    P_protein = P[:protein_len]
    Q_protein = Q[:protein_len]
    R, translation, protein_RMSD = kabsch_align(
        P_protein, Q_protein
    )  # this RMSD is supposed to near 0, as the protein atoms are aligned
    P_aligned = torch.matmul(P, R.T) + translation
    # calculate ligand RMSD
    ligand_RMSD = torch.sqrt(
        torch.sum((P_aligned[protein_len:] - Q[protein_len:]) ** 2)
        / (P.shape[0] - protein_len)
    )
    # protein_RMSD = torch.sqrt(torch.mean((P_aligned[:protein_len] - Q[:protein_len]) ** 2)) # this RMSD is supposed to near 0, as the protein atoms are aligned
    total_RMSD = torch.sqrt(torch.sum((P_aligned - Q) ** 2) / P.shape[0])
    return total_RMSD, ligand_RMSD, protein_RMSD


def load_data(args):
    dataset = ComplexDataset(args, args.data_path)
    train_data, valid_data = dataset.split_dataset()
    return train_data, valid_data


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

    logger.info(f"Loading model from {args.loadcheck_path}.")
    model = ComplexModel(args, loss_fn=ComplexMAE3dCriterions).cuda()
    model.eval()

    logger.info(f"Loading test dataset from {args.data_path}.")
    train_data, valid_data = load_data(args)

    sampler = DistributedSampler(valid_data, shuffle=False)
    train_data_loader = DataLoader(
        valid_data,
        sampler=sampler,
        batch_size=1,
        collate_fn=valid_data.collate,
        drop_last=False,
    )

    logger.info(f"Start to evaluate protein structure for {args.data_path}.")
    total_cnt = 0
    hit_cnt = 0
    for data in train_data_loader:
        data = {k: v.cuda() for k, v in data.items()}
        # Skip the data with protein_len + 1 == num_atoms
        # if data["protein_len"][0] + 1 == data["num_atoms"][0]:
        #   continue
        ori_pos = data["pos"][0].cpu().tolist()
        idx = data["idx"][0].cpu()
        target = valid_data.key_list[idx]
        logger.success(f"Start to evaluate structure for {target}.")

        sequence = data["token_id"][0].cpu().tolist()
        logger.success(f"Sequence length for {target} is {len(sequence)}.")

        tarname, chain = parse_name_chain_from_target(target)
        logger.success(f"The {target} name is {tarname} and chain is {chain}.")

        # write predicted CA position to pdb file
        pdb_file = os.path.join(args.save_dir, f"{target}-1.pdb")
        std_file = os.path.join(args.save_dir, f"{target}-std.pdb")
        if os.path.exists(pdb_file):
            logger.warning(f"{pdb_file} already exists, skip generating.")
            continue

        logger.success(f"Predicted {target} structure by SFM model.")
        result = model.sample(data)
        coords = result["pred_pos"][0].tolist()

        logger.success(f"Write predicted {target} structure to {pdb_file}.")
        ligand_name = chain.split("_")[-1]
        chain = tarname.split("_")[-1]

        predict = torch.tensor(coords)
        standard = torch.tensor(ori_pos)
        total_RMSD, ligand_RMSD, protein_RMSD = ComplexRMSD(
            predict, standard, data["protein_len"][0]
        )
        pocket_RMSD = kabsch_align(
            predict[data["protein_len"][0] :], standard[data["protein_len"][0] :]
        )
        pocket_RMSD = pocket_RMSD[2]
        if pocket_RMSD <= 2:
            hit_cnt += 1
        total_cnt += 1
        logger.success(
            f"Total RMSD: {total_RMSD:.4f}, Ligand RMSD: {ligand_RMSD:.4f}, Protein RMSD: {protein_RMSD:.4f}, Pocket RMSD: {pocket_RMSD:.4f}"
        )
        logger.info(f"Hit rate: {hit_cnt}/{total_cnt} = {hit_cnt/total_cnt:.4f}")

        def getPDB(coords):
            atomlines = []
            for i, (x, y, z) in enumerate(coords):
                atomidx = i + 1
                if sequence[i] in VOCAB2AA:
                    resname = VOCAB2AA[sequence[i]]
                    resnumb = i + 1
                    atomlines.append(
                        f"ATOM  {atomidx:>5d}  CA  {resname} {chain}{resnumb:>4d}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C  \n"
                    )
                else:
                    Z_symbol_dict = {
                        1: "H",
                        2: "He",
                        3: "Li",
                        4: "Be",
                        5: "B",
                        6: "C",
                        7: "N",
                        8: "O",
                        9: "F",
                        10: "Ne",
                        11: "Na",
                        12: "Mg",
                        13: "Al",
                        14: "Si",
                        15: "P",
                        16: "S",
                        17: "Cl",
                        18: "Ar",
                        19: "K",
                        20: "Ca",
                        21: "Sc",
                        22: "Ti",
                        23: "V",
                        24: "Cr",
                        25: "Mn",
                        26: "Fe",
                        27: "Co",
                        28: "Ni",
                        29: "Cd",
                        30: "Zn",
                        31: "Ge",
                        32: "Ga",
                        33: "As",
                        34: "Se",
                        35: "Br",
                        36: "Kr",
                        37: "Rb",
                        38: "Sr",
                        39: "Y",
                        40: "Zr",
                        41: "Nb",
                        42: "Mo",
                        43: "Tc",
                        44: "Ru",
                        45: "Rh",
                        46: "Pd",
                        47: "Ag",
                        48: "Cd",
                        49: "In",
                        50: "Sn",
                        51: "Sb",
                        52: "Te",
                        53: "I",
                        54: "Xe",
                        55: "Cs",
                        56: "Ba",
                        57: "La",
                        72: "Hf",
                        73: "Ta",
                        74: "W",
                        75: "Re",
                        76: "Os",
                        77: "Ir",
                        78: "Pt",
                        79: "Au",
                        80: "Hg",
                        81: "Tl",
                        82: "Pb",
                        83: "Bi",
                        84: "Po",
                        85: "At",
                        86: "Rn",
                    }
                    resnumb = i + 1
                    atomname = Z_symbol_dict[sequence[i] - 1]
                    if len(atomname) == 1:
                        atomname = atomname + " "
                    atomlines.append(
                        f"HETATM{atomidx:>5d}  {atomname}  {ligand_name} {chain}{resnumb:>4d}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           {atomname}  \n"
                    )

            atomlines.append("TER\n")
            atomlines.append("END\n")
            return atomlines

        atomlines = getPDB(coords)
        with open(pdb_file, "w") as fp:
            fp.writelines(atomlines)
        stdlines = getPDB(ori_pos)
        with open(std_file, "w") as fp:
            fp.writelines(stdlines)

    torch.distributed.barrier()
    result_list = [torch.zeros(2, dtype=torch.int32) for _ in range(args.world_size)]
    torch.distributed.all_gather(
        result_list, torch.tensor([hit_cnt, total_cnt], dtype=torch.int32)
    )
    if args.rank == 0:
        hit_cnt = sum([x[0] for x in result_list])
        total_cnt = sum([x[1] for x in result_list])
        logger.info(f"Hit rate: {hit_cnt}/{total_cnt} = {hit_cnt/total_cnt:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
    finally:
        wandb.finish()  # support to finish wandb logging
        logger.info("wandb finish logging!")
