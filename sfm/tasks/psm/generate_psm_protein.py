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

VOCAB = {
    # "<pad>": 0,  # padding
    # "1"-"127": 1-127, # atom type
    # "<cell_corner>": 128, use for pbc material
    "L": 130,
    "A": 131,
    "G": 132,
    "V": 133,
    "S": 134,
    "E": 135,
    "R": 136,
    "T": 137,
    "I": 138,
    "D": 139,
    "P": 140,
    "K": 141,
    "Q": 142,
    "N": 143,
    "F": 144,
    "Y": 145,
    "M": 146,
    "H": 147,
    "W": 148,
    "C": 149,
    "X": 150,
    "B": 151,
    "U": 152,
    "Z": 153,
    "O": 154,
    "-": 155,
    ".": 156,
    "<mask>": 157,
    "<cls>": 158,
    "<eos>": 159,
    # "<unk>": 160,
}
AA1TO3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


@dataclass
class Config(DistributedTrainConfig, PSMConfig):
    backbone_config: Dict[str, Any] = MISSING
    backbone: str = "graphormer"
    ode_mode: bool = False
    fasta_list: str = "/tmp/fasta_list"
    output_dir: str = "/tmp/output_dir"


cs = ConfigStore.instance()
cs.store(name="config_psm_schema", node=Config)


def parse_fastafile(fastafile):
    """Parse fasta file."""

    seqs = []
    try:
        with open(fastafile, "r") as fin:
            header, seq = "", []
            for line in fin:
                if line[0] == ">":
                    seqs.append((header, "".join(seq)))
                    header, seq = line.strip(), []
                else:
                    seq.append(line.strip())
            seqs.append((header, "".join(seq)))
            del seqs[0]

    except Exception as e:
        print(f"ERROR: wrong fasta {fastafile}\n      {e}", file=sys.stderr)

    return seqs


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

    model = PSMModel(args, loss_fn=DiffMAE3dCriterions, load_ckpt=True).cuda()
    model.eval()

    args.data_path = "/home/peiranjin/output/sample_result/casp_14and15.lmdb"
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
        # sequence = data["token_id"][0]

        result = model.sample(data, data)

        pred_pos = result["pred_pos"].tolist()
        print(f"pred_pos: {pred_pos}")
        # assert 1 == len(pred_pos), f"ERROR: batch size of {target} should =1"
        # assert len(pred_pos[0]) == len(sequence), f"ERROR: wrong length {target}"
        # assert 3 == len(pred_pos[0][0]), f"ERROR: 3D coordinates for {target}"

        # # generate atom lines with atom position
        # coords = pred_pos[0]
        # atomlines = []
        # for i, (x, y, z) in enumerate(coords):
        #     atomidx = i + 1
        #     resname = AA1TO3.get(sequence[i], "UNK")
        #     resnumb = i + 1
        #     atomlines.append(
        #         f"ATOM  {atomidx:>5d}  CA  {resname} {chain}{resnumb:>4d}    "
        #         f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C  \n"
        #     )
        # atomlines.append("TER\n")
        # atomlines.append("END\n")

        # # write results to .pdb
        # pdb_file = os.path.join(args.output_dir, f"{target}.pdb")
        # with open(pdb_file, "w") as fp:
        #     fp.writelines(atomlines)
        # print(f"Predict structure for {target} and write {pdb_file} done.")

    # # parse candidate list
    # with open(args.fasta_list, "r") as fp:
    #     fasta_paths = [_.strip() for _ in fp]
    # print(f"There are {len(fasta_paths)} targets in {args.fasta_list}.")

    # for path in tqdm(fasta_paths):
    #     # parse fasta file
    #     assert os.path.exists(path), f"ERROR: fasta file {path} does not exist!"
    #     seqs = parse_fastafile(path)
    #     assert 1 == len(seqs), f"ERROR: wrong file {path}"
    #     assert 2 == len(seqs[0]), f"ERROR: wrong fasta context {path}"
    #     print(f"Read sequence from fasta {path}.")

    #     header = seqs[0][0]
    #     assert header[0] == ">", f"ERROR: wrong header {header}"
    #     target = header.split()[0][1:]
    #     sequence = seqs[0][1]
    #     print(f"Sequence name is {target} and context is \n{sequence}")

    #     # parse target name and chain
    #     assert len(target) >= 5, f"ERROR: {target} is not a valid target name."
    #     if len(target) == 6 and target[4] == "_":
    #         tarname, chain = target[:4], target[5]
    #     elif target[0] == "T":
    #         tarname, chain = target, " "
    #     else:
    #         print(f"ERROR: {target} may be a wrong name.", file=sys.stderr)
    #         tarname, chain = target, " "
    #     print(target, tarname, chain)

    #     # predict CA position for sequence
    #     indices = [VOCAB.get(_, 150) for _ in sequence]
    #     aa_seq = torch.tensor(indices, dtype=torch.int64).unsqueeze(0).cuda()
    #     result = model.seq2structure(aa_seq)

    #     pred_pos = result["pred_pos"].tolist()
    #     assert 1 == len(pred_pos), f"ERROR: batch size of {target} should =1"
    #     assert len(pred_pos[0]) == len(sequence), f"ERROR: wrong length {target}"
    #     assert 3 == len(pred_pos[0][0]), f"ERROR: 3D coordinates for {target}"

    #     # generate atom lines with atom position
    #     coords = pred_pos[0]
    #     atomlines = []
    #     for i, (x, y, z) in enumerate(coords):
    #         atomidx = i + 1
    #         resname = AA1TO3.get(sequence[i], "UNK")
    #         resnumb = i + 1
    #         atomlines.append(
    #             f"ATOM  {atomidx:>5d}  CA  {resname} {chain}{resnumb:>4d}    "
    #             f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C  \n"
    #         )
    #     atomlines.append("TER\n")
    #     atomlines.append("END\n")

    #     # write results to .pdb
    #     pdb_file = os.path.join(args.output_dir, f"{target}.pdb")
    #     with open(pdb_file, "w") as fp:
    #         fp.writelines(atomlines)
    #     print(f"Predict structure for {target} and write {pdb_file} done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
    finally:
        wandb.finish()  # support to finish wandb logging
        logger.info("wandb finish logging!")
