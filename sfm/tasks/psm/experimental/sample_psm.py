# -*- coding: utf-8 -*-
import io
import os
import pathlib
import sys
from typing import Any, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from dataclasses import asdict, dataclass

import hydra
import rdkit
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdmolfiles import MolToMolFile
from rdkit.Chem.rdmolops import SanitizeMol

from sfm.data.psm_data.ft_mol_dataset import GenericMoleculeLMDBDataset
from sfm.data.psm_data.unifieddataset import BatchedDataDataset, PM6FullLMDBDataset
from sfm.models.psm.loss.mae3ddiff import DiffMAE3dCriterions
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psmmodel import PSMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer, seed_everything
from sfm.tasks.psm.evaluate_molconfgen import evaluate_conf
from sfm.utils import env_init
from sfm.utils.cli_utils import wandb_init
from sfm.utils.move_to_device import move_to_device

bond_dict = {
    0: rdkit.Chem.rdchem.BondType.UNSPECIFIED,
    1: rdkit.Chem.rdchem.BondType.SINGLE,
    2: rdkit.Chem.rdchem.BondType.DOUBLE,
    3: rdkit.Chem.rdchem.BondType.TRIPLE,
    4: rdkit.Chem.rdchem.BondType.AROMATIC,
}


@dataclass
class PSMSampleConfig:
    num_conformers: int = 1
    sampling_output_dir: str = "./output/sampling"
    save_sampled_molecules: bool = True
    align_sampled_molecules: bool = True

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
    if args.dataset_names == "geom":
        dataset = GenericMoleculeLMDBDataset(args, args.data_path)
        data = dataset.get_dataset()
    elif args.dataset_names == "pm6":
        dataset = PM6FullLMDBDataset(args, args.data_path)
        _, data = dataset.split_dataset()

        data = BatchedDataDataset(args, [data], len(data))
    else:
        raise ValueError("invalid dataset name")

    return data


def setup_output_directory(args):
    if args.save_sampled_molecules is True:
        output_dir = args.sampling_output_dir
        if not os.path.exists(output_dir):
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=False)


def set_molecular_bonds(mol, edge_index, edge_order):
    mol = RWMol(mol)

    for index, order in zip(edge_index, edge_order):
        if int(index[0]) < 0 or int(index[1]) < 0:
            break

        mol.AddBond(int(index[0]), int(index[1]), bond_dict[int(order)])

    return mol


def convert_coords_to_molecule(atomic_numbers, positions, edge_index, edge_attr):
    pt = Chem.GetPeriodicTable()

    xyz = io.StringIO()
    xyz.write(f"{len(atomic_numbers)}\n\n")
    for atomic_number, position in zip(atomic_numbers, positions):
        symbol = pt.GetElementSymbol(int(atomic_number))
        x, y, z = map(float, position)
        xyz.write(f"{symbol}\t\t{x:.9f}\t\t{y:.9f}\t\t{z:.9f}\n")

    mol = Chem.MolFromXYZBlock(xyz.getvalue())
    mol = set_molecular_bonds(mol, edge_index, edge_attr[:, 0] + 1)

    return mol


def convert_output_to_molecule(batch_data, idx):
    num_atoms = batch_data["num_atoms"][idx]
    atomic_numbers = batch_data["node_attr"][idx][:num_atoms, 0] - 1
    positions = batch_data["pos"][idx][:num_atoms]

    num_edges = batch_data["num_edges"][idx]
    edge_index = batch_data["edge_index"][idx][:num_edges].reshape(
        num_edges // 2, 2, -1
    )[:, 0]
    edge_attr = batch_data["edge_attr"][idx][:num_edges].reshape(num_edges // 2, 2, -1)[
        :, 0
    ]

    mol = convert_coords_to_molecule(atomic_numbers, positions, edge_index, edge_attr)

    return mol


def strip_molecule(mol):
    mol = Chem.Mol(mol)
    mol = Chem.RemoveHs(mol)
    mol.RemoveAllConformers()

    return mol


def extract_molecules_from_batch(batch_data):
    batch_size = len(batch_data["pos"])
    return [convert_output_to_molecule(batch_data, i) for i in range(batch_size)]


def evaluate_rmsd(gen, ref):
    result = []
    for i, j in zip(gen, ref):
        try:
            result.append(torch.tensor(evaluate_conf(i, j), dtype=torch.float64))
        except Exception:
            result.append(torch.tensor([torch.nan, torch.nan], dtype=torch.float64))

    return result


def save_molecule_to_file(mol, path, label, rank, batch, index):
    MolToMolFile(mol, f"{path}/{label}_{rank}_{batch}_{index}.mol")


def save_molecule_files(ref, gen, output_dir, rank, idx, align=True):
    for i, (g, r) in enumerate(zip(ref, gen)):
        if align is True:
            AlignMol(g, r)

        save_molecule_to_file(r, output_dir, "ref", rank, idx, i)
        save_molecule_to_file(g, output_dir, "gen", rank, idx, i)


@dataclass
class Config(DistributedTrainConfig, PSMConfig, PSMSampleConfig):
    backbone_config: Dict[str, Any] = MISSING
    backbone: str = "graphormer"
    ode_mode: bool = False


cs = ConfigStore.instance()
cs.store(name="config_psm_schema", node=Config)


@hydra.main(
    version_base="1.3", config_path="../../../config_file", config_name="config_psm"
)
def sample(args):
    args = OmegaConf.to_object(args)
    assert isinstance(
        args, Config
    ), f"args must be an instance of Config! But it is {type(args)}"

    wandb_init(args)
    seed_everything(args.seed)
    env_init.set_env(args)

    # load dataset
    data = load_data(args)

    # load model checkpoint
    model = PSMModel(args, load_ckpt=True, loss_fn=DiffMAE3dCriterions)
    trainer = Trainer(args, model, train_data=data, valid_data=data)

    device = torch.device("cuda", args.local_rank)

    # setup
    model.eval()
    setup_output_directory(args)

    result = []

    with torch.no_grad():
        for idx, batch_data in enumerate(trainer.valid_data_loader):
            # extract original molecules
            ref = extract_molecules_from_batch(batch_data)
            # zero out positions
            batch_data["pos"] = torch.zeros_like(batch_data["pos"])

            move_to_device(batch_data, device)

            model.sample(batch_data)
            # extract generated molecules
            gen = extract_molecules_from_batch(batch_data)

            # evaluate RMSD
            result.extend(evaluate_rmsd(gen, ref))

            # save (aligned) molecules
            if args.save_sampled_molecules:
                save_molecule_files(
                    ref,
                    gen,
                    output_dir=args.sampling_output_dir,
                    rank=args.local_rank,
                    idx=idx,
                    align=args.align_sampled_molecules,
                )

            print(
                "mean rmsd: {:.4f} [{}]".format(
                    torch.nanmean(torch.stack(result)[:, 1]), len(result)
                )
            )

    result = torch.stack(result).to(device)

    # gather results
    if args.local_rank != 0:
        torch.distributed.gather(result, gather_list=None, dst=0)
        return

    results = [
        torch.zeros((len(result), 2), dtype=torch.float64, device=device)
        for _ in range(args.world_size)
    ]
    torch.distributed.gather(result, gather_list=results, dst=0)

    results = torch.cat(results)[:]
    covs = results[:, 0]
    covs = covs[~covs.isnan()]
    mats = results[:, 1]
    mats = mats[~mats.isnan()]

    print(
        "Coverage Mean: {:.4f} | Coverage Median: {:.4f} | Match Mean: {:.4f} | Match Median: {:.4f}".format(
            torch.mean(covs),
            torch.median(covs),
            torch.mean(mats),
            torch.median(mats),
        )
    )


if __name__ == "__main__":
    sample()
