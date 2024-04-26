# -*- coding: utf-8 -*-
import argparse
import copy
import pickle
import zlib

import lmdb
import numpy as np
import torch
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign as MA
from spyrmsd.molecule import Molecule
from spyrmsd.rmsd import rmsdwrapper
from tqdm import tqdm


def obj2bstr(obj):
    return zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def bstr2obj(bstr):
    return pickle.loads(zlib.decompress(bstr))


def set_rdmol_positions(rdkit_mol: Chem.Mol, pos: torch.Tensor):
    mol = copy.deepcopy(rdkit_mol)
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def GetBestRMSD(probe: Chem.Mol, ref: Chem.Mol, keepHs: bool):
    if not keepHs:
        probe = Chem.RemoveHs(probe)
        ref = Chem.RemoveHs(ref)
    # old method with default GetBestRMS
    rmsd = MA.GetBestRMS(probe, ref, prbId=0, refId=0)
    # new method with spyrmsd
    # ref = Molecule.from_rdkit(ref, )
    # prb = Molecule.from_rdkit(prb, )
    # rmsd = rmsdwrapper(ref, prb, center=True, minimize=True, strip=True)[0]
    return rmsd


def get_rmsd_confusion_matrix(
    mol: Chem.Mol, ref: Chem.Mol, useFF: bool = False, keepHs: bool = False
):
    pos_gen = torch.cat(
        [
            torch.tensor(
                mol.GetConformer(i).GetPositions(), dtype=torch.float32
            ).unsqueeze(0)
            for i in range(mol.GetNumConformers())
        ],
        0,
    )
    pos_ref = torch.cat(
        [
            torch.tensor(
                ref.GetConformer(i).GetPositions(), dtype=torch.float32
            ).unsqueeze(0)
            for i in range(ref.GetNumConformers())
        ],
        0,
    )

    num_gen = len(pos_gen)
    num_ref = len(pos_ref)

    rmsd_confusion_mat = -1 * np.ones([num_ref, num_gen], dtype=float)

    for i in range(num_gen):
        gen_mol = set_rdmol_positions(mol, pos_gen[i])
        if useFF:
            AllChem.MMFFOptimizeMolecule(gen_mol)
        for j in range(num_ref):
            ref_mol = set_rdmol_positions(ref, pos_ref[j])
            rmsd_confusion_mat[j, i] = GetBestRMSD(gen_mol, ref_mol, keepHs=keepHs)

    return rmsd_confusion_mat


def evaluate_conf(
    mol: Chem.Mol,
    ref: Chem.Mol,
    useFF: bool = False,
    threshold: float = 0.5,
    keepHs: bool = False,
):
    rmsd_confusion_mat = get_rmsd_confusion_matrix(mol, ref, useFF=useFF, keepHs=keepHs)
    rmsd_ref_min = rmsd_confusion_mat.min(-1)
    return (rmsd_ref_min <= threshold).mean(), rmsd_ref_min.mean()


def rdkit_generate_conformers(mol: Chem.Mol, num_confs: int):
    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    assert mol.GetNumConformers() == 0

    AllChem.EmbedMultipleConfs(
        mol,
        numConfs=num_confs,
        maxAttempts=0,
        ETversion=2,
        ignoreSmoothingFailures=True,
    )
    if mol.GetNumConformers() != num_confs:
        print(
            "Warning: Failure cases occured, generated: %d , expected: %d."
            % (
                mol.GetNumConformers(),
                num_confs,
            )
        )
    return mol


def custom_generate_conformers(mol: Chem.Mol, num_confs: int):
    # mol = copy.deepcopy(mol)
    # mol.RemoveAllConformers()
    # assert mol.GetNumConformers() == 0

    # conformation generation code begins

    # conformation generation code ends
    return mol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input LMDB file.")
    parser.add_argument(
        "--start_idx", type=int, default=0, help="Start index for testing set."
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=200,
        help="End index for testing set. Default is 200 to keep same as ConfGF and DMCG paper.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of conformers generated for each molecule.",
    )
    parser.add_argument(
        "--core",
        type=int,
        default=6,
        help="Number of cores used for parallel evaluation.",
    )
    parser.add_argument(
        "--FF",
        action="store_true",
        help="Optimize generated conformation with UFF forcefield.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold used for COV metrics"
    )
    parser.add_argument("--keepHs", action="store_true", help="Keep Hs in the molecule")
    args = parser.parse_args()
    print(args)

    input_lmdb = lmdb.open(args.input, readonly=True, lock=False)
    txn = input_lmdb.begin(write=False)
    metadata = bstr2obj(txn.get(b"__metadata__"))
    keys = metadata["keys"]
    print(f"Found {len(keys)} keys in {args.input}")

    generated_mol_list = []
    for i in tqdm(range(args.start_idx, len(keys))):
        if i >= args.end_idx:
            break
        data = bstr2obj(txn.get(keys[i]))
        mol = rdkit_generate_conformers(data["mol"], num_confs=args.num_samples)
        num_pos_gen = mol.GetNumConformers()
        if num_pos_gen == 0:
            continue
        generated_mol_list.append([mol, data["mol"]])

    print("start getting results!")
    bad_case = 0
    filtered_data_list = []
    for i in tqdm(range(len(generated_mol_list))):
        mol, ref = generated_mol_list[i]
        smiles = Chem.MolToSmiles(mol)
        if "." in smiles:
            bad_case += 1
            continue
        filtered_data_list.append([mol, ref])

    cnt_conf = 0
    for i in range(len(filtered_data_list)):
        mol, ref = filtered_data_list[i]

        cnt_conf += ref.GetNumConformers()
    print(
        "%d bad cases, use %d mols with total %d confs"
        % (bad_case, len(filtered_data_list), cnt_conf)
    )

    result = Parallel(n_jobs=args.core)(
        delayed(evaluate_conf)(
            mol, ref, useFF=args.FF, threshold=args.threshold, keepHs=args.keepHs
        )
        for mol, ref in filtered_data_list
    )
    covs = np.array([r[0] for r in result])
    mats = np.array([r[1] for r in result])

    print(
        "Coverage Mean: %.4f | Coverage Median: %.4f | Match Mean: %.4f | Match Median: %.4f"
        % (covs.mean(), np.median(covs), mats.mean(), np.median(mats))
    )
