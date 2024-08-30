# -*- coding: utf-8 -*-
import os
import logging
from typing import Tuple, List
import numpy as np

from Bio.SVDSuperimposer import SVDSuperimposer
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
import tempfile

import subprocess
import sys
import argparse

logger=logging.getLogger(__name__)


def get_xyz(atomlines: List) -> Tuple[List, List]:
    """
    Extracts the xyz coordinates from the atomlines
    Returns a tuple of two lists, the first list contains the protein atoms, the second list contains the ligand atoms
    """
    protpos, ligpos = [], []
    for line in atomlines:
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        xyz = float(line[30:38]), float(line[38:46]), float(line[46:54])
        key = line[12:16] + line[21:26]
        if line[:6] == "ATOM  ":
            protpos.append((key, xyz))
        else:
            ligpos.append((key, xyz))
    return protpos, ligpos

def do_alignmet(x: np.ndarray, ref1: np.ndarray, ref2: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    ref1: Standard reference
    ref2: Reference to be aligned
    Aligns x via superimposition of ref2 to ref1
    returns the aligned x and the rmsd of ref1 and ref2
    """
    sup = SVDSuperimposer()
    sup.set(ref1, ref2)
    sup.run()
    rot, tran = sup.get_rotran()
    xt = np.dot(x, rot) + tran
    ref_rmsd = sup.get_rms()
    return xt, ref_rmsd

def protein_alignment(original_complex :Tuple[List, List], sampled_complex: Tuple[List, List]) -> Tuple[np.ndarray, float]:
    """
    Apply superimposition to align the ligand of the sampled complex to the ligand of the original complex
    Use the protein of the original complex as reference
    Returns the aligned ligand and the rmsd of the protein atoms
    """
    protein1, _ = original_complex
    protein2, ligand2 = sampled_complex

    protein1 = np.array([x[1] for x in protein1])
    protein2 = np.array([x[1] for x in protein2])
    ligand2 = np.array([x[1] for x in ligand2])

    return do_alignmet(ligand2, protein1, protein2)

def pocket_alignment(original_complex :Tuple[List, List], sampled_complex: Tuple[List, List], pocket_distance: int = 10) -> Tuple[np.ndarray, float]:
    """
    Aligns the protein of the sampled complex to the protein of the original complex
    Uses the pocket of the original complex as reference
    Returns the aligned ligand and the rmsd of the pocket atoms
    """
    sampled_protein = sampled_complex[0]
    original_protein = original_complex[0]
    sampled_ligand = sampled_complex[1]
    original_ligand = original_complex[1]

    commprt = set([_[0] for _ in sampled_protein]) & set(
                [_[0] for _ in original_protein]
            )
    commlig = set([_[0] for _ in sampled_ligand]) & set(
                [_[0] for _ in original_ligand]
            )
    smplprt = np.array([_[1] for _ in sampled_protein if _[0] in commprt])
    smpllig = np.array([_[1] for _ in sampled_ligand if _[0] in commlig])
    origprt = np.array([_[1] for _ in original_protein if _[0] in commprt])
    # origlig = np.array([_[1] for _ in original_ligand if _[0] in commlig])

    # Find the pocket atoms
    dist = np.linalg.norm(smplprt[:, None, :] - smpllig[None, :, :], axis=-1)
    mask = np.min(dist, axis=-1) < pocket_distance # in Angstrom

    smpl_pocket, orig_pocket = smplprt[mask], origprt[mask]
    return do_alignmet(smpllig,orig_pocket,smpl_pocket)

def process_record(sampled_filename, original_filename, ligand_sdf_filename, result_filename, pocket_boundary) -> float:
    """
    Processes a single record
    Aligns the ligand of the sampled complex to the ligand of the original complex, and saves the aligned ligand to result_filename
    Returns the rmsd of alignment reference
    """
    with open(sampled_filename, 'r') as f:
        sampled_lines = f.readlines()
    with open(original_filename, 'r') as f:
        original_lines = f.readlines()
    sampled_content = get_xyz(sampled_lines)
    original_content = get_xyz(original_lines)
    if pocket_boundary == -1:
        aligned_ligand, ref_rmsd = protein_alignment(original_content, sampled_content)
    else:
        aligned_ligand, ref_rmsd = pocket_alignment(original_content, sampled_content, pocket_boundary)

    mol = Chem.MolFromMolFile(ligand_sdf_filename)
    w = Chem.SDWriter(result_filename)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = aligned_ligand[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    w.write(mol)
    w.close()
    logging.info(f'Processed {sampled_filename} and saved to {result_filename}')
    return ref_rmsd

def process_run_record(sampled_folder: str, posebusters_folder: str, run_id: int, pocket_boundary: int) -> Tuple[str, dict]:
    """
    Processes all the records for a
    Returns csv for buster input, and extra info (currently only ref_rmsd)
    """
    csv_lines = []
    extra_info = {'ref_rmsd': []}
    POSEBUSTER_IDS = sorted(os.listdir(posebusters_folder))

    for idx in POSEBUSTER_IDS:
        try:
            sampled_filename = os.path.join(sampled_folder, f'{idx[:4]}-{run_id}.pdb')
            original_filename = os.path.join(sampled_folder, f'{idx[:4]}.pdb')
            posebuster_pdb_filename = os.path.join(posebusters_folder, idx, f'{idx}_protein.pdb')
            ligand_sdf_filename = os.path.join(posebusters_folder, idx, f'{idx}_ligand.sdf')
            ligands_sdf_filename = os.path.join(posebusters_folder, idx, f'{idx}_ligands.sdf')
            result_filename = os.path.join(sampled_folder, f'{idx[:4]}-{run_id}.sdf')

            ref_rmsd = process_record(sampled_filename, original_filename, ligand_sdf_filename, result_filename, pocket_boundary)

            extra_info['ref_rmsd'].append(ref_rmsd)
            csv_lines.append(f'{idx},{posebuster_pdb_filename},{ligands_sdf_filename},{result_filename}\n')
        except Exception as e:
            logger.warning(f'Error processing {idx} run {run_id}: {e}')
    return csv_lines, extra_info

def main(sampled_folder, posebusters_folder, run_count, result_path, pocket_boundary):
    extra_info_global = {}
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w+') as busters_input:
        busters_input.write('name,mol_cond,mol_true,mol_pred\n')
        for i in range(1, run_count+1):
            csvlines, extra_info = process_run_record(sampled_folder, posebusters_folder, i, pocket_boundary)
            for k, v in extra_info.items():
                if k not in extra_info_global:
                    extra_info_global[k] = []
                extra_info_global[k].extend(v)
            busters_input.writelines(csvlines)
            busters_input.flush()

        logging.info(f'Processed all {run_count} runs')

        # bust -t input.csv --full-report --outfmt csv > target.csv
        logger.info(f'Running bust -t {busters_input.name} --full-report --outfmt csv > {result_path}')
        with open(result_path, 'w') as f:
            subprocess.run(['bust', '-t', busters_input.name, '--full-report', '--outfmt', 'csv'], stdout=f)

        # Append extra info
        with open(result_path, 'r') as f:
            lines = f.readlines()

        lines[0] = lines[0].strip() + ''.join([f',{k}' for k in extra_info_global.keys()]) + '\n'
        for i, line in enumerate(lines[1:]):
            lines[i+1] = line.strip() + ''.join([f',{v[i]}' for v in extra_info_global.values()]) + '\n'

        with open (result_path, 'w') as f:
            f.writelines(lines)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('sampled_folder', type=str, help='Folder containing the sampled complexes')
    parser.add_argument('posebusters_folder', type=str, help='Folder containing the posebusters dataset')
    parser.add_argument('run_count', type=int, help='Number of runs', default=1)
    parser.add_argument('result_path', type=str, help='Path to save the result csv', default='result.csv')
    parser.add_argument('pocket_boundary', type=int, help='Distance in Angstrom to consider atoms in the pocket. Set to -1 to use the whole protein', default=-1)

    args = parser.parse_args()

    sampled_folder = args.sampled_folder
    posebusters_folder = args.posebusters_folder
    run_count = args.run_count
    result_path = args.result_path
    pocket_boundary = args.pocket_boundary

    main(sampled_folder, posebusters_folder, run_count, result_path, pocket_boundary)
