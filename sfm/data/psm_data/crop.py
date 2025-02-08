# -*- coding: utf-8 -*-
import copy
import random
from typing import Dict

import numpy as np


def crop_chain(
    chain_coords: np.ndarray,
    crop_size: int,
    crop_center: np.ndarray,
    sample_mode: bool = False,
) -> np.ndarray:
    """
    Crop a chain refer to crop_size.
    Args:
        chain_coords: the coordinates of the chain
        crop_size: the size of the cropped chain
        crop_center: the center of the crop
    Returns:
        cropped_chain: a cropped chain
    """
    if sample_mode:
        chain_coords[np.isnan(chain_coords)] = 0.0
    else:
        # remove the nan from the chain_coords
        chain_coords[np.isnan(chain_coords)] = 10000

    # Calculate the distance between the atoms and the crop center
    dists = np.linalg.norm(chain_coords - crop_center, axis=1)

    # Get the index of the atoms within the crop size
    idxes = np.where(dists < crop_size)[0]

    return idxes, dists


def ligand_max_dist(ligand_coords: np.ndarray, crop_center: np.ndarray) -> np.ndarray:
    """
    Calculate the distance between the atoms and the crop center.
    Args:
        ligand_coords: the coordinates of the ligand
        crop_center: the center of the crop
    Returns:
        dists: a list of distances between the atoms and the crop center
    """
    ligand_coords[np.isnan(ligand_coords)] = 10000

    # Calculate the distance between the atoms and the crop center
    dists = np.linalg.norm(ligand_coords - crop_center, axis=1)
    # Get the index of the atoms within the crop size
    idxes = np.where(dists < 10000)[0]

    return dists[idxes].max()


def crop_ligand(
    ligand_coords: np.ndarray, crop_size: int, crop_center: np.ndarray
) -> np.ndarray:
    """
    Crop a chain refer to crop_size.
    Args:
        ligand_coords: the coordinates of the ligand
        crop_size: the size of the cropped chain
        crop_center: the center of the crop
    Returns:
        cropped_chain: a cropped chain
    """

    ligand_center = np.nanmean(ligand_coords, axis=0)

    # determine whether take the ligand with distance of the ligand_center and crop_center
    if np.linalg.norm(ligand_center - crop_center) > crop_size:
        return None, None, None

    ligand_coords[np.isnan(ligand_coords)] = 10000

    # Calculate the distance between the atoms and the crop center
    dists = np.linalg.norm(ligand_coords - crop_center, axis=1)

    # Get the index of the atoms within the crop size
    idxes = np.where(dists < 10000)[0]

    return idxes, dists, ligand_center


def keep_nearest_residue(
    cropped_chain_idxes_list: list, dists: list, keep_num: int = 768
) -> np.ndarray:
    """
    Keep the nearest residues to the crop center in a list of chains.
    Args:
        cropped_chain_idxes_list: a list of cropped polymer chains index
        dists: a list of distances between the atoms and the crop center
        keep_num: the number of residues to keep
    Returns:
        cropped_chain: a cropped chain index
    """

    nearest_residue_idx = []
    dists_all = np.concatenate(dists)
    dist_idx = np.argsort(dists_all)[:keep_num]
    total_residue_num = 0
    select_residue_num = 0

    for idx, chain in enumerate(cropped_chain_idxes_list):
        chain_len = len(dists[idx])
        cropped_chain_idxes = chain["cropped_chain_idxes"]
        # get the index of cropped_chain_idxes also in dist_idx
        new_crop_chain_idxes = np.intersect1d(
            cropped_chain_idxes + total_residue_num, dist_idx
        )
        if len(new_crop_chain_idxes) > 0:
            new_crop_chain_idxes = np.sort(new_crop_chain_idxes)
            nearest_residue_idx.append(
                {
                    "center_ligand_idx": chain["center_ligand_idx"],
                    "number_of_ligands": chain["number_of_ligands"],
                    "number_of_polymer_chains": chain["number_of_polymer_chains"],
                    "chain_name": chain["chain_name"],
                    "cropped_chain_idxes": new_crop_chain_idxes - total_residue_num,
                }
            )

        total_residue_num += chain_len
        select_residue_num += len(new_crop_chain_idxes)

    assert (
        select_residue_num == keep_num
    ), f"select_residue_num: {select_residue_num}, keep_num: {keep_num} shouid be equal, {len(dists_all)=}"

    return nearest_residue_idx


def spatial_crop_psm(
    polymer_chains: Dict,
    non_polymers: list,
    polymer_chains_idxes: list,
    crop_size: int,
    center_ligand_idx: int,
    crop_center: np.ndarray,
    ligand_buffer: int = 5,
    keep_num: int = 768,
    sample_mode: bool = False,
) -> Dict:
    """
    Crop the polymer chains and non-polymer chains refer to crop_size.
    Args:
        polymer_chains: a dictionary of polymer chains
        non_polymers: a list of non-polymer chains(ligand)
        polymer_chains_idx: a list of polymer chains index
        crop_size: the size of the cropped polymer chains
        crop_center: the center of the crop
        ligand_buffer: the buffer size of the ligand
        keep_num: the number of residues to keep
    Returns:
        cropped_chain_idxes_list: a list of cropped polymer chains index
    """

    cropped_chain_idxes_list = []
    total_residue_num = 0
    dists = []
    for chain_name in polymer_chains_idxes:
        polymer_chain = polymer_chains[chain_name]

        # Crop the polymer chain
        cropped_chain_idxes, dist = crop_chain(
            copy.deepcopy(polymer_chain["center_coord"]),
            crop_size,
            crop_center,
            sample_mode=sample_mode,
        )

        # if the cropped chain is not empty
        if len(cropped_chain_idxes) > 0:
            cropped_chain_idxes_list.append(
                {
                    "center_ligand_idx": center_ligand_idx,
                    "number_of_ligands": len(non_polymers),
                    "number_of_polymer_chains": len(polymer_chains_idxes),
                    "chain_name": chain_name,
                    "cropped_chain_idxes": cropped_chain_idxes,
                }
            )
            dists.append(dist)

        total_residue_num += len(cropped_chain_idxes)

    # include if ligand is fully in the crop radius
    total_atom_num = 0
    candidate_ligand_idx_list = []
    for ligand_idx, ligand in enumerate(non_polymers):
        max_dist = (
            ligand_max_dist(copy.deepcopy(ligand["node_coord"]), crop_center)
            + ligand_buffer
        )
        if max_dist > crop_size:
            continue
        candidate_ligand_idx_list.append(ligand_idx)
        dists.append([max_dist] * len(ligand["node_coord"]))
        total_atom_num += len(ligand["node_coord"])

    # if the total number of tokens cropped by the crop_radius is less than keep_num
    # then we include all of them
    if total_residue_num + total_atom_num < keep_num:
        return cropped_chain_idxes_list, candidate_ligand_idx_list

    # Recalculate the crop radius and crop the polymer chains and ligands again
    dists_all = np.concatenate(dists)
    cutoff_dists = np.sort(dists_all)[keep_num - 1]

    cropped_chain_idxes_list = []
    total_residue_num = 0
    for chain_name in polymer_chains_idxes:
        polymer_chain = polymer_chains[chain_name]

        # Crop the polymer chain
        cropped_chain_idxes, _ = crop_chain(
            copy.deepcopy(polymer_chain["center_coord"]),
            cutoff_dists,
            crop_center,
            sample_mode=sample_mode,
        )

        # if the cropped chain is not empty
        if len(cropped_chain_idxes) > 0:
            cropped_chain_idxes_list.append(
                {
                    "center_ligand_idx": center_ligand_idx,
                    "number_of_ligands": len(non_polymers),
                    "number_of_polymer_chains": len(polymer_chains_idxes),
                    "chain_name": chain_name,
                    "cropped_chain_idxes": cropped_chain_idxes,
                }
            )

        total_residue_num += len(cropped_chain_idxes)

    total_atom_num = 0
    candidate_ligand_idx_list = []
    for ligand_idx, ligand in enumerate(non_polymers):
        max_dist = (
            ligand_max_dist(copy.deepcopy(ligand["node_coord"]), crop_center)
            + ligand_buffer
        )
        if max_dist >= cutoff_dists:
            continue
        candidate_ligand_idx_list.append(ligand_idx)
        total_atom_num += len(ligand["node_coord"])

    assert (
        total_residue_num + total_atom_num <= keep_num
    ), f"{total_residue_num=}, {total_atom_num=}, {keep_num=}"

    return cropped_chain_idxes_list, candidate_ligand_idx_list
