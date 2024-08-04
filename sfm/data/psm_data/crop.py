# -*- coding: utf-8 -*-
import copy
from typing import Dict

import numpy as np
import torch

from sfm.data.prot_data import residue_constants as rc
from sfm.data.psm_data.utils import VOCAB


def crop_chain(
    chain_coords: np.ndarray, crop_size: int, crop_center: np.ndarray
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

    # remove the nan from the chain_coords
    chain_coords[np.isnan(chain_coords)] = 10000

    # Calculate the distance between the atoms and the crop center
    dists = np.linalg.norm(chain_coords - crop_center, axis=1)

    # Get the index of the atoms within the crop size
    idxes = np.where(dists < crop_size)[0]

    return idxes, dists


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
    dists = np.concatenate(dists)
    dist_idx = np.argsort(dists)[:keep_num]
    total_residue_num = 0
    select_residue_num = 0

    for idx, chain in enumerate(cropped_chain_idxes_list):
        cropped_chain_idxes = chain["cropped_chain_idxes"]
        # get the index of cropped_chain_idxes also in dist_idx
        new_crop_chain_idxes = np.intersect1d(
            cropped_chain_idxes + total_residue_num, dist_idx
        )

        if len(cropped_chain_idxes) > 0:
            nearest_residue_idx.append(
                {
                    "center_ligand_idx": chain["center_ligand_idx"],
                    "number_of_ligands": chain["number_of_ligands"],
                    "number_of_polymer_chains": chain["number_of_polymer_chains"],
                    "chain_name": chain["chain_name"],
                    "cropped_chain_idxes": new_crop_chain_idxes - total_residue_num,
                }
            )

        total_residue_num += len(cropped_chain_idxes)
        select_residue_num += len(new_crop_chain_idxes)

    assert (
        select_residue_num == keep_num
    ), f"select_residue_num: {select_residue_num}, keep_num: {keep_num} shouid be equal"

    return nearest_residue_idx


def spatial_crop_psm(
    polymer_chains: Dict,
    non_polymers: list,
    polymer_chains_idxes: list,
    crop_size: int,
    center_ligand_idx: int,
    crop_center: np.ndarray,
    keep_num: int = 768,
) -> Dict:
    """
    Crop the polymer chains and non-polymer chains refer to crop_size.
    Args:
        polymer_chains: a dictionary of polymer chains
        non_polymers: a list of non-polymer chains(ligand)
        polymer_chains_idx: a list of polymer chains index
        crop_size: the size of the cropped polymer chains
        crop_center: the center of the crop
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
            copy.deepcopy(polymer_chain["center_coord"]), crop_size, crop_center
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
        dists.append(dist)

    if total_residue_num < keep_num:
        return cropped_chain_idxes_list

    cropped_chain_idxes_list = keep_nearest_residue(
        cropped_chain_idxes_list, dists, keep_num=keep_num
    )

    return cropped_chain_idxes_list
