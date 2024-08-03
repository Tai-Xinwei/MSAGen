# -*- coding: utf-8 -*-
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

    if len(idxes) > 768:
        idxes2 = np.argsort(dists)[:768]
        # only pick those in indes
        idxes = np.intersect1d(idxes, idxes2)
        idxes = np.sort(idxes)

    return idxes


def spatial_crop_psm(
    polymer_chains: Dict,
    non_polymers: list,
    polymer_chains_idxes: list,
    crop_size: int,
    center_ligand_idx: int,
    crop_center: np.ndarray,
) -> Dict:
    """
    Crop the polymer chains and non-polymer chains refer to crop_size.
    Args:
        polymer_chains: a dictionary of polymer chains
        non_polymers: a list of non-polymer chains(ligand)
        polymer_chains_idx: a list of polymer chains index
        crop_size: the size of the cropped polymer chains
        crop_center: the center of the crop
    Returns:
        cropped_chain_idxes_list: a list of cropped polymer chains index
    """

    cropped_chain_idxes_list = []
    for chain_name in polymer_chains_idxes:
        polymer_chain = polymer_chains[chain_name]

        # Crop the polymer chain
        cropped_chain_idxes = crop_chain(
            polymer_chain["center_coord"], crop_size, crop_center
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

    return cropped_chain_idxes_list
