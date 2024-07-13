# -*- coding: utf-8 -*-
from typing import Dict

import numpy as np
import torch

from sfm.data.psm_data.utils import VOCAB


def spatial_crop_psm(
    polymer_chains: Dict,
    non_polymers: Dict,
    polymer_chains_idx: list,
    crop_size: int,
    crop_center: np.ndarray,
) -> Dict:
    """
    Crop the polymer chains and non-polymer chains refer to crop_size.
    Args:
        polymer_chains: a dictionary of polymer chains
        non_polymers: a dictionary of non-polymer chains
        polymer_chains_idx: a list of polymer chains index
        crop_size: the size of the cropped polymer chains
        crop_center: the center of the crop
    Returns:
        cropped_polymer_chains: a dictionary of cropped polymer chains
        cropped_non_polymers: a dictionary of cropped non-polymer chains
    """
    pass
    # cropped_polymer_chains = {}
    # cropped_non_polymers = {}
    # for chain_id, chain in polymer_chains.items():
    #     cropped_chain = crop_chain(chain, crop_size, device)
    #     cropped_polymer_chains[chain_id] = cropped_chain
    # for chain_id, chain in non_polymers.items():
    #     cropped_chain = crop_chain(chain, crop_size, device)
    #     cropped_non_polymers[chain_id] = cropped_chain
    # return cropped_polymer_chains, cropped_non_polymers
