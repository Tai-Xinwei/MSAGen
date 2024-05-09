# -*- coding: utf-8 -*-
import logging

import torch
from ocpmodels.common.utils import (
    compute_neighbors,
    get_pbc_distances,
    radius_graph_pbc,
)
from torch_cluster import radius_graph
from torch_geometric.data import Data


def generate_graph(
    data,
    cutoff,
    max_neighbors=None,
    use_pbc=None,
    otf_graph=None,
    enforce_max_neighbors_strictly=True,
):
    if not otf_graph:
        try:
            edge_index = data.edge_index
            if use_pbc:
                cell_offsets = data.cell_offsets
                neighbors = data.neighbors

        except AttributeError:
            logging.warning(
                "Turning otf_graph=True as required attributes not present in data object"
            )
            otf_graph = True

    if use_pbc:
        if otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data,
                cutoff,
                max_neighbors,
                enforce_max_neighbors_strictly,
            )

        out = get_pbc_distances(
            data.pos,
            edge_index,
            data.cell,
            cell_offsets,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )

        edge_index = out["edge_index"]
        edge_dist = out["distances"]
        cell_offset_distances = out["offsets"]
        distance_vec = out["distance_vec"]
    else:
        if otf_graph:
            edge_index = radius_graph(
                data.pos,
                r=cutoff,
                batch=data.batch,
                max_num_neighbors=max_neighbors,
            )

        j, i = edge_index
        distance_vec = data.pos[j] - data.pos[i]

        edge_dist = distance_vec.norm(dim=-1)
        cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
        cell_offset_distances = torch.zeros_like(cell_offsets, device=data.pos.device)
        neighbors = compute_neighbors(data, edge_index)

    return (
        edge_index,
        edge_dist,
        distance_vec,
        cell_offsets,
        cell_offset_distances,
        neighbors,
    )


def construct_o3irrps(dim, order):
    string = []
    for l in range(order + 1):
        string.append(f"{dim}x{l}e" if l % 2 == 0 else f"{dim}x{l}o")
    return "+".join(string)


def to_torchgeometric_Data(data: dict):
    torchgeometric_data = Data()
    for key in data.keys():
        torchgeometric_data[key] = data[key]
    return torchgeometric_data


def construct_o3irrps_base(dim, order):
    string = []
    for l in range(order + 1):
        string.append(f"{dim}x{l}e")
    return "+".join(string)
