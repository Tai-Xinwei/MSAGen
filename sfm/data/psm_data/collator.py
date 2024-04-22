# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy

import numpy as np
import torch


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(0.0)
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_pos_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(-1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def collate_fn(
    items,
    min_node=-1,
    max_node=512,
    multi_hop_max_dist=20,
    spatial_pos_max=20,
    use_pbc=True,
):  # unify the data format
    # include the following fields: sample_type, token_type, idx, coords, cell, pbc, stress, forces, energy
    # need to add: node_type_edge, edge_input, in_degree, attn_edge_type, attn_bias, spatial_pos

    for item in items:
        if "pbc" not in item:
            item["pbc"] = torch.tensor([False, False, False])
        if "cell" not in item:
            item["cell"] = torch.zeros([3, 3])
        if "num_atoms" not in item:
            item["num_atoms"] = item["x"].size()[0]

        item["edge_input"] = item["edge_input"][:, :, :multi_hop_max_dist, :]

    # for idx, item in enumerate(items):
    #     item["attn_bias"][1:, 1:][item["spatial_pos"] >= spatial_pos_max] = float(
    #         "-inf"
    #     )

    max_node_num = max(i["token_type"].shape[0] for i in items)
    max_dist = max(i["edge_input"].size(-2) for i in items)
    energy = [i["energy"] for i in items]
    forces = torch.cat([pad_pos_unsqueeze(i["forces"], max_node_num) for i in items])

    energy = torch.cat(energy)
    x = torch.cat([pad_2d_unsqueeze(i["node_attr"], max_node_num) for i in items])

    edge_input = torch.cat(
        [
            pad_3d_unsqueeze(i["edge_input"], max_node_num, max_node_num, max_dist)
            for i in items
        ]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i["attn_bias"], max_node_num + 1) for i in items]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i["attn_edge_type"], max_node_num) for i in items]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i["spatial_pos"], max_node_num) for i in items]
    )
    in_degree = torch.cat(
        [pad_1d_unsqueeze(i["in_degree"], max_node_num) for i in items]
    )

    pos = torch.cat([pad_pos_unsqueeze(i["coords"], max_node_num) for i in items])

    pbc = torch.cat([i["pbc"].unsqueeze(0) for i in items], dim=0) if use_pbc else None
    cell = (
        torch.cat([i["cell"].unsqueeze(0) for i in items], dim=0) if use_pbc else None
    )
    num_atoms = torch.tensor([i["num_atoms"] for i in items]) if use_pbc else None

    node_type_edges = []
    for item in items:
        node_atom_type = item["token_type"]
        n_nodes = node_atom_type.shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)
        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    return dict(
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        token_id=x[:, :, 0],
        node_attr=x,
        edge_input=edge_input,
        energy=energy,
        forces=forces,
        pos=pos,
        node_type_edge=node_type_edge,
        pbc=pbc,
        cell=cell,
        num_atoms=num_atoms,
    )
