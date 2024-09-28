# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_1d_confidence_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = -1 * x.new_ones([padlen], dtype=x.dtype)
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


def pad_adj_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(False)
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = False
        x = new_x
    return x.unsqueeze(0)


def pad_attn_edge_input_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    num_features = x.size(-1)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, num_features], dtype=x.dtype).fill_(0)
        new_x[:xlen, :xlen, :] = x
        new_x[xlen:, :xlen, :] = 0
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


def pad_edge_info_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def collate_fn(
    items,
    multi_hop_max_dist=20,
    use_pbc=True,
    preprocess_2d_bond_features_with_cuda=True,
    sample_in_validation: bool = False,
):  # unify the data format
    # include the following fields: sample_type, token_type, idx, coords, cell, pbc, stress, forces, energy
    # need to add: node_type_edge, edge_input, in_degree, attn_bias, spatial_pos

    for item in items:
        if "pbc" not in item:
            item["pbc"] = torch.tensor([False, False, False])
        if "cell" not in item:
            item["cell"] = torch.zeros([3, 3])
        if "num_atoms" not in item:
            item["num_atoms"] = item["x"].size()[0]
        if not preprocess_2d_bond_features_with_cuda:
            item["edge_input"] = item["edge_input"][:, :, :multi_hop_max_dist, :]
        if "position_ids" not in item:
            item["position_ids"] = torch.arange(
                0, item["token_type"].shape[0], dtype=torch.long
            )
        if "confidence" not in item:
            item["confidence"] = -2 * torch.ones(item["token_type"].shape[0])

    idx = torch.tensor([i["idx"] for i in items], dtype=torch.long)
    sample_type = torch.tensor([i["sample_type"] for i in items], dtype=torch.long)
    max_node_num = max(i["token_type"].shape[0] for i in items)
    energy = [i["energy"] for i in items]
    energy_per_atom = [i["energy_per_atom"] for i in items]
    forces = torch.cat([pad_pos_unsqueeze(i["forces"], max_node_num) for i in items])
    energy = torch.cat(energy)
    has_energy = torch.cat([i["has_energy"] for i in items], dim=0)
    has_forces = torch.cat([i["has_forces"] for i in items], dim=0)
    energy_per_atom = torch.cat(energy_per_atom)

    x = torch.cat([pad_2d_unsqueeze(i["node_attr"], max_node_num) for i in items])
    position_ids = torch.cat(
        [pad_1d_unsqueeze(i["position_ids"], max_node_num) for i in items]
    )

    confidence = torch.cat(
        [pad_1d_confidence_unsqueeze(i["confidence"], max_node_num) for i in items]
    )

    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i["attn_bias"], max_node_num + 1) for i in items]
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

    if preprocess_2d_bond_features_with_cuda:
        adj = torch.cat(
            [pad_attn_bias_unsqueeze(i["adj"], max_node_num) for i in items]
        )
        attn_edge_type = torch.cat(
            [
                pad_attn_edge_input_unsqueeze(i["attn_edge_type"], max_node_num)
                for i in items
            ]
        )
    else:
        max_dist = max(i["edge_input"].size(-2) for i in items)
        edge_input = torch.cat(
            [
                pad_3d_unsqueeze(i["edge_input"], max_node_num, max_node_num, max_dist)
                for i in items
            ]
        )
        spatial_pos = torch.cat(
            [pad_spatial_pos_unsqueeze(i["spatial_pos"], max_node_num) for i in items]
        )

    if (
        sample_in_validation
        and "edge_attr" in items[0]
        and items[0]["edge_attr"] is not None
    ):
        # add original edge information to recover the molecule
        max_num_edges = max(i["edge_attr"].size()[0] for i in items)
        edge_attr = torch.cat(
            [pad_edge_info_unsqueeze(i["edge_attr"], max_num_edges) for i in items]
        )
        edge_index = torch.cat(
            [pad_edge_info_unsqueeze(i["edge_index"].T, max_num_edges) for i in items]
        )
        num_edges = torch.tensor(
            [int(i["edge_attr"].size()[0]) for i in items], dtype=torch.long
        )
        idx = torch.tensor([int(i["idx"]) for i in items], dtype=torch.long)

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

    is_stable_periodic = torch.tensor(
        [("is_stable_periodic" in i) and i["is_stable_periodic"] for i in items],
        dtype=torch.bool,
    )

    batched_data = dict(
        idx=idx,
        sample_type=sample_type,
        attn_bias=attn_bias,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        token_id=x[:, :, 0],
        node_attr=x,
        energy=energy,
        energy_per_atom=energy_per_atom,
        forces=forces,
        has_energy=has_energy,
        has_forces=has_forces,
        pos=pos,
        node_type_edge=node_type_edge,
        pbc=pbc,
        cell=cell,
        num_atoms=num_atoms,
        is_stable_periodic=is_stable_periodic,
        position_ids=position_ids,
        confidence=confidence,
    )

    if preprocess_2d_bond_features_with_cuda:
        batched_data.update(
            dict(
                adj=adj,
                attn_edge_type=attn_edge_type,
            )
        )
    else:
        batched_data.update(
            dict(
                spatial_pos=spatial_pos,
                edge_input=edge_input,
            )
        )

    if (
        sample_in_validation
        and "edge_attr" in items[0]
        and items[0]["edge_attr"] is not None
    ):
        batched_data.update(
            dict(
                edge_attr=edge_attr, edge_index=edge_index, num_edges=num_edges, idx=idx
            )
        )
        if "key" in items[0]:
            batched_data["key"] = [i["key"] for i in items]
    return batched_data
