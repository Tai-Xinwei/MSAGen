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
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
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


def collator(
    items,
    min_node=-1,
    max_node=512,
    multi_hop_max_dist=20,
    spatial_pos_max=20,
    infer=False,
    use_pbc=False,
):
    # return 0
    items = [
        item
        for item in items
        if item is not None and item.x.size(0) <= max_node and item.x.size(0) > min_node
    ]

    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            item.__num_nodes__,
            # spd_count, full_path_information
            # item.spatial_pos_count, item.node_input[:, :, :multi_hop_max_dist+1, :]
        )
        for item in items
    ]

    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        nnodes,
    ) = zip(*items)

    # spd_count, full_path_information
    # idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys, spatial_pos_counts, node_inputs = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)

    y = torch.cat(ys)
    nnodes = torch.cat(nnodes)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,
        pos=None,
        node_type_edge=None,
        node_mask=None,
        nnodes=nnodes,
    )


# @profile(precision=4, stream=open('/home/peiran/MFM_DS/memory_profiler.log','w+'))
def collator_3d(
    items,
    min_node=-1,
    max_node=512,
    multi_hop_max_dist=20,
    spatial_pos_max=20,
    infer=False,
    use_pbc=False,
):
    items = [
        item
        for item in items
        if item is not None and item.x.size(0) <= max_node and item.x.size(0) > min_node
    ]

    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            item.pos,
            item.node_mask,
            (item.pbc if hasattr(item, "pbc") else torch.tensor([False, False, False]))
            if use_pbc
            else None,
            (item.cell if hasattr(item, "cell") else torch.zeros([3, 3]))
            if use_pbc
            else None,
            (int(item.num_atoms) if hasattr(item, "num_atoms") else item.x.size()[0]),
        )
        for item in items
    ]

    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        poses,
        node_masks,
        pbcs,
        cells,
        natoms,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")

    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])
    node_mask = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in node_masks])
    pbc = torch.cat([i.unsqueeze(0) for i in pbcs], dim=0) if use_pbc else None
    cell = torch.cat([i.unsqueeze(0) for i in cells], dim=0) if use_pbc else None
    natoms = torch.tensor(natoms) if use_pbc else None

    # @ Roger added
    node_type_edges = []
    for idx in range(len(items)):
        node_atom_type = items[idx][6][:, 0]
        n_nodes = items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)
        node_type_edges.append(node_atom_edge.long())

    node_type_edge = torch.cat(node_type_edges)

    return dict(
        idx=torch.LongTensor(idxs),
        # idx=idx_n,
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,
        pos=pos,
        node_type_edge=node_type_edge,
        node_mask=node_mask,
        pbc=pbc,
        cell=cell,
        natoms=natoms,
    )


def collator_3d_pp(
    items,
    min_node=-1,
    max_node=512,
    multi_hop_max_dist=20,
    spatial_pos_max=20,
    infer=False,
    use_pbc=False,
):
    items = [
        item
        for item in items
        if item is not None and item.x.size(0) <= max_node and item.x.size(0) > min_node
    ]

    # if not infer:
    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            item.pos,
            item.node_mask,
            (item.pbc if hasattr(item, "pbc") else torch.tensor([False, False, False]))
            if use_pbc
            else None,
            (item.cell if hasattr(item, "cell") else torch.zeros([3, 3]))
            if use_pbc
            else None,
            (int(item.num_atoms) if hasattr(item, "num_atoms") else item.x.size()[0])
            # spd_count, full_path_information
            # item.spatial_pos_count, item.node_input[:, :, :multi_hop_max_dist+1, :]
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        poses,
        node_masks,
        pbcs,
        cells,
        natoms,
    ) = zip(*items)
    # else:
    #     items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
    #             item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y, item.pos, item.node_mask, item.smiles
    #             # spd_count, full_path_information
    #             # item.spatial_pos_count, item.node_input[:, :, :multi_hop_max_dist+1, :]
    #             ) for item in items]
    #     idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys, poses, node_masks, smiles = zip(*items)

    # spd_count, full_path_information
    # idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys, spatial_pos_counts, node_inputs = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    # max_node_num = 80
    max_dist = max(i.size(-2) for i in edge_inputs)
    if not infer:
        y = torch.cat(ys)
    else:
        y = None
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    if not infer:
        pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])
    else:
        pos = None

    node_mask = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in node_masks])
    pbc = torch.cat([i.unsqueeze(0) for i in pbcs], dim=0) if use_pbc else None
    cell = torch.cat([i.unsqueeze(0) for i in cells], dim=0) if use_pbc else None
    natoms = torch.tensor(natoms) if use_pbc else None
    # @ Roger added

    node_type_edges = []
    for idx in range(len(items)):
        node_atom_type = items[idx][6][:, 0]
        n_nodes = items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    # spd_count, full_path_information
    # node_input = torch.cat([pad_3d_unsqueeze(
    #     i, max_node_num, max_node_num, max_dist + 1) for i in node_inputs])
    # spatial_pos_count = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
    #                                for i in spatial_pos_counts])

    if not infer:
        intput = [
            torch.LongTensor(idxs),  # 0
            # attn_bias,
            attn_bias,  # 1
            attn_edge_type,  # 2
            spatial_pos,  # 3
            in_degree,  # 4
            in_degree,  # 5 for undirected graph
            x,  # 6
            edge_input,  # 7
            y,  # 8
            pos,  # 9
            node_type_edge,  # 10
            node_mask  # 11
            # spd_count, full_path_information
            # spatial_pos_count=spatial_pos_count,
            # node_input=node_input,
        ]
        if use_pbc:
            intput.append(pbc)
            intput.append(cell)
            intput.append(natoms)
        label = [
            x,
            pos,
            node_mask,
        ]
    else:
        intput = [
            torch.LongTensor(idxs),  # 0
            # attn_bias,
            attn_bias,  # 1
            attn_edge_type,  # 2
            spatial_pos,  # 3
            in_degree,  # 4
            in_degree,  # 5 for undirected graph
            x,  # 6
            edge_input,  # 7
            x,  # 8
            x,  # 9
            node_type_edge,  # 10
            x  # 11
            # spd_count, full_path_information
            # spatial_pos_count=spatial_pos_count,
            # node_input=node_input,
        ]
        if use_pbc:
            intput.append(pbc)
            intput.append(cell)
            intput.append(natoms)
        label = [
            torch.LongTensor(idxs),
            x,
            # x,
        ]

    intput = tuple(intput)
    label = tuple(label)

    return (intput, label)


def collator_ft(
    items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20, use_pbc=False
):
    original_len = len(items)
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    filtered_len = len(items)
    if filtered_len < original_len:
        pass
        # print("warning: molecules with atoms more than %d are filtered" % max_node)
    pos = None
    head = None
    cofeat = None
    max_node_num = max(item.x.size(0) for item in items if item is not None)
    forces = None

    if hasattr(items[0], "pos") and items[0].pos is not None:
        poses = [item.pos - item.pos.mean(dim=0, keepdim=True) for item in items]
        # poses = [item.pos for item in items]
        pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])
    if hasattr(items[0], "forces") and items[0].forces is not None:
        forcess = [item.forces for item in items]
        forces = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in forcess])
    if hasattr(items[0], "head") and items[0].head is not None:
        head = torch.cat([item.head for item in items])
    if hasattr(items[0], "cofeat") and items[0].cofeat is not None:
        cofeat = torch.cat([item.cofeat.unsqueeze(0) for item in items])

    # @ Roger added
    num_node = None
    node_type_edge = None
    if hasattr(items[0], "num_node") and items[0].num_node is not None:
        num_nodes = [item.num_node for item in items]
        num_node = torch.stack(num_nodes)

        num_ligand_nodes = [int(item.num_node[0]) for item in items]
        num_protein_nodes = [int(item.num_node[1]) for item in items]
        node_type_edges = []
        for idx in range(len(num_ligand_nodes)):
            n_nodes, n_ligand_nodes = (
                num_ligand_nodes[idx] + num_protein_nodes[idx],
                num_ligand_nodes[idx],
            )
            node_type_edge = torch.zeros(n_nodes, n_nodes)
            node_type_edge[:n_ligand_nodes, :n_ligand_nodes] = 3
            node_type_edge[n_ligand_nodes:, :n_ligand_nodes] = 4
            node_type_edge[:n_ligand_nodes, n_ligand_nodes:] = 5
            node_type_edge[n_ligand_nodes:, n_ligand_nodes:] = 6

            node_atom_type = items[idx].x[:, 0]
            node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
            node_atom_i = pad_spatial_pos_unsqueeze(
                node_atom_i, max_node_num
            ).unsqueeze(-1)
            node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
            node_atom_j = pad_spatial_pos_unsqueeze(
                node_atom_j, max_node_num
            ).unsqueeze(-1)
            node_type_edge = pad_spatial_pos_unsqueeze(
                node_type_edge, max_node_num
            ).unsqueeze(-1)
            node_atom_edge = torch.cat(
                [node_type_edge, node_atom_i, node_atom_j], dim=-1
            )
            node_atom_edge = convert_to_single_emb(node_atom_edge)

            node_type_edges.append(node_atom_edge.long())
        node_type_edge = torch.cat(node_type_edges)

    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            (item.pbc if hasattr(item, "pbc") else torch.tensor([False, False, False]))
            if use_pbc
            else None,
            (item.cell if hasattr(item, "cell") else torch.zeros([3, 3]))
            if use_pbc
            else None,
            (int(item.num_atoms) if hasattr(item, "num_atoms") else item.x.size()[0]),
            (int(item.num_atoms_in_cell) if use_pbc else None),
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        pbcs,
        cells,
        natoms,
        num_atoms_in_cells,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    pbc = torch.cat([i.unsqueeze(0) for i in pbcs], dim=0) if use_pbc else None
    cell = torch.cat([i.unsqueeze(0) for i in cells], dim=0) if use_pbc else None
    natoms = torch.tensor(natoms) if use_pbc else None
    num_atoms_in_cell = torch.tensor(num_atoms_in_cells) if use_pbc else None

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        token_id=x[:, :, 0],
        edge_input=edge_input,
        y=y,
        pos=pos,
        head=head,
        cofeat=cofeat,
        num_node=num_node,
        node_type_edge=node_type_edge,
        pbc=pbc,
        cell=cell,
        natoms=natoms,
        num_atoms_in_cell=num_atoms_in_cell,
        forces=forces,
    )


def collator_copilot(
    items,
    min_node=-1,
    max_node=512,
    multi_hop_max_dist=20,
    spatial_pos_max=20,
    infer=False,
):
    items = [
        item
        for item in items
        if item is not None and item.x.size(0) <= max_node and item.x.size(0) > min_node
    ]

    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            item.pos,
            item.node_mask,
            item.input_ids,
            item.target_ids,
            item.llm_mask,
            # spd_count, full_path_information
            # item.spatial_pos_count, item.node_input[:, :, :multi_hop_max_dist+1, :]
        )
        for item in items
    ]

    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        poses,
        node_masks,
        input_ids,
        target_ids,
        llm_masks,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    # max_node_num = 80
    max_dist = max(i.size(-2) for i in edge_inputs)
    # if not infer:
    #     y = torch.cat(ys)
    # else:
    #     y = None
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    input_ids = torch.cat([i.unsqueeze(0) for i in input_ids])
    target_ids = torch.cat([i.unsqueeze(0) for i in target_ids])
    llm_mask = torch.cat([i.unsqueeze(0) for i in llm_masks])

    # if not infer:
    #     pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])
    # else:
    #     pos = None

    # node_mask = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in node_masks])

    node_type_edges = []
    for idx in range(len(items)):
        node_atom_type = items[idx][6][:, 0]
        n_nodes = items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    intput = [
        torch.LongTensor(idxs),  # 0
        attn_bias,  # 1
        attn_edge_type,  # 2
        spatial_pos,  # 3
        in_degree,  # 4
        in_degree,  # 5 for undirected graph
        x,  # 6
        edge_input,  # 7
        x,  # 8
        x,  # 9
        node_type_edge,  # 10
        x,  # 11
        input_ids,  # 12
        llm_mask,  # 13
        # spd_count, full_path_information
        # spatial_pos_count=spatial_pos_count,
        # node_input=node_input,
    ]
    label = [
        target_ids,
        input_ids,
        torch.LongTensor(idxs),
    ]

    intput = tuple(intput)
    label = tuple(label)

    return (intput, label)


def collator_copilot_multi_mol_PP(
    items,
    min_node=-1,
    max_node=512,
    multi_hop_max_dist=20,
    spatial_pos_max=20,
    infer=False,
    pad_token_id=32000,
):
    mol_items = []
    num_mol_offsets = []
    for item in items:
        processed_mol_data = item["processed_mol_data"]
        num_mol_offsets.append(len(mol_items))
        mol_items.extend(processed_mol_data)
    num_mol_offsets.append(len(mol_items))
    num_mol_offsets = torch.tensor(num_mol_offsets)

    mol_items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            item.pos,
            item.node_mask,
            # spd_count, full_path_information
            # item.spatial_pos_count, item.node_input[:, :, :multi_hop_max_dist+1, :]
        )
        for item in mol_items
    ]

    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        poses,
        node_masks,
    ) = zip(*mol_items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    # max_node_num = 80
    max_dist = max(i.size(-2) for i in edge_inputs)
    # if not infer:
    #     y = torch.cat(ys)
    # else:
    #     y = None
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    input_ids = [item["input_ids"] for item in items]
    labels = [item["labels"] for item in items]
    llm_mask = [item["llm_mask"] for item in items]

    max_seq_len = max(len(i) for i in input_ids)
    input_ids = torch.cat(
        [
            torch.cat(
                [i, torch.ones(max_seq_len - len(i), dtype=i.dtype).fill_(pad_token_id)]
            ).unsqueeze(0)
            for i in input_ids
        ]
    )
    labels = torch.cat(
        [
            torch.cat(
                [i, torch.ones(max_seq_len - len(i), dtype=i.dtype).fill_(-100)]
            ).unsqueeze(0)
            for i in labels
        ]
    )
    llm_mask = torch.cat([i.ne(pad_token_id).unsqueeze(0) for i in input_ids])

    # if not infer:
    #     pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])
    # else:
    #     pos = None

    # node_mask = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in node_masks])

    node_type_edges = []
    for idx in range(len(mol_items)):
        node_atom_type = mol_items[idx][6][:, 0]
        n_nodes = mol_items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    intput = [
        torch.LongTensor(idxs),  # 0
        attn_bias,  # 1
        attn_edge_type,  # 2
        spatial_pos,  # 3
        in_degree,  # 4
        in_degree,  # 5 for undirected graph
        x,  # 6
        edge_input,  # 7
        x,  # 8
        x,  # 9
        node_type_edge,  # 10
        x,  # 11
        input_ids,  # 12
        llm_mask,  # 13
        # num_mol_offsets, #14
        # spd_count, full_path_information
        # spatial_pos_count=spatial_pos_count,
        # node_input=node_input,
    ]
    label = [
        labels,
        input_ids,
        torch.LongTensor(idxs),
    ]

    intput = tuple(intput)
    label = tuple(label)

    return (intput, label)


def collator_copilot_multi_mol(
    items,
    min_node=-1,
    max_node=512,
    multi_hop_max_dist=20,
    spatial_pos_max=20,
    infer=False,
    pad_token_id=32000,
):
    mol_items = []
    num_mol_offsets = []
    for item in items:
        processed_mol_data = item["processed_mol_data"]
        num_mol_offsets.append(len(mol_items))
        mol_items.extend(processed_mol_data)
    num_mol_offsets.append(len(mol_items))
    num_mol_offsets = torch.tensor(num_mol_offsets)

    mol_items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            item.pos,
            item.node_mask,
        )
        for item in mol_items
    ]

    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        poses,
        node_masks,
    ) = zip(*mol_items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    # max_node_num = 80
    max_dist = max(i.size(-2) for i in edge_inputs)

    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    input_ids = [item["input_ids"] for item in items]
    labels = [item["labels"] for item in items]
    llm_mask = [item["llm_mask"] for item in items]

    max_seq_len = max(len(i) for i in input_ids)
    input_ids = torch.cat(
        [
            torch.cat(
                [i, torch.ones(max_seq_len - len(i), dtype=i.dtype).fill_(pad_token_id)]
            ).unsqueeze(0)
            for i in input_ids
        ]
    )
    labels = torch.cat(
        [
            torch.cat(
                [i, torch.ones(max_seq_len - len(i), dtype=i.dtype).fill_(-100)]
            ).unsqueeze(0)
            for i in labels
        ]
    )
    llm_mask = torch.cat([i.ne(pad_token_id).unsqueeze(0) for i in input_ids])

    node_type_edges = []
    for idx in range(len(mol_items)):
        node_atom_type = mol_items[idx][6][:, 0]
        n_nodes = mol_items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
        node_type_edge=node_type_edge,
        input_ids=input_ids,
        labels=labels,
        llm_mask=llm_mask,
    )
