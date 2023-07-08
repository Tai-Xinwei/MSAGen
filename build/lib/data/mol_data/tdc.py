# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from functools import lru_cache

import numpy as np
import pyximport
import torch_geometric.datasets
from ogb.graphproppred import PygGraphPropPredDataset

pyximport.install(setup_args={"include_dirs": np.get_include()})
import copy

# from memory_profiler import profile
import json
import os
import os.path as osp
import pickle
import shutil
import tarfile
import time
from itertools import product, repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ogb.utils.features import (
    allowable_features,
    atom_feature_vector_to_dict,
    atom_to_feature_vector,
    bond_feature_vector_to_dict,
    bond_to_feature_vector,
)

# from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
from rdkit import Chem

# import numpy as np
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import rdDistGeom as molDG
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from . import algos


def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)

        fp = [x for x in MACCSkeys.GenMACCSKeys(mol)]
        fp = np.array(fp, dtype=np.int64)

        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype=np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        rdk_ds = get_rdk_descriptors(smiles_string)

        rdk_ds = np.concatenate((rdk_ds, fp), axis=0)

        graph = dict()
        graph["edge_index"] = edge_index
        graph["edge_feat"] = edge_attr
        graph["node_feat"] = x
        graph["num_nodes"] = len(x)
        graph["smile"] = smiles_string
        graph["rdk_ds"] = rdk_ds
        # graph['fp'] = fp

        return graph
    except:
        return None


def smiles2graphpos(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)

        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype=np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        try:
            mol_add_h = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_add_h)
            mol_r_h = Chem.RemoveHs(mol_add_h)
            pos = mol_r_h.GetConformer().GetPositions()
        except:
            pos = np.random.rand(x.shape[0], 3)

        graph = dict()
        graph["edge_index"] = edge_index
        graph["edge_feat"] = edge_attr
        graph["node_feat"] = x
        graph["num_nodes"] = len(x)
        graph["pos"] = pos

        return graph
    except:
        return None


def mol2graph(mol):
    try:
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype=np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        # positions
        positions = mol.GetConformer().GetPositions()

        graph = dict()
        graph["edge_index"] = edge_index
        graph["edge_feat"] = edge_attr
        graph["node_feat"] = x
        graph["num_nodes"] = len(x)
        graph["position"] = positions

        return graph
    except:
        return None


class TDCDataset(InMemoryDataset):
    def __init__(
        self, data, smiles2graph=smiles2graph, transform=None, pre_transform=None
    ):
        # self.original_root = root
        self.smiles2graph = smiles2graph
        self.smiles2graphpos = smiles2graphpos
        self.data_df = data
        self.version = 1
        self._indices = None
        self.dataset_name = "TDC"
        self.pre_transform = pre_transform
        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        # self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        # if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
        # print('PCQM4Mv2 dataset has been updated.')
        # if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
        # shutil.rmtree(self.folder)

        # super(TDCDataset, self).__init__('../tdc_data', transform=None, pre_transform=None)
        self.std = 1.0
        self.mean = 0.0
        self.data = self.preprocess()
        self.len = len(self.data)

        # self.data, self.slices = torch.load(self.processed_paths[0])

    # @property
    # def raw_file_names(self):
    #     return 'data.csv.gz'

    # @property
    # def processed_file_names(self):
    #     return ['geometric_data_processed.pt']

    def preprocess(self):
        data_df = self.data_df
        smiles_list = data_df["Drug"]
        label_list = data_df["Y"]

        self.mean = np.mean(label_list)
        self.std = np.std(label_list)

        print("Converting SMILES strings into graphs...")
        data_list = []

        with Pool(processes=120) as pool:
            # iter = pool.imap(smiles2graphpos, smiles_list)
            iter = pool.imap(smiles2graph, smiles_list)

            for i, graph in enumerate(iter):
                try:
                    data = Data()
                    # graph = self.smiles2graph(smile)
                    # graph = self.smiles2graphpos(smile)
                    y = label_list[i]

                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]

                    data.__num_nodes__ = int(graph["num_nodes"])
                    data.edge_index = torch.from_numpy(graph["edge_index"]).to(
                        torch.int64
                    )
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(
                        torch.int64
                    )
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.idx = i
                    # data.pos = torch.from_numpy(graph['pos'])
                    data.ds = torch.from_numpy(graph["rdk_ds"])

                    # print(data.ds.siz())
                    data_list.append(data)
                except:
                    continue

        return data_list

    def __len__(self) -> int:
        return self.len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.data[idx]
        return preprocess_item(item)


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


# @profile(precision=4, stream=open('/home/peiran/MFM_DS/memory_profiler.log','w+'))
def preprocess_item(item, mask_ratio=0.5):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index.to(torch.int64), item.x

    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())

    # spd_count
    # shortest_path_count = algos.all_shortest_path_count_custom(adj.long().numpy(), shortest_path_result)
    # spatial_pos_count = torch.from_numpy((shortest_path_count)).long()

    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    # full path information
    # edge_input, node_input = algos.gen_edge_input_with_node(max_dist, path, attn_edge_type.numpy(), x.numpy())

    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # @ Roger added: mask
    mask_N = int(N * mask_ratio)
    mask_idx = torch.from_numpy(np.random.choice(N, mask_N, replace=False))

    node_mask = torch.zeros(N).float()
    node_mask[mask_idx] = 1.0

    # in_degree = adj.long().sum(dim=1).view(-1)
    # edge_input = torch.from_numpy(edge_input).long()
    # pos = pos - pos.mean(dim=0, keepdim=True)
    # node_mask = node_mask.unsqueeze(-1)
    # return  {
    #         "edge_index": edge_index,
    #         "edge_attr": edge_attr,
    #         "x": x,
    #         "y": y,
    #         "pos": pos,
    #         "idx": idx,
    #         "attn_bias": attn_bias,
    #         "attn_edge_type": attn_edge_type,
    #         "spatial_pos": spatial_pos,
    #         "in_degree": in_degree,
    #         "out_degree": in_degree,
    #         "edge_input": edge_input,
    #         "node_mask": node_mask,
    #         }
    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()

    # # @ shengjie added: mean centered
    if item.pos is not None:
        item.pos = item.pos - item.pos.mean(dim=0, keepdim=True)

    # @ Roger added
    item.node_mask = node_mask.unsqueeze(-1)

    return item


from rdkit.Chem import rdMolDescriptors


def get_rdk_descriptors(smile):
    descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
    get_descriptors = rdMolDescriptors.Properties(descriptor_names)
    # desps = []
    mol = Chem.MolFromSmiles(smile)
    descriptors = np.array(get_descriptors.ComputeProperties(mol))
    return descriptors
