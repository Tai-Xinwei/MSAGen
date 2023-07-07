# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# import torch_geometric.datasets
# from ogb.graphproppred import PygGraphPropPredDataset
# from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from functools import lru_cache

import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
import os
import os.path as osp
import pickle
import shutil
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ogb.utils.features import (allowable_features,
                                atom_feature_vector_to_dict,
                                atom_to_feature_vector,
                                bond_feature_vector_to_dict,
                                bond_to_feature_vector)
# from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from . import algos

# import numpy as np


# import copy
# from itertools import repeat, product


# from memory_profiler import profile


def smiles2graph(smiles_string):
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

        graph = dict()
        graph["edge_index"] = edge_index
        graph["edge_feat"] = edge_attr
        graph["node_feat"] = x
        graph["num_nodes"] = len(x)
        graph["smile"] = smiles_string

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


class PygPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(
        self,
        root="dataset",
        smiles2graph=smiles2graph,
        transform=None,
        pre_transform=None,
    ):
        """
        Pytorch Geometric PCQM4Mv2 dataset object
            - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2graph requires rdkit to be installed
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "pcqm4m-v2")
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = (
            "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip"
        )

        # check version and update if necessary
        if osp.isdir(self.folder) and (
            not osp.exists(osp.join(self.folder, f"RELEASE_v{self.version}.txt"))
        ):
            print("PCQM4Mv2 dataset has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2Dataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "data.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, "data.csv.gz"))
        smiles_list = data_df["smiles"]
        homolumogap_list = data_df["homolumogap"]

        print("Converting SMILES strings into graphs...")
        data_list = []
        with Pool(processes=120) as pool:
            iter = pool.imap(smiles2graph, smiles_list)

            for i, graph in tqdm(enumerate(iter), total=len(homolumogap_list)):
                try:
                    data = Data()

                    homolumogap = homolumogap_list[i]

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
                    data.y = torch.Tensor([homolumogap])

                    data_list.append(data)
                except:
                    continue

        # double-check prediction target
        # split_dict = self.get_idx_split()
        # assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        # assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        # assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        # assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(
            torch.load(osp.join(self.root, "split_dict.pt"))
        )
        return split_dict


class PygPCQM4Mv2PosDataset(InMemoryDataset):
    def __init__(
        self,
        root="dataset",
        smiles2graph=smiles2graph,
        transform=None,
        pre_transform=None,
    ):
        """
        Pytorch Geometric PCQM4Mv2 dataset object
            - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2graph requires rdkit to be installed
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "pcqm4m-v2")
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = (
            "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip"
        )
        self.pos_url = (
            "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz"
        )

        # check version and update if necessary
        if osp.isdir(self.folder) and (
            not osp.exists(osp.join(self.folder, f"RELEASE_v{self.version}.txt"))
        ):
            print("PCQM4Mv2 dataset has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2PosDataset, self).__init__(
            self.folder, transform, pre_transform
        )

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "data.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print("Stop download.")
            exit(-1)

        if decide_download(self.pos_url):
            path = download_url(self.pos_url, self.original_root)
            tar = tarfile.open(path, "r:gz")
            filenames = tar.getnames()
            for file in filenames:
                tar.extract(file, self.original_root)
            tar.close()
            os.unlink(path)
        else:
            print("Stop download")
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, "data.csv.gz"))
        graph_pos_list = Chem.SDMolSupplier(
            osp.join(self.original_root, "pcqm4m-v2-train.sdf")
        )
        smiles_list = data_df["smiles"]
        homolumogap_list = data_df["homolumogap"]

        print("Converting SMILES strings into graphs...")
        data_list = []
        with Pool(processes=120) as pool:
            iter = pool.imap(smiles2graph, smiles_list)

            for i, graph in tqdm(enumerate(iter), total=len(homolumogap_list)):
                try:
                    data = Data()

                    homolumogap = homolumogap_list[i]

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
                    data.y = torch.Tensor([homolumogap])
                    data.pos = torch.zeros(data.__num_nodes__, 3).to(torch.float32)
                    # data.pos = torch.from_numpy(graph['pos']).to(torch.float32)

                    data_list.append(data)
                except:
                    continue

        print("Extracting 3D positions from SDF files for Training Data...")
        train_data_with_position_list = []
        with Pool(processes=120) as pool:
            iter = pool.imap(mol2graph, graph_pos_list)

            for i, graph in tqdm(enumerate(iter), total=len(graph_pos_list)):
                try:
                    data = Data()
                    homolumogap = homolumogap_list[i]

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
                    data.y = torch.Tensor([homolumogap])
                    data.pos = torch.from_numpy(graph["position"]).to(torch.float32)

                    train_data_with_position_list.append(data)
                except:
                    continue
        data_list = (
            train_data_with_position_list
            + data_list[len(train_data_with_position_list) :]
        )

        # double-check prediction target
        # split_dict = self.get_idx_split()
        # assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        # assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        # assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        # assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(
            torch.load(osp.join(self.root, "split_dict.pt"))
        )
        return split_dict


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


class MyPygPCQM4MDataset(PygPCQM4Mv2Dataset):
    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    def process(self):
        super(MyPygPCQM4MDataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_item(item)


class MyPygPCQM4MPosDataset(PygPCQM4Mv2PosDataset):
    def download(self):
        super(MyPygPCQM4MPosDataset, self).download()

    def process(self):
        super(MyPygPCQM4MPosDataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_item(item)


# @ Roger added


class PM6FullLMDBDataset(InMemoryDataset):
    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        max_node: Optional[int] = 512,
        multi_hop_max_dist: Optional[int] = 20,
        spatial_pos_max: Optional[int] = 20,
        mask_ratio: Optional[float] = 0.5,
    ):
        self.root = root
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

        # assert
        self.db_path_list = self.processed_dir
        self.smiles_db_path_list = self.smiles_db_dir
        for item in self.db_path_list:
            assert Path(item).exists(), f"{item}: No such file or directory"
        for item in self.smiles_db_path_list:
            assert Path(item).exists(), f"{item}: No such file or directory"

        self.env_list = [
            lmdb.Environment(
                item,
                map_size=(1024**3) * 256,
                subdir=True,
                readonly=True,
                readahead=False,
                meminit=False,
            ).begin()
            for item in self.db_path_list
        ]

        self.key_list = self.get_keys_list()
        self.len_list = [
            # item.stat()['entries']
            # for item in self.env_list
            len(item)
            for item in self.key_list
        ]

        self.cursor_list = [0] + np.cumsum(self.len_list).tolist()

        self.total_len = sum(self.len_list)
        super().__init__(root, transform, pre_transform, pre_filter)
        self._indices = range(self.total_len)
        # self.__indices__ = range(self.total_len)
        # self.data = Data()
        self.np = 4
        self.pool = Pool(self.np)
        self.mask_ratio = mask_ratio

    def _download(self):
        for item in self.raw_dir:
            assert Path(item).exists(), f"{item}: No such file or directory"
        return

    def _process(self):
        for item in self.processed_dir:
            assert Path(item).exists(), f"{item}: No such file or directory"
        return

    def get_keys_list(self):
        msg_list = [
            lmdb.Environment(
                item,
                map_size=(1024**3) * 256,
                subdir=True,
                readonly=True,
                readahead=False,
                meminit=False,
                # ) for item in self.msg_dir
            )
            for item in self.smiles_db_path_list
        ]
        # key_dir_list = [osp.join(item, 'key_list.pt') for item in self.msg_dir]
        key_dir_list = [
            osp.join(item, "key_list.pt") for item in self.smiles_db_path_list
        ]
        key_list = []
        for i, item in enumerate(key_dir_list):
            begin_time = time.perf_counter()
            local_cursor_list = msg_list[i].begin().cursor()
            local_key_list = [k for k, _ in local_cursor_list]
            key_list.append(local_key_list)
            end_time = time.perf_counter()
            print(
                f'Loaded key_list for {item.split("/")[-2]}; time: {end_time - begin_time} s'
            )
        return key_list

    @property
    def raw_dir(self) -> List[str]:
        return [
            osp.join(osp.join(f"{self.root}", "merged"), "S0-msg"),
            # osp.join(osp.join(f"{self.root}", "merged"), "T0-msg"),
            # osp.join(osp.join(f"{self.root}", "merged"), "anion-msg"),
            # osp.join(osp.join(f"{self.root}", "merged"), "cation-msg"),
        ]

    @property
    def processed_dir(self) -> List[str]:
        return [
            osp.join(osp.join(f"{self.root}", "merged"), "S0"),
            # osp.join(osp.join(f"{self.root}", "merged"), "T0"),
            # osp.join(osp.join(f"{self.root}", "merged"), "anion"),
            # osp.join(osp.join(f"{self.root}", "merged"), "cation"),
        ]

    @property
    def msg_dir(self) -> List[str]:
        return [
            osp.join(osp.join(f"{self.root}", "merged"), "S0-msg"),
            # osp.join(osp.join(f"{self.root}", "merged"), "T0-msg"),
            # osp.join(osp.join(f"{self.root}", "merged"), "anion-msg"),
            # osp.join(osp.join(f"{self.root}", "merged"), "cation-msg"),
        ]

    @property
    def smiles_db_dir(self) -> List[str]:
        return [osp.join(osp.join(f"{self.root}", "merged"), "S0-smiles")]

    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join(
                osp.join(osp.join(f"{self.root}", "merged"), "S0-msg"), "data.mdb"
            ),
            # osp.join(osp.join(osp.join(f"{self.root}", "merged"), "T0-msg"), "data.mdb"),
            # osp.join(osp.join(osp.join(f"{self.root}", "merged"), "anion-msg"), "data.mdb"),
            # osp.join(osp.join(osp.join(f"{self.root}", "merged"), "cation-msg"), "data.mdb"),
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return [
            osp.join(osp.join(osp.join(f"{self.root}", "merged"), "S0"), "data.mdb"),
            # osp.join(osp.join(osp.join(f"{self.root}", "merged"), "T0"), "data.mdb"),
            # osp.join(osp.join(osp.join(f"{self.root}", "merged"), "anion"), "data.mdb"),
            # osp.join(osp.join(osp.join(f"{self.root}", "merged"), "cation"), "data.mdb"),
        ]

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(
            torch.load(osp.join(self.root, "split_dict.pt"))
        )
        return split_dict

    @lru_cache(maxsize=16)
    def __getitem__(self, idx: Union[int, np.integer]) -> Data:
        """
        data.__num_nodes__
        data.edge_index
        data.edge_attr
        data.x
        data.mulliken
        data.statez
        data.pos
        """
        # idxs = np.random.choice(self.__len__, self.np, replace=False)
        # return self.pool.map(self.generator, idxs)
        cidx = self.indices()[idx]
        cursor_idx = 0
        for i in range(len(self.cursor_list)):
            if cidx >= self.cursor_list[i] and cidx < self.cursor_list[i + 1]:
                cursor_idx = i
                break
        cur_env = self.env_list[cursor_idx]
        cur_idx = int(cidx - self.cursor_list[cursor_idx])
        ori_data = pickle.loads(cur_env.get(self.key_list[cursor_idx][cur_idx]))

        # return 0
        try:
            assert len(ori_data["edge_feat"]) == ori_data["edge_index"].shape[1]
            assert len(ori_data["node_feat"]) == ori_data["num_nodes"]
        except:
            ori_data = pickle.loads(cur_env.get(self.key_list[cursor_idx][0]))
            print(f"Error Index: idx {idx}; cidx {cidx}.")

        data = Data()
        data.__num_nodes__ = int(ori_data["num_nodes"])
        data.edge_index = ori_data["edge_index"].to(torch.int64)
        data.edge_attr = ori_data["edge_feat"].to(torch.int64)
        data.x = ori_data["node_feat"].to(torch.int64)
        data.y = torch.tensor([ori_data["gap"]]).to(torch.float32)
        data.pos = ori_data["atom_coords"].to(torch.float32)
        data.idx = idx

        return preprocess_item(data, self.mask_ratio)

    def __len__(self) -> int:
        if self._indices is None:
            return self.total_len
        else:
            return len(self._indices)

    # def collater(self, item):
    #     return collator_pcq(item,
    #         max_node=self.max_node,
    #         multi_hop_max_dist=self.multi_hop_max_dist,
    #         spatial_pos_max=self.spatial_pos_max)

    # def collater(self, edge_index, edge_attr, x, y, pos, idx):
    #     return collator_pcq(edge_index, edge_attr, x, y, pos, idx,
    #         max_node=self.max_node,
    #         multi_hop_max_dist=self.multi_hop_max_dist,
    #         spatial_pos_max=self.spatial_pos_max)


if __name__ == "__main__":
    dataset = PygPCQM4Mv2PosDataset()
    print(len(dataset))
