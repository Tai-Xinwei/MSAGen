# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
import os
import os.path as osp
import pickle
import pickle as pkl
import random
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import lmdb
import numpy as np
import torch
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from sfm.data.prot_data.dataset import LMDBDataset
from sfm.data.prot_data.sequence_masking import masking_registry
from sfm.data.prot_data.spatial_noise import noise_registry
from sfm.data.prot_data.util import bstr2obj


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

        self.env_list = None

        self.key_list = self.get_keys_list()
        self.len_list = [len(item) for item in self.key_list]

        self.cursor_list = [0] + np.cumsum(self.len_list).tolist()

        self.total_len = sum(self.len_list)
        super().__init__(root, transform, pre_transform, pre_filter)
        self._indices = range(self.total_len)
        # self.__indices__ = range(self.total_len)
        # self.data = Data()

        self.mask_ratio = mask_ratio

    def _download(self):
        for item in self.raw_dir:
            assert Path(item).exists(), f"{item}: No such file or directory"
        return

    def _process(self):
        for item in self.processed_dir:
            assert Path(item).exists(), f"{item}: No such file or directory"
        return

    def init_env_list(self):
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
        ]

    @property
    def processed_dir(self) -> List[str]:
        return [
            osp.join(osp.join(f"{self.root}", "merged"), "S0"),
        ]

    @property
    def msg_dir(self) -> List[str]:
        return [
            osp.join(osp.join(f"{self.root}", "merged"), "S0-msg"),
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
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return [
            osp.join(osp.join(osp.join(f"{self.root}", "merged"), "S0"), "data.mdb"),
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
        cidx = self.indices()[idx]
        cursor_idx = 0
        for i in range(len(self.cursor_list)):
            if cidx >= self.cursor_list[i] and cidx < self.cursor_list[i + 1]:
                cursor_idx = i
                break

        if self.env_list is None:
            self.init_env_list()
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


class MatterSimDataset:
    def __init__(self, data_path, split=None):
        self.data_name_to_lmdb = {}
        self.data_name_to_txn = {}
        self.index_to_dataset_name = []
        self.data_path = data_path
        for path_name in os.listdir(self.data_path):
            if os.path.isdir(f"{self.data_path}/{path_name}"):
                if split is None:
                    lmdb_path = f"{self.data_path}/{path_name}"
                else:
                    lmdb_path = f"{self.data_path}/{path_name}/{split}"
                self.data_name_to_lmdb[path_name] = lmdb.open(lmdb_path)
                self.data_name_to_txn[path_name] = self.data_name_to_lmdb[
                    path_name
                ].begin(write=False)
                for key, _ in tqdm(self.data_name_to_txn[path_name].cursor()):
                    self.index_to_dataset_name.append([path_name, key.decode()])

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data_name, key = self.index_to_dataset_name[idx]
        return preprocess_item(
            pkl.loads(self.data_name_to_txn[data_name].get(key.encode())), idx
        )

    def __len__(self):
        return len(self.index_to_dataset_name)


class ProteinLMDBDataset(LMDBDataset):
    """
    This is a dataset for protein information, including amino acid, position, angles and confidence score.
    All the information are raw data. Please ues other dataset to process the data, eg, tokenize, encode...

    The process pipeline will be changed in the future, but the interface will not change.
    """

    def __init__(self, args: Any) -> None:
        # here calls self.set_default_args(args)
        super().__init__(args)
        self.seed = self.args.seed
        self.seq_masking_method = self.args.seq_masking_method
        self.noise_method = self.args.noise_method
        self.pos_noise = self.args.pos_noise
        self.ang_noise = self.args.ang_noise

    def set_default_args(self, args):
        args.data_path = getattr(args, "data_path", None)
        args.seed = getattr(args, "seed", "2023")
        args.max_length = getattr(args, "max_length", 1024)
        args.seq_masking_method = getattr(args, "seq_masking_method", "transformerM")

        args.mask_prob = getattr(args, "mask_prob", 0.15)
        args.leave_unmasked_prob = getattr(args, "leave_unmasked_prob", 0.1)
        args.random_token_prob = getattr(args, "random_token_prob", 0.1)
        args.mask_multiple_length = getattr(args, "mask_multiple_length", 1)
        args.mask_stdev = getattr(args, "mask_stdev", 0.0)

        args.noise_method = getattr(args, "noise_method", "normal")
        args.pos_noise = getattr(args, "pos_noise", True)
        args.ang_noise = getattr(args, "ang_noise", True)

        args.coord_noise_mean = getattr(args, "coord_noise_mean", 0.0)
        args.coord_noise_stdev = getattr(args, "coord_noise_stdev", 0.1)
        args.angle_noise_mean = getattr(args, "angle_noise_mean", 0.0)
        args.angle_noise_stdev = getattr(args, "angle_noise_stdev", 0.003)

        return args

    def split_dataset(self, validation_ratio=0.03, sort=False):
        num_samples = len(self.keys)
        # Shuffle the indices and split them into training and validation sets
        indices = list(range(num_samples))
        random.Random(666).shuffle(indices)

        num_validation_samples = int(num_samples * validation_ratio)
        num_training_samples = num_samples - num_validation_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[num_training_samples:]

        # Create training and validation datasets
        dataset_train = self.__class__(self.args)
        dataset_train.keys = [self.keys[idx] for idx in training_indices]
        dataset_train.sizes = [self.sizes[idx] for idx in training_indices]

        dataset_val = self.__class__(self.args)
        dataset_val.keys = [self.keys[idx] for idx in validation_indices]
        dataset_val.sizes = [self.sizes[idx] for idx in validation_indices]

        if sort:
            dataset_train.__sort__()
            dataset_val.__sort__()

        return dataset_train, dataset_val

    def __getitem__(self, index: int) -> dict:
        key = self.keys[index]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = bstr2obj(value)
        item = {"id": index, **data}
        # item keys
        # {'name', 'size', "pos": np.ndarray(N, 37, 3, dtype=float32), "pos_mask": np.ndarray(N, 37, 3, dtype=int32), "ang": np.ndarray(N, 9, dtype=float32), "ang_mask": np.ndarray(N, 9, dtype=int32), "aa", "id": int}

        """
        - convert string sequence to int index
        """
        tokens = [self.vocab.tok_to_idx[tok] for tok in item["aa"]]
        # if len(tokens) > self.args.max_length - 2:
        #     start = random.randint(0, len(tokens) - self.args.max_length + 2)
        #     tokens = tokens[start : start + self.args.max_length - 2]
        # assert len(tokens) <= self.args.max_length - 2, f"len(tokens) = {len(tokens)} > {self.args.max_length - 2} = max_length - 2"

        # if self.vocab.prepend_bos:
        #     tokens.insert(0, self.vocab.cls_idx)
        # if self.vocab.append_eos:
        #     tokens.append(self.vocab.eos_idx)
        item["aa"] = np.array(tokens, dtype=np.int64)
        item["seq_length"] = len(tokens)

        """
        - mask the sequence in different ways
        """
        seed = int(hash((self.seed, index)) % 1e6)
        assert (
            "mask_idx" not in item
        ), "Item already contains mask_idx key, this is not expected!"

        masked_seq, mask_type, mask_pos = masking_registry[self.seq_masking_method](
            item, self.args, seed, self.vocab.mask_idx, self.vocab.standard_toks
        )

        item["masked_aa"] = mask_type
        item["mask_pos"] = mask_pos

        """
        Add noise to the coordinate or angles, manupilate the item['pos']/item['ang']:
        - add noise to the coordinate
        - add noise to the angles
        """
        assert (
            "pos_noise" or "ang_noise" not in item
        ), "Item already contains mask_idx key, this is not expected!"
        pos_noise, ang_noise = noise_registry[self.noise_method](
            item, self.args, seed, self.pos_noise, self.ang_noise
        )
        item["pos_noise"] = pos_noise
        item["ang_noise"] = ang_noise
        item["ang"] = item["ang"] / 180.0 * torch.pi  # + torch.pi

        # item["pos"] = item["pos"] + pos_noise
        item["ang"] = item["ang"] + ang_noise

        # TODO: considering mask the pos and ang, not used in the current version
        # set first position to zero
        # item["pos"] = (item["pos"] - item["pos"][0]) / 10.0
        item["pos"] = item["pos"]  # / 10.0

        return item

    def __len__(self) -> int:
        return len(self.keys)

    def size(self, index: int) -> int:
        sz = self.sizes[index]
        if self.vocab.prepend_bos:
            sz += 1
        if self.vocab.append_eos:
            sz += 1
        raise sz

    def num_tokens(self, index: int) -> int:
        return self.sizes[index]


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item, idx):
    numbers = item.pop("numbers")
    item["x"] = torch.tensor(numbers, dtype=torch.long).unsqueeze(-1)
    positions = item.pop("positions")
    item["pos"] = torch.tensor(positions, dtype=torch.float64)
    item["edge_attr"] = torch.zeros([0, 3], dtype=torch.long)
    item["edge_index"] = torch.zeros([2, 0], dtype=torch.long)
    item["cell"] = torch.tensor(item["cell"], dtype=torch.float64)
    item["pbc"] = torch.tensor(item["pbc"], dtype=torch.bool)
    item["idx"] = idx
    item["y"] = torch.tensor([item["info"]["energy"] / item["x"].size()[0]])
    item["stress"] = torch.tensor(item["info"]["stress"], dtype=torch.float64)
    item["forces"] = torch.tensor(item["forces"], dtype=torch.float64)

    item = Data(**item)

    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x

    N = x.size(0)
    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )
    shortest_path_result = (
        torch.full(adj.size(), 511, dtype=torch.long, device=x.device).cpu().numpy()
    )
    edge_input = (
        torch.zeros([N, N, 0, 3], dtype=torch.long, device=x.device).cpu().numpy()
    )
    spatial_pos = torch.from_numpy((shortest_path_result)).long()

    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)

    item.x = convert_to_single_emb(x)
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree
    item.edge_input = torch.from_numpy(edge_input).long()

    return item
