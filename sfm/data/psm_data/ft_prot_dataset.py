# -*- coding: utf-8 -*-
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
import random
from pathlib import Path
from typing import List, Optional, Union

import lmdb
import torch
from torch_geometric.data import Data
from torch_scatter import scatter_mean

from sfm.data.data_utils import _filter_by_size_dynamic
from sfm.data.dataset import FoundationModelDataset
from sfm.data.prot_data.util import bstr2obj
from sfm.data.psm_data.collator import collate_fn
from sfm.data.psm_data.dataset import AFDBLMDBDataset
from sfm.logging import logger
from sfm.models.psm.psm_config import PSMConfig


class ProteinSamplingDataset(AFDBLMDBDataset):
    def __init__(
        self,
        args: PSMConfig,
        lmdb_path: Optional[str],
    ):
        super().__init__(args, lmdb_path)

    def __getitem__(self, idx: Union[int, np.integer]) -> Data:
        key = self.keys[idx].encode()
        value = self.txn.get(key)
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        toks = bstr2obj(value)

        data = {}

        x = torch.tensor([self.vocab[tok] - 1 for tok in toks], dtype=torch.int64)
        coords = torch.zeros([x.size()[0], 3], dtype=torch.float64)

        data["sample_type"] = 2
        data["token_type"] = x
        data["idx"] = idx

        data["coords"] = coords
        data["num_atoms"] = x.size()[0]

        data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        data["pbc"] = torch.zeros(3, dtype=torch.float64).bool()
        data["stress"] = torch.zeros((3, 3), dtype=torch.float64, device=x.device)
        data["forces"] = torch.zeros(
            (x.size()[0], 3), dtype=torch.float64, device=x.device
        )
        data["energy"] = torch.tensor([0.0], dtype=torch.float64, device=x.device)
        data["energy_per_atom"] = torch.tensor(
            [0.0], dtype=torch.float64, device=x.device
        )

        data["has_energy"] = torch.tensor([0], dtype=torch.bool)
        data["has_forces"] = torch.tensor([0], dtype=torch.bool)

        data = self.generate_2dgraphfeat(data)

        return data

    def collate(self, samples):
        return collate_fn(
            samples,
            multi_hop_max_dist=5,
            preprocess_2d_bond_features_with_cuda=True,
            sample_in_validation=True,
        )


def mae(pred, true):
    return torch.mean(torch.abs(pred - true))


def mse(pred, true):
    return torch.mean(torch.square(pred - true))


def rmse(pred, true):
    return torch.sqrt(torch.mean(torch.square(pred - true)))


def f1_max(pred, target):
    """
    F1 score with the optimal threshold.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = (
        order
        + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
        * order.shape[1]
    )
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - torch.where(
        is_start, torch.zeros_like(precision), precision[all_order - 1]
    )
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(
        is_start, torch.zeros_like(recall), recall[all_order - 1]
    )
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()


def accuracy(pred, target):
    """
    Classification accuracy.

    Suppose there are :math:`N` sets and :math:`C` categories.

    Parameters:
        pred (Tensor): prediction of shape :math:`(N, C)`
        target (Tensor): target of shape :math:`(N,)`
    """
    return (pred.argmax(dim=-1) == target).float().mean()


def binary_accuracy(pred, target):
    return ((torch.sigmoid(pred) > 0.5) == target).float().mean()


def pearsonr(pred, target):
    """
    Pearson correlation between prediction and target.

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """
    pred_mean = pred.float().mean()
    target_mean = target.float().mean()
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    pred_normalized = pred_centered / pred_centered.norm(2)
    target_normalized = target_centered / target_centered.norm(2)
    pearsonr = pred_normalized @ target_normalized
    return pearsonr


def spearmanr(pred, target):
    """
    Spearman correlation between prediction and target.

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """

    def get_ranking(input):
        input_set, input_inverse = input.unique(return_inverse=True)
        order = input_inverse.argsort()
        ranking = torch.zeros(len(input_inverse), device=input.device)
        ranking[order] = torch.arange(
            1, len(input) + 1, dtype=torch.float, device=input.device
        )

        # for elements that have the same value, replace their rankings with the mean of their rankings
        mean_ranking = scatter_mean(
            ranking, input_inverse, dim=0, dim_size=len(input_set)
        )
        ranking = mean_ranking[input_inverse]
        return ranking

    pred = get_ranking(pred)
    target = get_ranking(target)
    covariance = (pred * target).mean() - pred.mean() * target.mean()
    pred_std = pred.std(unbiased=False)
    target_std = target.std(unbiased=False)
    spearmanr = covariance / (pred_std * target_std + 1e-10)
    return spearmanr


def area_under_prc(pred, target):
    """
    Area under precision-recall curve (PRC).

    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): binary targets of shape :math:`(n,)`
    """
    pred, target = pred.flatten(), target.flatten()
    order = pred.argsort(descending=True)
    target = target[order]
    precision = target.cumsum(0) / torch.arange(
        1, len(target) + 1, device=target.device
    )
    auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
    return auprc


class ProteinDownstreamDataset(FoundationModelDataset):
    """
    ProteinDownstreamDataset is a base class for downstream tasks.
    This class does not have the noise and masking method. It contains the labeled data for protein downstream tasks.
    It should be used for finetuning the model.

    """

    TASKINFO = {
        # single sequence --> single label
        "beta_lactamase": {
            "type": "regression",
            "splits": ["train", "valid", "test"],
            "mean_std": [0.7383112351980983, 0.31642946622284757],
            "classes": None,
        },
        "fluorescence": {
            "type": "regression",
            "splits": ["train", "valid", "test"],
            "mean_std": [3.180586883940159, 0.8339910288860691],
            "classes": None,
        },
        "solubility": {
            "type": "binary",
            "splits": ["train", "valid", "test"],
            "mean_std": [None, None],
            "classes": [0, 1],
        },  # 0-1
        "stability": {
            "type": "regression",
            "splits": ["train", "valid", "test"],
            "mean_std": [0.1790524860555312, 0.5662245232305079],
            "classes": None,
        },
        "subcellular_localization": {
            "type": "classification",
            "splits": ["train", "valid", "test"],
            "mean_std": [None, None],
            "classes": list(range(10)),
        },
        "subcellular_localization_2": {
            "type": "binary",
            "splits": ["train", "valid", "test"],
            "mean_std": [None, None],
            "classes": [0, 1],
        },
        # below three are same files, but different splits
        "remote_homology_fold": {
            "type": "classification",
            "splits": ["train", "valid", "test_fold_holdout"],
            "mean_std": [None, None],
            "classes": list(range(1195)),
        },
        "remote_homology_superfamily": {
            "type": "classification",
            "splits": ["train", "valid", "test_superfamily_holdout"],
            "mean_std": [None, None],
            "classes": list(range(1822)),
        },
        "remote_homology_family": {
            "type": "classification",
            "splits": ["train", "valid", "test_family_holdout"],
            "mean_std": [None, None],
            "classes": list(range(3439)),
        },
        # single sequence --> multiple labels
        "EnzymeCommission": {
            "type": "multi_classification",
            "splits": ["train", "valid", "test"],
            "mean_std": [None, None],
            "classes": list(range(538)),
        },
        "GeneOntology_mf": {
            "type": "multi_classification",
            "splits": ["train", "valid", "test"],
            "mean_std": [None, None],
            "classes": list(range(489)),
        },
        "GeneOntology_bp": {
            "type": "multi_classification",
            "splits": ["train", "valid", "test"],
            "mean_std": [None, None],
            "classes": list(range(1943)),
        },
        "GeneOntology_cc": {
            "type": "multi_classification",
            "splits": ["train", "valid", "test"],
            "mean_std": [None, None],
            "classes": list(range(320)),
        },
        # single sequence --> residue labels + residue masks
        "secondary_structure": {
            "type": "residue_classification",
            "splits": ["train", "valid", "casp12", "cb513", "ts115"],
            "mean_std": [None, None],
            "classes": list(range(3)),
        },
        # multiple sequences --> single label
        "human_ppi": {
            "type": "binary",
            "splits": ["train", "valid", "test", "cross_species_test"],
            "mean_std": [None, None],
            "classes": [0, 1],
        },
        "yeast_ppi": {
            "type": "binary",
            "splits": ["train", "valid", "test", "cross_species_test"],
            "mean_std": [None, None],
            "classes": [0, 1],
        },
        "ppi_affinity": {
            "type": "regression",
            "splits": ["train", "valid", "test"],
            "mean_std": [-11.654589870742205, 3.0832061340498975],
            "classes": None,
        },
        # TODO: single sequence --> contact map
    }

    def __init__(self, args: PSMConfig, direct=True) -> None:
        if direct:
            raise ValueError(
                "DownstreamLMDBDataset should not be initialized directly, please use DownstreamLMDBDataset.load_dataset(args) instead."
            )
        super().__init__()

        self.args = self.set_default_args(args)
        assert (
            self.args.data_path and Path(self.args.data_path).is_dir()
        ), f"Processed file not found: {self.args.data_path}"
        self.lmdb_path = Path(self.args.data_path)

        self.env = lmdb.open(
            str(self.lmdb_path), subdir=True, readonly=True, lock=False, readahead=False
        )
        self.txn = self.env.begin(write=False)

        metadata = bstr2obj(self.txn.get("__metadata__".encode()))
        self.sizes, self.keys = metadata["sizes"], metadata["keys"]
        self.comment = metadata["comment"]
        if args.ifstack:
            self.filter_indices_by_size(
                indices=np.array(range(len(self.keys))),
                max_sizes=self.args.max_length - 2,
            )
        else:
            self.filter_indices_by_size(
                indices=np.array(range(len(self.keys))), max_sizes=self.args.max_length
            )

        self.lmdb_basepath = Path(self.args.data_basepath)
        self.max_length = self.args.max_length
        self.task_name = self.args.task_name
        self.label_field = self.args.label_field
        self.split = self.args.split
        self.normalize_label = self.args.normalize_label

        self.vocab = {
            # "<pad>": 0,  # padding
            # "1"-"127": 1-127, # atom type
            # "<cell_corner>": 128, use for pbc material
            "L": 130,
            "A": 131,
            "G": 132,
            "V": 133,
            "S": 134,
            "E": 135,
            "R": 136,
            "T": 137,
            "I": 138,
            "D": 139,
            "P": 140,
            "K": 141,
            "Q": 142,
            "N": 143,
            "F": 144,
            "Y": 145,
            "M": 146,
            "H": 147,
            "W": 148,
            "C": 149,
            "X": 150,
            "B": 151,
            "U": 152,
            "Z": 153,
            "O": 154,
            "-": 155,
            ".": 156,
            "<mask>": 157,
            "<cls>": 158,
            "<eos>": 159,
            # "<unk>": 160,
        }

        assert (
            self.split in ProteinDownstreamDataset.TASKINFO[self.task_name]["splits"]
        ), f"split must be one of {self.TASKINFO[self.task_name]['splits']} for task {self.task_name}, but got {self.split}"

    def set_default_args(self, args):
        args.data_basepath = getattr(args, "data_basepath", None)
        args.task_name = getattr(args, "task_name", None)
        args.label_field = getattr(args, "label_field", "target")
        args.split = getattr(args, "split", None)
        args.max_length = getattr(args, "max_length", 1024)
        args.normalize_label = getattr(args, "normalize_label", False)
        # this should be set by self.load_dataset
        args.data_path = getattr(args, "data_path", None)
        # currently, we do not use seed in DownstreamLMDBDataset since all datasets are splitted already, but we keep it for future use
        # args.seed = getattr(args, "seed", "2023")
        return args

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)
        """
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif hasattr(self, "sizes") and isinstance(self.sizes, list):
                sizes = np.array(self.sizes)
                ignored = indices[np.array(sizes[indices]) > max_sizes].tolist()
                indices = indices[np.array(sizes[indices]) <= max_sizes]
            else:
                indices, ignored = _filter_by_size_dynamic(
                    indices, self.size, max_sizes
                )
        else:
            indices, ignored = _filter_by_size_dynamic(indices, self.size, max_sizes)

        logger.warning(
            f"Removed {len(ignored)} examples from the dataset because they are longer than {max_sizes}."
        )
        self.sizes = [self.sizes[idx] for idx in indices]
        self.keys = [self.keys[idx] for idx in indices]

    def generate_2dgraphfeat(self, data):
        N = data["token_type"].shape[0]
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.zeros([2, 0], dtype=torch.long)
        edge_attr = torch.zeros([0, 3], dtype=torch.long)
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = torch.cat(
            [
                data["token_type"].unsqueeze(-1),
                torch.zeros([data["token_type"].size()[0], 8], dtype=torch.long),
            ],
            dim=-1,
        )
        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = indgree

        if self.args.preprocess_2d_bond_features_with_cuda:
            attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
            data["adj"] = adj
            data["attn_edge_type"] = attn_edge_type
        else:
            shortest_path_result = (
                torch.full(adj.size(), 511, dtype=torch.long).cpu().numpy()
            )
            edge_input = torch.zeros([N, N, 0, 3], dtype=torch.long)
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            data["edge_input"] = edge_input
            data["spatial_pos"] = spatial_pos

        return data

    def __getitem__(self, index: int) -> dict:
        key = self.keys[index]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        val = bstr2obj(value)
        # item = {"id": index, **data}
        data = {}
        if isinstance(val["aa"][0], list):
            x = []
            for idx, seq in enumerate(val["aa"]):
                tokens = [self.vocab[tok] - 1 for tok in seq]
                tokens.insert(0, self.vocab["<cls>"] - 1)
                tokens.append(self.vocab["<eos>"] - 1)
                x.extend(tokens)
            x = torch.tensor(x, dtype=torch.int64)

        else:
            tokens = [self.vocab[tok] - 1 for tok in val["aa"]]
            tokens.insert(0, self.vocab["<cls>"] - 1)
            tokens.append(self.vocab["<eos>"] - 1)
            x = torch.tensor(tokens, dtype=torch.int64)

        # minus 1 due to add padding index=0 in collator
        # x = torch.tensor([self.vocab[tok] - 1 for tok in val["aa"]], dtype=torch.int64)
        # CA atom positions, assume all values are valid.
        coords = torch.zeros((len(x), 3), dtype=torch.float64)  # data["pos"][:, 1, :]
        data["coords"] = coords

        data["sample_type"] = 2
        data["token_type"] = x
        data["idx"] = index

        data["num_atoms"] = x.size()[0]

        data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        data["pbc"] = torch.zeros(3, dtype=torch.float64).bool()
        data["stress"] = torch.zeros((3, 3), dtype=torch.float64, device=x.device)
        data["forces"] = torch.zeros(
            (x.size()[0], 3), dtype=torch.float64, device=x.device
        )
        data["energy"] = torch.tensor([0.0], dtype=torch.float64, device=x.device)
        data["energy_per_atom"] = torch.tensor(
            [0.0], dtype=torch.float64, device=x.device
        )
        data["has_energy"] = torch.tensor([0], dtype=torch.bool)
        data["has_forces"] = torch.tensor([0], dtype=torch.bool)

        data = self.generate_2dgraphfeat(data)

        data["is_stable_periodic"] = False
        # make the label's type right
        if self.TASKINFO[self.task_name]["type"] == "regression":
            data[self.label_field] = torch.tensor(
                val[self.label_field], dtype=torch.float64
            )
        elif self.TASKINFO[self.task_name]["type"] in {
            "classification",
            "binary",
            "multi_classification",
        }:
            # patch for remote_homology
            if self.task_name == "remote_homology_fold":
                data[self.label_field] = torch.tensor(
                    [val[self.label_field][1]], dtype=torch.int64
                )
            elif self.task_name == "remote_homology_superfamily":
                data[self.label_field] = torch.tensor(
                    [val[self.label_field][2]], dtype=torch.int64
                )
            elif self.task_name == "remote_homology_family":
                data[self.label_field] = torch.tensor(
                    [val[self.label_field][3]], dtype=torch.int64
                )
            else:
                data[self.label_field] = torch.tensor(
                    val[self.label_field], dtype=torch.int64
                )
            # patch end
        elif self.TASKINFO[self.task_name]["type"] == "residue_classification":
            data[self.label_field], data[f"{self.label_field}_mask"] = val[
                self.label_field
            ]
        return data

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

    # def collate(self, samples: List[dict]) -> dict:
    #     batch = collate_fn(samples)
    #     # batch["id"] = torch.tensor([s["id"] for s in samples], dtype=torch.long)
    #     # batch["naa"] = torch.tensor([len(s["aa"]) for s in samples], dtype=torch.long)
    #     batch["target"] = torch.cat([s["target"] for s in samples])
    #     batch["target_offset"] = torch.tensor(
    #         [len(s["target"]) for s in samples], dtype=torch.long
    #     )
    #     return batch

    def collate(self, samples: List[dict]) -> dict:
        if self.task_name in {"human_ppi", "yeast_ppi", "ppi_affinity"}:
            raise NotImplementedError()
            # return collate_multiseq_downstream_fn(samples, self.vocab)
        elif self.task_name == "secondary_structure":
            # return collate_secondary_structure_fn(samples, self.vocab)
            raise NotImplementedError()
        else:
            return collate_fn_protein_downstream(
                samples, self.vocab, single_sequence=True
            )

    @classmethod
    def load_dataset(cls, args):
        if not hasattr(args, "task_name"):
            raise ValueError(
                "args must have task_name to load ProteinDownstreamDataset."
            )
        if args.task_name not in ProteinDownstreamDataset.TASKINFO:
            raise ValueError(
                f"args.task_name = {args.task_name} not support yet, must be one of {ProteinDownstreamDataset.TASKINFO.keys()}"
            )
        dset_dict = {}
        for split in ProteinDownstreamDataset.TASKINFO[args.task_name]["splits"]:
            args.split = split
            args.data_path = str(
                Path(args.data_basepath)
                / args.task_name
                / f"{args.task_name}_{split}.lmdb"
            )
            dset_dict[split] = cls(args, direct=False)
        return dset_dict


from sfm.data.psm_data.collator import (
    convert_to_single_emb,
    pad_1d_unsqueeze,
    pad_2d_unsqueeze,
    pad_3d_unsqueeze,
    pad_attn_bias_unsqueeze,
    pad_attn_edge_input_unsqueeze,
    pad_edge_info_unsqueeze,
    pad_pos_unsqueeze,
    pad_spatial_pos_unsqueeze,
)

# this function is from sfm.data.psm_data.collator
# do not pollute the original function, just modify it here.


def collate_fn_protein_downstream(
    items,
    multi_hop_max_dist=20,
    use_pbc=True,
    preprocess_2d_bond_features_with_cuda=True,
    sample_in_validation: bool = False,
    single_sequence=False,
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

    idx = torch.tensor([i["idx"] for i in items], dtype=torch.long)
    max_node_num = max(i["token_type"].shape[0] for i in items)
    energy = [i["energy"] for i in items]
    energy_per_atom = [i["energy_per_atom"] for i in items]
    forces = torch.cat([pad_pos_unsqueeze(i["forces"], max_node_num) for i in items])
    energy = torch.cat(energy)
    has_energy = torch.cat([i["has_energy"] for i in items], dim=0)
    has_forces = torch.cat([i["has_forces"] for i in items], dim=0)
    energy_per_atom = torch.cat(energy_per_atom)

    x = torch.cat([pad_2d_unsqueeze(i["node_attr"], max_node_num) for i in items])

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

    if sample_in_validation:
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

    if sample_in_validation:
        batched_data.update(
            dict(
                edge_attr=edge_attr, edge_index=edge_index, num_edges=num_edges, idx=idx
            )
        )

    if single_sequence:
        batched_data["target"] = torch.cat([s["target"] for s in items])
        batched_data["target_offset"] = torch.tensor(
            [len(s["target"]) for s in items], dtype=torch.long
        )

    return batched_data
