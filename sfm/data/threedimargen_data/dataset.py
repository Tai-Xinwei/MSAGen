# -*- coding: utf-8 -*-
import heapq
import json
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Union

import mendeleev
import numpy as np
import torch
from mendeleev import element

from sfm.data.dataset import Batch, Data, InMemoryFoundationModelDataset
from sfm.data.threedimargen_data.tokenizer import ThreeDimTokenizer
from sfm.logging import logger


# allow pad_num to be int or float
def pad_1d_unsqueeze(
    x: torch.Tensor, padlen: int, start: int, pad_num: Union[int, float]
):
    # (N) -> (1, padlen)
    xlen = x.size(0)
    assert (
        start + xlen <= padlen
    ), f"padlen {padlen} is too small for xlen {xlen} and start point {start}"
    new_x = x.new_full([padlen], pad_num, dtype=x.dtype)
    new_x[start : start + xlen] = x
    x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(
    x: torch.Tensor, padlen: int, start: int, pad_num: Union[int, float]
):
    # (N, 3) -> (1, padlen, 3)
    xlen = x.size(0)
    assert (
        start + xlen <= padlen
    ), f"padlen {padlen} is too small for xlen {xlen} and start point {start}"
    new_x = x.new_full([padlen, 3], pad_num, dtype=x.dtype)
    new_x[start : start + xlen] = x
    x = new_x
    return x.unsqueeze(0)


def collate_fn(samples: List[dict], vocab: ThreeDimTokenizer):
    """
    Overload BaseWrapperDataset.collater
    May be future changes need config

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """
    max_tokens = max(len(s["tokens"]) for s in samples)

    batch = dict()

    batch["id"] = torch.tensor([s["id"] for s in samples], dtype=torch.long)

    batch["ntokens"] = torch.tensor(
        [len(s["tokens"]) for s in samples], dtype=torch.long
    )

    batch["input_ids"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["tokens"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    )
    batch["input_coordinates"] = torch.cat(
        [torch.from_numpy(s["coordinates"]) for s in samples]
    )
    batch["label_ids"] = batch["input_ids"].clone()
    batch["label_coordinates"] = batch["input_coordinates"].clone()
    batch["coordinates_mask"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["coordinates_mask"]),
                max_tokens,
                0,
                vocab.padding_idx,
            )
            for s in samples
        ]
    )
    return batch


def collate_fn_pp(samples: List[dict], vocab: ThreeDimTokenizer):
    """
    Overload BaseWrapperDataset.collater
    May be future changes need config

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """
    max_tokens = max(len(s["tokens"]) for s in samples)

    input_ids = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["tokens"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    )
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=vocab.padding_idx
    )
    input = tuple([input_ids, input_ids.ne(vocab.padding_idx)])
    labels = input
    return (input, labels)


class BatchedDataDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        args=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.args = args
        self.vocab = dataset.vocab

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collate(self, samples):
        if self.args is None or self.args.pipeline_model_parallel_size == 0:
            return collate_fn(samples, self.vocab)
        else:
            return collate_fn_pp(samples, self.vocab)

    def num_tokens(self, index: int) -> int:
        return self.dataset.sizes[index]


def traverse(start, sites):
    # Initialize the priority queue with the starting element
    queue = [(0, start)]
    heapq.heapify(queue)
    visited = set([start])
    visiting_order = [start]

    while queue:
        _, current = heapq.heappop(queue)

        # Get the neighbors of the current element
        neighbors = find_neighbors(current, sites)

        # Add unvisited neighbors to the priority queue and mark them as visited
        for neighbor in neighbors:
            if neighbor not in visited:
                distance = calculate_distance(current, neighbor)
                heapq.heappush(queue, (distance, neighbor))
                visited.add(neighbor)
                visiting_order.append(neighbor)

    return visiting_order


def find_neighbors(current, sites):
    neighbors = [site for site in sites if site != current]
    return neighbors


def calculate_distance(site1, site2):
    return np.linalg.norm(np.array(site1["xyz"]) - np.array(site2["xyz"]))


def normalize_coordinates(coordinates):
    ret = []
    for x in coordinates:
        if x < 0:
            x = x + abs(int(x)) + 1
        if x > 1:
            x = x - int(x)
        ret.append(round(x, 6))
    return ret


class ThreeDimGenInferDataset:
    def __init__(self, dict_path: str, args=None):
        self.vocab = ThreeDimTokenizer.from_file(dict_path)
        self.args = args

    def tokenize(self, formula):
        item = dict()

        formula_ids = self.vocab.encode(
            formula, prepend_bos=True, append_gen=True, append_eos=False
        )

        coordinates_mask = [0 for _ in range(len(formula_ids))]
        item["tokens"] = formula_ids
        item["coordinates_mask"] = np.array(coordinates_mask)
        return item

    def collate(self, samples):
        if self.args is None or self.args.pipeline_model_parallel_size == 0:
            return collate_fn(samples, self.vocab)
        else:
            return collate_fn_pp(samples, self.vocab)

    def encode(self, formula):
        item = self.tokenize(formula)
        return self.collate([item])


class ThreeDimGenDataset(InMemoryFoundationModelDataset):
    def __init__(
        self,
        tokenizer,
        data_path: Union[str, list[str]],
        args=None,
        shuffle: bool = True,
    ):
        self.vocab = tokenizer
        self.args = args
        # self.max_position_embeddings = args.max_position_embeddings
        # self.atomic_number = {e.symbol:element(e.symbol).atomic_number for e in mendeleev.get_all_elements()}

        if data_path.count(",") > 0:
            data_path = data_path.split(",")

        if isinstance(data_path, str):
            data_path = [data_path]

        self.data = []
        self.sizes = []
        for path in data_path:
            self._load_data(path)

        if shuffle:
            random.shuffle(self.data)

        if self.args.scale_digit:
            logger.info(f"scale digit with scale {self.args.scale_digit}")

    def _load_data(self, data_path):
        data_path = data_path.strip()
        with open(data_path, "r") as f:
            for line in f:
                line = json.loads(line)
                if len(line["sites"]) > self.args.max_sites:
                    continue
                self.data.append(line)
                self.sizes.append(len(line["sites"]) * 2 + 1 + 1 + 3 + 1 + 1)

    def __len__(self):
        return len(self.data)

    # def _sort_sites(self, formula, sites):
    #     ele_num = self.vocab.get_ele_num(formula)
    #     ele_num = sorted(ele_num, key=lambda x: self.atomic_number[x[0]], reverse=True)
    #     start_ele = ele_num[0][0]
    #     eles = []
    #     for s in sites:
    #         if s["element"] == start_ele:
    #             eles.append(s)
    #     eles = sorted(eles, key=lambda x: np.linalg.norm(x["xyz"]))
    #     start_site = eles[0]
    #     # sort the sites according to their distance to the start site
    #     sites = sorted(sites, key=lambda x: calculate_distance(x, start_site))
    #     return sites

    # def serialize_sites(self, formula, sites):
    #     sites_coords = {}
    #     for site in sites:
    #         elem = site["element"]
    #         if elem not in sites_coords:
    #             sites_coords[elem] = []
    #         sites_coords[elem].append(site)
    #     tokens = self.vocab.tokenize(formula, sites, prepend_bos=False, append_gen=False, append_eos=False)
    #     ret = []
    #     for token in tokens:
    #         if token in sites_coords:
    #             site = sites_coords[token].pop(0)
    #             ret.append(site)
    #         else:
    #             raise ValueError(f"token {token} not in sites_coords")
    #     return ret

    def sort_sites(self, sites):
        # sort the sites according to their distance to the start site
        sites_dict = {}
        for site in sites:
            elem = site["element"]
            if elem not in sites_dict:
                sites_dict[elem] = []
            sites_dict[elem].append(site)
        for elem in sites_dict:
            sites_dict[elem] = sorted(
                sites_dict[elem],
                key=lambda x: np.linalg.norm(x["fractional_coordinates"]),
            )
        ret = []
        for elem in sites_dict:
            ret.extend(sites_dict[elem])
        return ret

    def __getitem__(self, index):
        item = dict()
        data_item = self.data[index]

        # sort sites
        # sorted_sites = self.sort_sites(data_item["sites"])
        sorted_sites = data_item["sites"]

        # get all sites
        sites_ids = [self.vocab.bos_idx]
        sites_ids.extend([self.vocab.get_idx(site["element"]) for site in sorted_sites])
        coordinates_mask = [0 for _ in range(len(sites_ids))]

        # add space group
        space_group_no = str(data_item["space_group"]["no"])
        sites_ids.append(self.vocab.get_idx(space_group_no))
        coordinates_mask.append(0)

        # add special token
        sites_ids.append(self.vocab.gen_idx)
        coordinates_mask.append(0)

        # add lattice
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        sites_ids.extend([self.vocab.mask_idx for _ in range(3)])
        coordinates_mask.extend([1 for _ in range(3)])

        # add coordinates
        sites_ids.extend([self.vocab.mask_idx for _ in range(len(sorted_sites))])
        coordinates_mask.extend([1 for _ in range(len(sorted_sites))])
        coordinates = [
            np.array(normalize_coordinates(site["fractional_coordinates"])).astype(
                np.float32
            )
            for site in sorted_sites
        ]
        if self.args.scale_digit:
            coordinates = [
                np.array([x * self.args.scale_digit for x in coordinate])
                for coordinate in coordinates
            ]

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['formula']}"

        # eos
        sites_ids.append(self.vocab.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates = np.concatenate([lattice, coordinates], axis=0)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(coordinates_mask)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def collate(self, samples):
        if self.args is None or self.args.pipeline_model_parallel_size == 0:
            return collate_fn(samples, self.vocab)
        else:
            return collate_fn_pp(samples, self.vocab)

    def num_tokens(self, index: int) -> int:
        return self.sizes[index]


if __name__ == "__main__":

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Namespace()
    args.dict_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dict.txt"
    )
    tokenizer = ThreeDimTokenizer.from_file(args.dict_path, args)
    args.train_data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sample_data.jsonl"
    )
    args.max_sites = 100
    args.scale_digit = 10

    print(args)
    print("=================")
    print("Test ThreeDimGen dataset")
    dataset = ThreeDimGenDataset(tokenizer, args.train_data_path, args, shuffle=False)
    print(len(dataset))
    print(dataset[12])
    print(dataset[512])
    print()
    batch_dataset = BatchedDataDataset(dataset)
    print(batch_dataset.collate([dataset[0], dataset[100], dataset[500], dataset[800]]))

    # dataset = ThreeDimGenDataset(tokenizer, "/blob/v-yihangzhai/materials_data/mp/materialProject_all_20230717_props_sharedByTian_sg.jsonl", shuffle=False)
    # for d in dataset:
    # print()
    # print(batch_dataset.collate([dataset[6123], dataset[6001], dataset[6299], dataset[6599]]))
