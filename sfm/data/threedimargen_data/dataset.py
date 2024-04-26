# -*- coding: utf-8 -*-
import heapq
import json
import os
import random
import sys
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
from functools import cmp_to_key
from typing import Any, List, Union

import numpy as np
import torch
from pymatgen.core.structure import Structure

from sfm.data.dataset import Batch, Data, InMemoryFoundationModelDataset
from sfm.data.threedimargen_data.tokenizer import (  # ThreeDimARGenSlicesEnergyTokenizer,
    ThreeDimARGenLanEnergyTokenizer,
    ThreeDimARGenLanTokenizer,
    ThreeDimARGenLanV2EnergyTokenizer,
    ThreeDimARGenLanV2Tokenizer,
    ThreeDimARGenNumEnergyTokenizer,
    ThreeDimARGenNumTokenizer,
    ThreeDimARGenSlicesTokenizer,
    ThreeDimARGenTokenizer,
    normalize_frac_coordinate,
    tokenize_float,
    tokenize_float_v2,
)
from sfm.logging import logger


class MODE(Enum):
    TRAIN = 1
    VAL = 2
    INFER = 3


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


def collate_fn(samples: List[dict], tokenizer, mode=MODE.TRAIN):
    """
    Overload BaseWrapperDataset.collater
    May be future changes need config

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """
    max_tokens = max(len(s["tokens"]) for s in samples)
    max_masks = max(len(s["coordinates_mask"]) for s in samples)

    batch = dict()

    if "id" in samples[0]:
        batch["id"] = torch.tensor([s["id"] for s in samples], dtype=torch.long)

    batch["ntokens"] = torch.tensor(
        [len(s["tokens"]) for s in samples], dtype=torch.long
    )

    batch["input_ids"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["tokens"]).long(),
                max_tokens,
                0 if mode != MODE.INFER else max_tokens - len(s["tokens"]),
                tokenizer.padding_idx,
            )
            for s in samples
        ]
    )

    batch["attention_mask"] = batch["input_ids"].ne(tokenizer.padding_idx).long()

    if mode != MODE.INFER:
        batch["label_ids"] = batch["input_ids"].clone()

    if "coordinates" in samples[0]:
        batch["input_coordinates"] = torch.cat(
            [torch.from_numpy(s["coordinates"]) for s in samples]
        ).to(torch.float32)
        if mode != MODE.INFER:
            batch["label_coordinates"] = batch["input_coordinates"].clone()

    batch["coordinates_mask"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["coordinates_mask"]).long(),
                max_masks,
                0 if mode != MODE.INFER else max_tokens - len(s["tokens"]),
                tokenizer.padding_idx,
            )
            for s in samples
        ]
    )
    return batch


def collate_fn_language(samples: List[dict], tokenizer, mode=MODE.TRAIN):
    """
    Overload BaseWrapperDataset.collater
    May be future changes need config

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """
    max_tokens = max(len(s["tokens"]) for s in samples)

    batch = dict()

    if "id" in samples[0]:
        batch["id"] = torch.tensor([s["id"] for s in samples], dtype=torch.long)

    batch["ntokens"] = torch.tensor(
        [len(s["tokens"]) for s in samples], dtype=torch.long
    )

    batch["input_ids"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["tokens"]).long(),
                max_tokens,
                0 if mode != MODE.INFER else max_tokens - len(s["tokens"]),
                tokenizer.padding_idx,
            )
            for s in samples
        ]
    )

    batch["attention_mask"] = batch["input_ids"].ne(tokenizer.padding_idx).long()

    if mode != MODE.INFER:
        batch["label_ids"] = batch["input_ids"].clone()

    return batch


def collate_fn_pp(samples: List[dict], tokenizer):
    """
    Overload BaseWrapperDataset.collater
    May be future changes need config

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """
    max_tokens = max(len(s["tokens"]) for s in samples)

    input_ids = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["tokens"]), max_tokens, 0, tokenizer.padding_idx
            )
            for s in samples
        ]
    )
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.padding_idx
    )
    input = tuple([input_ids, input_ids.ne(tokenizer.padding_idx)])
    labels = input
    return (input, labels)


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


def normalize_frac_coordinates(coordinates: list, margin: float = 1e-4):
    return [normalize_frac_coordinate(x, margin) for x in coordinates]


def compare_by_coords(order=None):
    def innfer_f(a, b):
        frac_a = a["fractional_coordinates"]
        frac_b = b["fractional_coordinates"]
        if order == "<orderxyz>" or order is None:
            pass
        elif order == "<orderxzy>":
            frac_a = [frac_a[0], frac_a[2], frac_a[1]]
            frac_b = [frac_b[0], frac_b[2], frac_b[1]]
        elif order == "<orderyxz>":
            frac_a = [frac_a[1], frac_a[0], frac_a[2]]
            frac_b = [frac_b[1], frac_b[0], frac_b[2]]
        elif order == "<orderyzx>":
            frac_a = [frac_a[1], frac_a[2], frac_a[0]]
            frac_b = [frac_b[1], frac_b[2], frac_b[0]]
        elif order == "<orderzxy>":
            frac_a = [frac_a[2], frac_a[0], frac_a[1]]
            frac_b = [frac_b[2], frac_b[0], frac_b[1]]
        elif order == "<orderzyx>":
            frac_a = [frac_a[2], frac_a[1], frac_a[0]]
            frac_b = [frac_b[2], frac_b[1], frac_b[0]]
        else:
            raise ValueError(f"Unknown order {order}")
        if frac_a[0] > frac_b[0]:
            return 1
        elif frac_a[0] < frac_b[0]:
            return -1
        elif frac_a[1] > frac_b[1]:
            return 1
        elif frac_a[1] < frac_b[1]:
            return -1
        elif frac_a[2] > frac_b[2]:
            return 1
        elif frac_a[2] < frac_b[2]:
            return -1
        else:
            return 0

    return innfer_f


def get_niggli_reduced_form(species, lattice, coords):
    crystal = Structure(
        species=species, coords=coords, lattice=lattice, coords_are_cartesian=True
    )
    new_crystal = Structure(
        species=crystal.species,
        coords=crystal.cart_coords,
        lattice=crystal.lattice.get_niggli_reduced_lattice(),
        coords_are_cartesian=True,
    )
    new_lattice = new_crystal.lattice.matrix.tolist()
    new_coords = new_crystal.frac_coords.tolist()
    return new_lattice, new_coords


def sort_sites(sites, order=None):
    # sort the sites according to their distance to the start site
    sites_dict = OrderedDict()
    for site in sites:
        elem = site["element"]
        if elem not in sites_dict:
            sites_dict[elem] = []
        sites_dict[elem].append(site)
    for elem in sites_dict:
        sites_dict[elem] = sorted(
            sites_dict[elem],
            key=cmp_to_key(compare_by_coords(order)),
        )
    ret = []
    for elem in sites_dict:
        ret.extend(sites_dict[elem])
    return ret


class ThreeDimARGenDataset(InMemoryFoundationModelDataset):
    def __init__(
        self,
        tokenizer,
        data_path: Union[str, list[str]] = None,
        args=None,
        shuffle: bool = True,
        mode=MODE.TRAIN,
    ):
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.max_position_embeddings = args.max_position_embeddings

        self.data = []
        self.sizes = []

        if data_path is not None:
            if data_path.count(",") > 0:
                data_path = data_path.split(",")
            if isinstance(data_path, str):
                data_path = [data_path]
            for path in data_path:
                self.load_data_from_file(path)

        if shuffle:
            random.shuffle(self.data)

        if self.args.scale_coords:
            logger.info(f"scale coords with scale {self.args.scale_coords}")

    def get_sequence_length(self, data_item):
        if isinstance(self.tokenizer, ThreeDimARGenNumTokenizer):
            # <bos> [n] <sp> <sgn> <coords> [3 lattice] [n coords] <eos>
            n = len(data_item["sites"])
            return 1 + n + 1 + 1 + 1 + 3 + n + 1
        elif isinstance(self.tokenizer, ThreeDimARGenLanTokenizer):
            # <bos> [n] <sp> <sgn> <coords> [9 lattice * (7 digits + <cs>)] [n * (3 coords * (6 digits + <cs>))]
            n = len(data_item["sites"])
            return 1 + n + 2 + 1 + 9 * (7 + 1) + n * (3 * (6 + 1))
        elif isinstance(self.tokenizer, ThreeDimARGenLanV2Tokenizer):
            # <bos> [n] <sp> <sgn> <coords> [9 lattice * (4 tokens + <cs>)] [n * (3 coords * (3 tokens + <cs>))]
            n = len(data_item["sites"])
            return 1 + n + 2 + 1 + 9 * (4 + 1) + n * (3 * (3 + 1))
        elif isinstance(self.tokenizer, ThreeDimARGenSlicesTokenizer):
            if isinstance(data_item, str):
                return len(data_item.split(" ")) + 2
            else:
                return len(data_item["sites"]) + 2

    def load_dict(self, lines: List[dict]):
        skipped = 0
        for data_item in lines:
            size = self.get_sequence_length(data_item)
            if (
                self.args.max_sites is not None
                and len(data_item["sites"]) > self.args.max_sites
            ):
                skipped += 1
                continue
            if size > self.args.max_position_embeddings:
                skipped += 1
                continue

            if self.mode != MODE.INFER:
                # get niggli reduced structure
                if self.args.niggli_reduced:
                    species = [site["element"] for site in data_item["sites"]]
                    cart_coords = [
                        site["cartesian_coordinates"] for site in data_item["sites"]
                    ]
                    lattice = data_item["lattice"]
                    new_lattice, new_coords = get_niggli_reduced_form(
                        species, lattice, cart_coords
                    )
                    data_item["lattice"] = new_lattice
                    for i in range(len(data_item["sites"])):
                        data_item["sites"][i]["fractional_coordinates"] = new_coords[i]
                # normalize fractional coordinates
                for i in range(len(data_item["sites"])):
                    data_item["sites"][i][
                        "fractional_coordinates"
                    ] = normalize_frac_coordinates(
                        data_item["sites"][i]["fractional_coordinates"]
                    )
                sorted_sites = sort_sites(data_item["sites"])
                data_item["sites"] = sorted_sites

            self.data.append(data_item)  # type(data_item:) = dict
            self.sizes.append(size)
        logger.info(f"skipped {skipped} samples due to length constraints")

    def load_json(self, lines: List[str]):
        lines = [json.loads(line) for line in lines]
        self.load_dict(lines)

    def load_txt(self, lines: List[str]):
        skipped = 0
        for line in lines:
            data_item = line.strip()
            size = self.get_sequence_length(data_item)
            if size > self.args.max_position_embeddings:
                skipped += 1
                continue
            self.data.append(data_item)
            self.sizes.append(size)
        logger.info(f"skipped {skipped} samples due to length constraints")

    def infer_data_format(self, data_path, data_format):
        if data_path.endswith(".jsonl") or data_path.endswith(".json"):
            file_format = "json"
        elif data_path.endswith(".txt"):
            file_format = "txt"
        else:
            file_format = "txt"
        if data_format is not None:
            if data_format == file_format:
                return file_format
            else:
                return data_format
        return file_format

    def load_data_from_file(self, data_path, data_format=None):
        data_path = data_path.strip()
        data_format = self.infer_data_format(data_path, data_format)
        with open(data_path, "r") as f:
            lines = f.readlines()
        if data_format == "json":
            self.load_json(lines)
        elif data_format == "txt":
            self.load_txt(lines)
        else:
            raise ValueError(f"Unknown data format {data_format}")

    def __len__(self):
        return len(self.data)

    def get_train_item_num(self, index):
        item = dict()
        data_item = self.data[index]

        # sort sites if reorder
        if self.args.reorder:
            order = random.choice(self.tokenizer.order_tokens)
            sites = sort_sites(data_item["sites"], order)
        else:
            sites = data_item["sites"]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # get all sites
        sites_ids.extend([self.tokenizer.get_idx(site["element"]) for site in sites])
        coordinates_mask.extend([0 for _ in range(len(sites))])

        # add space group
        sites_ids.append(self.tokenizer.sg_idx)
        coordinates_mask.append(0)
        space_group_no = str(data_item["space_group"]["no"])
        space_group_tok = f"<sgn>{space_group_no}"
        sites_ids.append(self.tokenizer.get_idx(space_group_tok))
        coordinates_mask.append(0)

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add order if needed
        if self.args.reorder:
            sites_ids.append(self.tokenizer.get_idx(order))
            coordinates_mask.append(0)

        # add lattice
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(3)])
        coordinates_mask.extend([1 for _ in range(3)])

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(sites))])
        coordinates_mask.extend([1 for _ in range(len(sites))])
        coordinates = np.array(
            [site["fractional_coordinates"] for site in sites]
        ).astype(np.float32)
        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['formula']}"

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates = np.concatenate([lattice, coordinates], axis=0)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_infer_item_num(self, index):
        item = dict()
        data_item = self.data[index]

        sorted_sites = data_item["sites"]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # get all sites
        sites_ids.extend(
            [self.tokenizer.get_idx(site["element"]) for site in sorted_sites]
        )
        coordinates_mask.extend([0 for _ in range(len(sorted_sites))])

        # add space group
        sites_ids.append(self.tokenizer.sg_idx)
        coordinates_mask.append(0)
        if self.args.space_group:
            space_group_no = str(data_item["space_group"]["no"])
            space_group_tok = f"<sgn>{space_group_no}"
            sites_ids.append(self.tokenizer.get_idx(space_group_tok))
            coordinates_mask.append(0)

            # add special token
            sites_ids.append(self.tokenizer.coord_idx)
            coordinates_mask.append(0)

            # add order if needed
            if self.args.reorder:
                sites_ids.append(self.tokenizer.get_idx(self.tokenizer.order_tokens[0]))
                coordinates_mask.append(0)
        else:
            # mask for space group
            coordinates_mask.append(0)
            # mask for special token
            coordinates_mask.append(0)
            # mask for order
            if self.args.reorder:
                coordinates_mask.append(0)

        # add mask for lattice
        coordinates_mask.extend([1 for _ in range(3)])

        # add mask for coordinates
        coordinates_mask.extend([1 for _ in range(len(sorted_sites))])

        # add mask for eos
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_train_item_lan(self, index):
        item = dict()
        data_item = self.data[index]

        # sort sites if reorder
        if self.args.reorder:
            order = random.choice(self.tokenizer.order_tokens)
            sites = sort_sites(data_item["sites"], order)
        else:
            sites = data_item["sites"]

        # begin with bos    <bos>
        sites_ids = [self.tokenizer.bos_idx]

        # get all sites     <bos> H H O
        sites_ids.extend([self.tokenizer.get_idx(site["element"]) for site in sites])

        # add space group   <bos> H H O <sp> <sgn>222
        sites_ids.append(self.tokenizer.sg_idx)
        space_group_no = str(data_item["space_group"]["no"])
        space_group_tok = f"<sgn>{space_group_no}"
        sites_ids.append(self.tokenizer.get_idx(space_group_tok))

        # add special token <bos> H H O <sp> <sgn>222 <coord>
        sites_ids.append(self.tokenizer.coord_idx)

        # add order if needed
        if self.args.reorder:
            sites_ids.append(self.tokenizer.get_idx(order))

        # add lattice       <bos> H H O <sp> <sgn>222 <coord> 0 . 0 <cs> 0 . 0 ...
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        for l in lattice:
            for x in l:
                sites_ids.extend(
                    [self.tokenizer.get_idx(s) for s in tokenize_float(x, frac=False)]
                )
                sites_ids.append(self.tokenizer.cs_idx)

        # add coordinates
        coordinates = np.array(
            [site["fractional_coordinates"] for site in sites]
        ).astype(np.float32)
        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['formula']}"

        for coord in coordinates:
            for x in coord:
                sites_ids.extend(
                    [self.tokenizer.get_idx(s) for s in tokenize_float(x, frac=True)]
                )
                sites_ids.append(self.tokenizer.cs_idx)

        # eos
        sites_ids[-1] = self.tokenizer.eos_idx  # replace last '<cs>' with '<eos>'
        # sites_ids.append(self.tokenizer.eos_idx)

        sites_ids = np.array(sites_ids)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids

        return item

    def get_infer_item_lan(self, index):
        item = dict()
        data_item = self.data[index]

        sites = data_item["sites"]

        # begin with bos    <bos>
        sites_ids = [self.tokenizer.bos_idx]

        # get all sites     <bos> H H O
        sites_ids.extend([self.tokenizer.get_idx(site["element"]) for site in sites])

        # add space group   <bos> H H O <sp> <sgn>222
        sites_ids.append(self.tokenizer.sg_idx)
        if self.args.space_group:
            space_group_no = str(data_item["space_group"]["no"])
            space_group_tok = f"<sgn>{space_group_no}"
            sites_ids.append(self.tokenizer.get_idx(space_group_tok))

            # add special token <bos> H H O <sp> <sgn>222 <coord>
            sites_ids.append(self.tokenizer.coord_idx)

        sites_ids = np.array(sites_ids)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids

        return item

    def get_train_item_lanv2(self, index):
        item = dict()
        data_item = self.data[index]

        # sort sites if reorder
        if self.args.reorder:
            order = random.choice(self.tokenizer.order_tokens)
            sites = sort_sites(data_item["sites"], order)
        else:
            sites = data_item["sites"]

        # begin with bos    <bos>
        sites_ids = [self.tokenizer.bos_idx]

        # get all sites     <bos> H H O
        sites_ids.extend([self.tokenizer.get_idx(site["element"]) for site in sites])

        # add space group   <bos> H H O <sp> <sgn>222
        sites_ids.append(self.tokenizer.sg_idx)
        space_group_no = str(data_item["space_group"]["no"])
        space_group_tok = f"<sgn>{space_group_no}"
        sites_ids.append(self.tokenizer.get_idx(space_group_tok))

        # add special token <bos> H H O <sp> <sgn>222 <coord>
        sites_ids.append(self.tokenizer.coord_idx)

        # add order if needed
        if self.args.reorder:
            sites_ids.append(self.tokenizer.get_idx(order))

        # add lattice       <bos> H H O <sp> <sgn>222 <coord> 0 . 0 <cs> 0 . 0 ...
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        for l in lattice:
            for x in l:
                sites_ids.extend(
                    [
                        self.tokenizer.get_idx(s)
                        for s in tokenize_float_v2(x, frac=False)
                    ]
                )
                sites_ids.append(self.tokenizer.cs_idx)

        # add coordinates
        coordinates = np.array(
            [site["fractional_coordinates"] for site in sites]
        ).astype(np.float32)
        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['formula']}"

        for coord in coordinates:
            for x in coord:
                sites_ids.extend(
                    [self.tokenizer.get_idx(s) for s in tokenize_float_v2(x, frac=True)]
                )
                sites_ids.append(self.tokenizer.cs_idx)

        # eos
        sites_ids[-1] = self.tokenizer.eos_idx  # replace last '<cs>' with '<eos>'
        # sites_ids.append(self.tokenizer.eos_idx)

        sites_ids = np.array(sites_ids)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids

        return item

    def get_infer_item_lanv2(self, index):
        return self.get_infer_item_lan(index)

    def get_train_item_slices(self, index):
        item = dict()
        data_item = self.data[index]
        tokens = self.tokenizer.encode(data_item, prepend_bos=True, append_eos=True)
        item["id"] = index
        item["tokens"] = tokens

        return item

    def get_infer_item_slices(self, index):
        item = dict()
        data_item = self.data[index]
        if isinstance(data_item, str):
            slices = data_item.strip()
            tokens = self.tokenizer.tokenize(slices, prepend_bos=True, append_gen=True)
        else:
            sites = data_item["sites"]
            # begin with bos    <bos>
            tokens = [self.tokenizer.bos_idx]
            # get all sites     <bos> H H O
            tokens.extend([self.tokenizer.get_idx(site["element"]) for site in sites])
            tokens.append(self.tokenizer.gen_idx)
            tokens = np.array(tokens)
        item["id"] = index
        item["tokens"] = tokens

        return item

    def get_train_item(self, index):
        if isinstance(self.tokenizer, ThreeDimARGenNumTokenizer):
            return self.get_train_item_num(index)
        elif isinstance(self.tokenizer, ThreeDimARGenLanTokenizer):
            return self.get_train_item_lan(index)
        elif isinstance(self.tokenizer, ThreeDimARGenLanV2Tokenizer):
            return self.get_train_item_lanv2(index)
        elif isinstance(self.tokenizer, ThreeDimARGenSlicesTokenizer):
            return self.get_train_item_slices(index)
        else:
            raise ValueError(f"Unknown tokenizer type {self.tokenizer}")

    def get_infer_item(self, index):
        if isinstance(self.tokenizer, ThreeDimARGenNumTokenizer):
            return self.get_infer_item_num(index)
        elif isinstance(self.tokenizer, ThreeDimARGenLanTokenizer):
            return self.get_infer_item_lan(index)
        elif isinstance(self.tokenizer, ThreeDimARGenLanV2Tokenizer):
            return self.get_infer_item_lanv2(index)
        elif isinstance(self.tokenizer, ThreeDimARGenSlicesTokenizer):
            return self.get_infer_item_slices(index)
        else:
            raise ValueError(f"Unknown tokenizer type {self.tokenizer}")

    def __getitem__(self, index):
        if self.mode in [MODE.TRAIN, MODE.VAL]:
            return self.get_train_item(index)
        elif self.mode == MODE.INFER:
            return self.get_infer_item(index)

    def collate(self, samples):
        if self.args is None or self.args.pipeline_model_parallel_size == 0:
            if isinstance(self.tokenizer, ThreeDimARGenNumTokenizer):
                return collate_fn(samples, self.tokenizer, self.mode)
            else:
                return collate_fn_language(samples, self.tokenizer, self.mode)
        else:
            return collate_fn_pp(samples, self.tokenizer)

    def num_tokens(self, index: int) -> int:
        return self.sizes[index]

    @classmethod
    def from_file(cls, tokenizer, data_path, args=None, shuffle=True, mode=MODE.TRAIN):
        return cls(tokenizer, data_path, args, shuffle, mode)

    @classmethod
    def from_dict(cls, tokenizer, data, args=None, shuffle=True, mode=MODE.TRAIN):
        dataset = cls(tokenizer, args=args, shuffle=shuffle, mode=mode)
        dataset.load_dict(data)
        return dataset

    @classmethod
    def from_json(cls, tokenizer, data, args=None, shuffle=True, mode=MODE.TRAIN):
        dataset = cls(tokenizer, args=args, shuffle=shuffle, mode=mode)
        dataset.load_json(data)
        return dataset


class ThreeDimARGenEnergyDataset(ThreeDimARGenDataset):
    def __init__(
        self,
        tokenizer,
        data_path: Union[str, list[str]],
        args=None,
        shuffle: bool = True,
        mode=MODE.TRAIN,
    ):
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.max_position_embeddings = args.max_position_embeddings

        if data_path.count(",") > 0:
            data_path = data_path.split(",")

        if isinstance(data_path, str):
            data_path = [data_path]

        self.data = []
        self.sizes = []
        for path in data_path:
            self.load_data(path)

        if shuffle:
            random.shuffle(self.data)

        if self.args.scale_coords:
            logger.info(f"scale digit with scale {self.args.scale_coords}")

    def get_sequence_length(self, data_item):
        n = len(data_item["sites"])
        if isinstance(self.tokenizer, ThreeDimARGenNumEnergyTokenizer):
            # <bos> [n] <coords> [3 lattice] [n coords] <energy> energy <eos>
            return 1 + n + 1 + 3 + n + 1 + 1 + 1
        elif isinstance(self.tokenizer, ThreeDimARGenLanEnergyTokenizer):
            # <bos> [n] <coords> [9 lattice * (7 digits + <cs>)] [n * (3 coords * (6 digits + <cs>))] <energy> [6 ~ 10 energy digits] <eos>
            return 1 + n + 1 + 9 * (7 + 1) + n * (3 * (6 + 1)) + 1 + 10 + 1
        elif isinstance(self.tokenizer, ThreeDimARGenLanV2EnergyTokenizer):
            # <bos> [n] <coords> [9 lattice * (4 tokens + <cs>)] [n * (3 coords * (3 tokens + <cs>))] <energy> [max 5 energy tokens] <eos>
            return 1 + n + 1 + 9 * (4 + 1) + n * (3 * (3 + 1)) + 1 + 5 + 1

    def get_train_item_num(self, index):
        item = dict()
        data_item = self.data[index]

        sites = data_item["sites"]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # get all sites
        sites_ids.extend([self.tokenizer.get_idx(site["element"]) for site in sites])
        coordinates_mask.extend([0 for _ in range(len(sites))])

        # add coord token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add lattice
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(3)])
        coordinates_mask.extend([1 for _ in range(3)])

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(sites))])
        coordinates_mask.extend([1 for _ in range(len(sites))])
        coordinates = np.array(
            [site["fractional_coordinates"] for site in sites]
        ).astype(np.float32)
        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        # add energy
        sites_ids.append(self.tokenizer.energy_idx)
        coordinates_mask.append(0)
        energy = data_item["energy"]
        if self.args.scale_energy:
            energy = energy * self.args.scale_energy
        sites_ids.append(self.tokenizer.mask_idx)
        coordinates_mask.append(2)
        energy = np.array([[energy, energy, energy]]).astype(np.float32)

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates = np.concatenate([lattice, coordinates, energy], axis=0)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def __get_infer_item_num__(self, index):
        item = dict()
        data_item = self.data[index]

        sites = data_item["sites"]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # get all sites
        sites_ids.extend([self.tokenizer.get_idx(site["element"]) for site in sites])
        coordinates_mask.extend([0 for _ in range(len(sites))])

        # add coord token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add lattice
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(3)])
        coordinates_mask.extend([1 for _ in range(3)])

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(sites))])
        coordinates_mask.extend([1 for _ in range(len(sites))])
        coordinates = np.array(
            [site["fractional_coordinates"] for site in sites]
        ).astype(np.float32)
        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        # add energy
        sites_ids.append(self.tokenizer.energy_idx)
        coordinates_mask.append(0)
        # add mask for energy
        coordinates_mask.append(2)

        # add mask for eos
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates = np.concatenate([lattice, coordinates], axis=0)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_train_item_lan(self, index):
        item = dict()
        data_item = self.data[index]

        sites = data_item["sites"]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]

        # get all sites
        sites_ids.extend([self.tokenizer.get_idx(site["element"]) for site in sites])

        # add coord token
        sites_ids.append(self.tokenizer.coord_idx)

        # add lattice
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        for l in lattice:
            for x in l:
                sites_ids.extend(
                    [self.tokenizer.get_idx(s) for s in tokenize_float(x, frac=False)]
                )
                sites_ids.append(self.tokenizer.cs_idx)

        # add coordinates
        coordinates = np.array(
            [site["fractional_coordinates"] for site in sites]
        ).astype(np.float32)

        for coord in coordinates:
            for x in coord:
                sites_ids.extend(
                    [self.tokenizer.get_idx(s) for s in tokenize_float(x, frac=True)]
                )
                sites_ids.append(self.tokenizer.cs_idx)

        # add energy
        sites_ids.append(self.tokenizer.energy_idx)
        energy = data_item["energy"]
        sites_ids.extend(
            [self.tokenizer.get_idx(s) for s in tokenize_float(energy, frac=False)]
        )

        # eos
        sites_ids.append(self.tokenizer.eos_idx)

        sites_ids = np.array(sites_ids)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        return item

    def get_infer_item_lan(self, index):
        item = dict()
        data_item = self.data[index]

        sites = data_item["sites"]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]

        # get all sites
        sites_ids.extend([self.tokenizer.get_idx(site["element"]) for site in sites])

        # add coord token
        sites_ids.append(self.tokenizer.coord_idx)

        # add lattice
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        for l in lattice:
            for x in l:
                sites_ids.extend(
                    [self.tokenizer.get_idx(s) for s in tokenize_float(x, frac=False)]
                )
                sites_ids.append(self.tokenizer.cs_idx)

        # add coordinates
        coordinates = np.array(
            [site["fractional_coordinates"] for site in sites]
        ).astype(np.float32)

        for coord in coordinates:
            for x in coord:
                sites_ids.extend(
                    [self.tokenizer.get_idx(s) for s in tokenize_float(x, frac=True)]
                )
                sites_ids.append(self.tokenizer.cs_idx)

        # add energy
        sites_ids.append(self.tokenizer.energy_idx)

        if (
            len(sites_ids) > self.args.max_position_embeddings - 11
        ):  # leave for <eos> and at leat 10 tokens for energy
            sites_ids = sites_ids[: self.args.max_position_embeddings - 11]
            right_most_cs_idx = (
                len(sites_ids) - sites_ids[::-1].index(self.tokenizer.cs_idx) - 1
            )
            sites_ids = sites_ids[: right_most_cs_idx + 1] + [self.tokenizer.energy_idx]

        sites_ids = np.array(sites_ids)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids

        return item

    def get_train_item_lanv2(self, index):
        item = dict()
        data_item = self.data[index]

        sites = data_item["sites"]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]

        # get all sites
        sites_ids.extend([self.tokenizer.get_idx(site["element"]) for site in sites])

        # add coord token
        sites_ids.append(self.tokenizer.coord_idx)

        # add lattice
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        for l in lattice:
            for x in l:
                sites_ids.extend(
                    [
                        self.tokenizer.get_idx(s)
                        for s in tokenize_float_v2(x, frac=False)
                    ]
                )
                sites_ids.append(self.tokenizer.cs_idx)

        # add coordinates
        coordinates = np.array(
            [site["fractional_coordinates"] for site in sites]
        ).astype(np.float32)

        for coord in coordinates:
            for x in coord:
                sites_ids.extend(
                    [self.tokenizer.get_idx(s) for s in tokenize_float_v2(x, frac=True)]
                )
                sites_ids.append(self.tokenizer.cs_idx)

        # add energy
        sites_ids.append(self.tokenizer.energy_idx)
        energy = data_item["energy"]
        sites_ids.extend(
            [self.tokenizer.get_idx(s) for s in tokenize_float_v2(energy, frac=False)]
        )

        # eos
        sites_ids.append(self.tokenizer.eos_idx)

        sites_ids = np.array(sites_ids)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        return item

    def get_infer_item_lanv2(self, index):
        return self.get_infer_item_lan(index)


if __name__ == "__main__":

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    # args = Namespace(max_sites=500, scale_coords=10, scale_energy=10, reorder=False)
    # args.dict_path = os.path.join(
    #     os.path.dirname(os.path.abspath(__file__)), "dict_.txt"
    # )
    # tokenizer = ThreeDimARGenLanTokenizer.from_file(args.dict_path, args)
    # args.train_data_path = os.path.join("/home/yinxia/Wei/SFM_framework/", "data.jsonl")

    # print(args)
    # print("=================")
    # print("Test ThreeDimGen dataset")
    # dataset = ThreeDimARGenDataset(tokenizer, args.train_data_path, args, shuffle=False)
    # print(len(dataset))
    # print(dataset[12])
    # print(dataset[512])
    # print(dataset.collate([dataset[0], dataset[100], dataset[500], dataset[800]]))

    # print("=================")
    # print("Test ThreeDimEnery dataset")
    # tokenizer = ThreeDimTokenizer.from_file(args.dict_path, args)
    # dataset = ThreeDimEneryDataset(tokenizer, args.train_data_path, args, shuffle=False)

    # dataset = ThreeDimGenDataset(tokenizer, "/blob/v-yihangzhai/materials_data/mp/materialProject_all_20230717_props_sharedByTian_sg.jsonl", shuffle=False)
    # for d in dataset:
    # print()
    # print(batch_dataset.collate([dataset[6123], dataset[6001], dataset[6299], dataset[6599]]))

    print("=================")
    print("Test ThreeDimSlices dataset")
    args = Namespace(
        max_sites=None,
        scale_coords=None,
        scale_energy=None,
        reorder=False,
        tokenizer="slices",
        max_position_embeddings=2048,
        pipeline_model_parallel_size=0,
    )
    args.dict_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dict_slices.txt"
    )
    tokenizer = ThreeDimARGenTokenizer.from_file(args.dict_path, args)
    dataset = ThreeDimARGenDataset(
        tokenizer,
        "/hai1/SFM/threedimargen/data/materials_data/mp_nomad_qmdb_train_slices00",
        args,
        shuffle=False,
    )
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    print(dataset.collate([dataset[0], dataset[1], dataset[2], dataset[3]]))
