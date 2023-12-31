# -*- coding: utf-8 -*-
from typing import List, Union

import numpy as np
import torch

from sfm.logging import logger

from .vocalubary import Alphabet


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
    # (N, d) -> (1, padlen, d)
    xlen, xdim = x.size()
    assert (
        start + xlen <= padlen
    ), f"padlen {padlen} is too small for xlen {xlen} and start point {start}"
    new_x = x.new_full([padlen, xdim], pad_num, dtype=x.dtype)
    new_x[start : start + xlen, :] = x
    x = new_x
    return x.unsqueeze(0)


def pad_2d_square_unsqueeze(
    x: torch.Tensor, padlen: int, start: int, pad_num: Union[int, float]
):
    # (N, N) -> (1, padlen, padlen)
    xlen = x.size(0)
    assert (
        start + xlen <= padlen
    ), f"padlen {padlen} is too small for xlen {xlen} and start point {start}"
    new_x = x.new_full([padlen, padlen], pad_num, dtype=x.dtype)
    new_x[start : start + xlen, start : start + xlen] = x
    x = new_x
    return x.unsqueeze(0)


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(-1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def collate_fn(samples: List[dict], vocab: Alphabet):
    """
    Overload BaseWrapperDataset.collater
    May be future changes need config

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """
    # max_tokens = Nres+2 (<cls> and <eos>)
    max_tokens = max(len(s["aa"]) for s in samples)

    offset = int(vocab.prepend_bos)
    batch = dict()

    batch["id"] = torch.tensor([s["id"] for s in samples], dtype=torch.long)
    batch["name"] = [s["name"] for s in samples]
    # naa = [Nres+2, ...] for each sample
    batch["naa"] = torch.tensor([len(s["aa"]) for s in samples], dtype=torch.long)

    # (Nres+2,) -> (B, Nres+2)
    batch["x"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["aa"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    )
    # logger.debug("naa: {}".format(batch["x"].shape))
    for prop_idx, prop_name in enumerate(
        ["chem_polar", "net_charge", "hydropathy", "mol_mass"]
    ):
        batch[prop_name] = torch.cat(
            [
                pad_1d_unsqueeze(
                    torch.from_numpy(s[prop_name]),
                    max_tokens,
                    0,
                    vocab.unk_prop_feat[prop_idx],
                )
                for s in samples
            ]
        )
    batch["hydropathy"] = batch["hydropathy"].unsqueeze(-1)
    batch["mol_mass"] = batch["mol_mass"].unsqueeze(-1)

    batch["masked_aa"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["masked_aa"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    ).unsqueeze(-1)

    batch["mask_pos"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["mask_pos"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    ).unsqueeze(-1)

    # batch["mask"] = torch.cat(
    #     [
    #         pad_1d_unsqueeze(
    #             torch.from_numpy(s["mask"]), max_tokens, 0, vocab.padding_idx
    #         )
    #         for s in samples
    #     ]
    # )
    # batch["replace_mask"] = torch.cat(
    #     [
    #         pad_1d_unsqueeze(
    #             torch.from_numpy(s["replace_mask"]), max_tokens, 0, vocab.padding_idx
    #         )
    #         for s in samples
    #     ]
    # )
    # for confidence score, mind the prepended <cls>
    batch["conf"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["conf"]), max_tokens, offset, vocab.padding_idx
            )
            for s in samples
        ]
    )

    # (Nres, 3) -> (B, Nres+2, 3)
    batch["pos"] = torch.cat(
        [
            pad_2d_unsqueeze(torch.from_numpy(s["pos"]), max_tokens, offset, 0.0)
            for s in samples
        ]
    )
    batch["ang"] = torch.cat(
        [
            pad_2d_unsqueeze(torch.from_numpy(s["ang"]), max_tokens, offset, torch.inf)
            for s in samples
        ]
    )

    # batch["pos_noise"] = torch.cat(
    #     [
    #         pad_2d_unsqueeze(
    #             torch.from_numpy(s["pos_noise"]), max_tokens, offset, torch.inf
    #         )
    #         for s in samples
    #     ]
    # )
    batch["ang_noise"] = torch.cat(
        [
            pad_2d_unsqueeze(
                torch.from_numpy(s["ang_noise"]), max_tokens, offset, torch.inf
            )
            for s in samples
        ]
    )

    # calculate distance matrix
    # (B, Nres+2, 3) -> (B, Nres+2, Nres+2)
    batch["dist"] = torch.cat(
        [
            pad_2d_square_unsqueeze(
                torch.from_numpy(
                    np.linalg.norm(s["pos"][:, None, :] - s["pos"][None, :, :], axis=-1)
                ),
                max_tokens,
                1,
                torch.inf,
            )
            for s in samples
        ]
    )

    # node type edges, used in 3d embedding
    node_type_edges = []
    ngraph, nnodes = batch["x"].shape[:2]

    for idx in range(ngraph):
        node_atom_type = batch["x"][idx]
        node_atom_i = (
            node_atom_type.unsqueeze(-1).repeat(1, nnodes).unsqueeze(0).unsqueeze(-1)
        )
        node_atom_j = (
            node_atom_type.unsqueeze(0).repeat(nnodes, 1).unsqueeze(0).unsqueeze(-1)
        )
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)
    batch["node_type_edge"] = node_type_edge
    # print("node_type_edge", node_type_edge.shape); exit()
    return batch


def collate_ur50_fn(samples: List[dict], vocab: Alphabet):
    """
    Overload BaseWrapperDataset.collater
    May be future changes need config

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """
    # max_tokens = Nres+2 (<cls> and <eos>)
    max_tokens = max(len(s["aa"]) for s in samples)

    int(vocab.prepend_bos)
    batch = dict()

    # batch["id"] = torch.tensor([s["id"] for s in samples], dtype=torch.long)
    batch["naa"] = torch.tensor([len(s["aa"]) for s in samples], dtype=torch.long)

    # (Nres+2,) -> (B, Nres+2)
    batch["x"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["aa"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    )

    batch["x_new"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["new_seq"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    )

    # logger.debug("naa: {}".format(batch["x"].shape))
    for prop_idx, prop_name in enumerate(
        ["chem_polar", "net_charge", "hydropathy", "mol_mass"]
    ):
        batch[prop_name] = torch.cat(
            [
                pad_1d_unsqueeze(
                    torch.from_numpy(s[prop_name]),
                    max_tokens,
                    0,
                    vocab.unk_prop_feat[prop_idx],
                )
                for s in samples
            ]
        )
    batch["hydropathy"] = batch["hydropathy"].unsqueeze(-1)
    batch["mol_mass"] = batch["mol_mass"].unsqueeze(-1)

    batch["masked_aa"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["masked_aa"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    ).unsqueeze(-1)

    batch["mask_pos"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["mask_pos"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    ).unsqueeze(-1)

    return batch


def collate_downstream_fn(samples: List[dict], vocab: Alphabet):
    """
    Overload BaseWrapperDataset.collater
    May be future changes need config

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """
    # max_tokens = Nres+2 (<cls> and <eos>)
    max_tokens = max(len(s["aa"]) for s in samples)

    int(vocab.prepend_bos)
    batch = dict()

    batch["id"] = torch.tensor([s["id"] for s in samples], dtype=torch.long)
    batch["naa"] = torch.tensor([len(s["aa"]) for s in samples], dtype=torch.long)

    # (Nres+2,) -> (B, Nres+2)
    batch["x"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["aa"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    )
    # logger.debug("naa: {}".format(batch["x"].shape))
    for prop_idx, prop_name in enumerate(
        ["chem_polar", "net_charge", "hydropathy", "mol_mass"]
    ):
        batch[prop_name] = torch.cat(
            [
                pad_1d_unsqueeze(
                    torch.from_numpy(s[prop_name]),
                    max_tokens,
                    0,
                    vocab.unk_prop_feat[prop_idx],
                )
                for s in samples
            ]
        )
    batch["hydropathy"] = batch["hydropathy"].unsqueeze(-1)
    batch["mol_mass"] = batch["mol_mass"].unsqueeze(-1)
    batch["target"] = torch.cat([torch.from_numpy(s["target"]) for s in samples])
    batch["target_offset"] = torch.tensor(
        [len(s["target"]) for s in samples], dtype=torch.long
    )

    return batch


def collate_multiseq_downstream_fn(samples: List[dict], vocab: Alphabet):
    """
    Overload BaseWrapperDataset.collater
    May be future changes need config

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """
    # max_tokens = Nres+2 (<cls> and <eos>)
    batch = dict()
    batch["id"] = torch.tensor([s["id"] for s in samples], dtype=torch.long)
    batch["target"] = torch.cat([torch.from_numpy(s["target"]) for s in samples])
    batch["target_offset"] = torch.tensor(
        [len(s["target"]) for s in samples], dtype=torch.long
    )

    idx = 0
    while True:
        if f"aa_{idx}" not in samples[0]:
            break

        max_tokens = max(len(s[f"aa_{idx}"]) for s in samples)
        batch[f"naa_{idx}"] = torch.tensor(
            [len(s[f"aa_{idx}"]) for s in samples], dtype=torch.long
        )

        # (Nres+2,) -> (B, Nres+2)
        batch[f"x_{idx}"] = torch.cat(
            [
                pad_1d_unsqueeze(
                    torch.from_numpy(s[f"aa_{idx}"]), max_tokens, 0, vocab.padding_idx
                )
                for s in samples
            ]
        )
        # logger.debug("naa: {}".format(batch["x"].shape))
        for prop_idx, prop_name in enumerate(
            [
                f"chem_polar_{idx}",
                f"net_charge_{idx}",
                f"hydropathy_{idx}",
                f"mol_mass_{idx}",
            ]
        ):
            batch[prop_name] = torch.cat(
                [
                    pad_1d_unsqueeze(
                        torch.from_numpy(s[prop_name]),
                        max_tokens,
                        0,
                        vocab.unk_prop_feat[prop_idx],
                    )
                    for s in samples
                ]
            )
        batch[f"hydropathy_{idx}"] = batch[f"hydropathy_{idx}"].unsqueeze(-1)
        batch[f"mol_mass_{idx}"] = batch[f"mol_mass_{idx}"].unsqueeze(-1)
        idx += 1
    return batch


def collate_secondary_structure_fn(samples: List[dict], vocab: Alphabet):
    """
    Overload BaseWrapperDataset.collater
    May be future changes need config

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """
    # max_tokens = Nres+2 (<cls> and <eos>)
    max_tokens = max(len(s["aa"]) for s in samples)

    int(vocab.prepend_bos)
    batch = dict()

    batch["id"] = torch.tensor([s["id"] for s in samples], dtype=torch.long)
    batch["naa"] = torch.tensor([len(s["aa"]) for s in samples], dtype=torch.long)

    # (Nres+2,) -> (B, Nres+2)
    batch["x"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["aa"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    )
    # logger.debug("naa: {}".format(batch["x"].shape))
    for prop_idx, prop_name in enumerate(
        ["chem_polar", "net_charge", "hydropathy", "mol_mass"]
    ):
        batch[prop_name] = torch.cat(
            [
                pad_1d_unsqueeze(
                    torch.from_numpy(s[prop_name]),
                    max_tokens,
                    0,
                    vocab.unk_prop_feat[prop_idx],
                )
                for s in samples
            ]
        )
    batch["hydropathy"] = batch["hydropathy"].unsqueeze(-1)
    batch["mol_mass"] = batch["mol_mass"].unsqueeze(-1)
    batch["target"] = torch.cat(
        [
            pad_1d_unsqueeze(torch.from_numpy(s["target"]), max_tokens, 0, 0)
            for s in samples
        ]
    )
    batch["target_mask"] = torch.cat(
        [
            pad_1d_unsqueeze(torch.from_numpy(s["target_mask"]), max_tokens, 0, 0)
            for s in samples
        ]
    )
    # batch["target_offset"] = torch.tensor([len(s['target']) for s in samples], dtype=torch.long)
    return batch
