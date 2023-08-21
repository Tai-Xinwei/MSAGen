# -*- coding: utf-8 -*-
from typing import List, Union

import numpy as np
import torch
from vocalubary import Alphabet


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
    batch["aa"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["aa"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    )
    batch["masked_aa"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["masked_aa"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    )
    batch["mask"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["mask"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    )
    batch["replace_mask"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["replace_mask"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    )
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
            pad_2d_unsqueeze(torch.from_numpy(s["pos"]), max_tokens, offset, torch.inf)
            for s in samples
        ]
    )
    batch["ang"] = torch.cat(
        [
            pad_2d_unsqueeze(torch.from_numpy(s["ang"]), max_tokens, offset, torch.inf)
            for s in samples
        ]
    )

    batch["pos_noise"] = torch.cat(
        [
            pad_2d_unsqueeze(
                torch.from_numpy(s["pos_noise"]), max_tokens, offset, torch.inf
            )
            for s in samples
        ]
    )
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

    return batch
