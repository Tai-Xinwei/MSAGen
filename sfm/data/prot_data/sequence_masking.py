# -*- coding: utf-8 -*-
import logging
from copy import deepcopy
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def no_sequence_masking(
    item: dict,
    args: dict,
    seed: int,
    mask_idx: int,
    standard_toks: List[int],
):
    seq = item["aa"]
    size = len(seq)
    return deepcopy(seq), np.full(size, False), np.full(size, False)


def BERT_sequence_masking(
    item: dict,
    args: dict,
    seed: int,
    mask_idx: int,
    standard_toks_idx: List[int],
):
    mask_prob = args.mask_prob
    leave_unmasked_prob = args.leave_unmasked_prob
    random_token_prob = args.random_token_prob
    mask_multiple_length = args.mask_multiple_length
    mask_stdev = args.mask_stdev

    assert 0.0 < mask_prob < 1.0
    assert 0.0 <= random_token_prob <= 1.0
    assert 0.0 <= leave_unmasked_prob <= 1.0
    assert random_token_prob + leave_unmasked_prob <= 1.0
    assert mask_multiple_length >= 1
    assert mask_stdev >= 0.0
    rng = np.random.default_rng(seed)
    # decide elements to mask
    seq = item["aa"]
    size = len(seq)
    mask = np.full(size, False)
    # at least mask one element or one span
    num_mask = int(mask_prob * size / float(mask_multiple_length) + 1)
    # print("aa", seq)

    # GLM like masking
    mask_idc = rng.choice(size, num_mask, replace=False)
    if mask_stdev > 0.0:
        lengths = rng.normal(mask_multiple_length, mask_stdev, size=num_mask)
        lengths = [max(0, int(round(x))) for x in lengths]
        mask_idc = np.asarray(
            [
                mask_idc[j] + offset
                for j in range(len(mask_idc))
                for offset in range(lengths[j])
            ],
            dtype=np.int64,
        )
    else:
        mask_idc = np.concatenate([mask_idc + i for i in range(mask_multiple_length)])
    mask_idc = mask_idc[mask_idc < len(mask)]
    try:
        mask[mask_idc] = True
    except:  # something wrong
        logging.error(f"Assigning mask indexes {mask_idc} to mask {mask} failed!")
        raise

    total_mask = deepcopy(mask)

    # BERT like masking: random token replacement and unmasking
    # decide unmasking and random replacement
    rand_or_unmask_prob = random_token_prob + leave_unmasked_prob
    if rand_or_unmask_prob > 0.0:
        rand_or_unmask = mask & (rng.random(size) < rand_or_unmask_prob)
        if random_token_prob == 0.0:
            unmask = rand_or_unmask
            rand_mask = None
        elif leave_unmasked_prob == 0.0:
            unmask = None
            rand_mask = rand_or_unmask
        else:
            unmask_prob = leave_unmasked_prob / rand_or_unmask_prob
            decision = rng.random(size) < unmask_prob
            unmask = rand_or_unmask & decision
            rand_mask = rand_or_unmask & (~decision)
    else:
        unmask = rand_mask = None

    if unmask is not None:
        mask = mask ^ unmask

    new_seq = deepcopy(seq)
    new_seq[mask] = mask_idx
    if rand_mask is not None:
        num_rand = rand_mask.sum()
        if num_rand > 0:
            new_seq[rand_mask] = rng.choice(
                np.array(standard_toks_idx),
                num_rand,
                # p=self.weights, assume equal weights
            )
    return new_seq, total_mask, rand_mask


def BPE_BERT_sequence_masking(
    item: dict,
    args: dict,
    seed: int,
    mask_idx: int,
    standard_toks_idx: List[int],
):
    mask_prob = args.mask_prob
    leave_unmasked_prob = args.leave_unmasked_prob
    random_token_prob = args.random_token_prob
    mask_multiple_length = args.mask_multiple_length
    mask_stdev = args.mask_stdev

    assert 0.0 < mask_prob < 1.0
    assert 0.0 <= random_token_prob <= 1.0
    assert 0.0 <= leave_unmasked_prob <= 1.0
    assert random_token_prob + leave_unmasked_prob <= 1.0
    assert mask_multiple_length >= 1
    assert mask_stdev >= 0.0
    rng = np.random.default_rng(seed)
    # decide elements to mask
    seq = item["aa"]
    bpe = item["bpe"]

    np.diff(bpe, prepend=bpe[0]) != 0
    unique_values, indices = np.unique(bpe, return_index=True)
    sorted_indices = np.sort(indices)

    # generate whole word mask
    ww_size = len(sorted_indices)
    ww_mask = np.full(ww_size, False)

    size = len(seq)
    mask = np.full(size, False)
    # at least mask one element or one span
    num_mask = int(mask_prob * ww_size / float(mask_multiple_length) + 1)

    # GLM like masking
    mask_idc = rng.choice(ww_size, num_mask, replace=False)
    if mask_stdev > 0.0:
        lengths = rng.normal(mask_multiple_length, mask_stdev, size=num_mask)
        lengths = [max(0, int(round(x))) for x in lengths]
        mask_idc = np.asarray(
            [
                mask_idc[j] + offset
                for j in range(len(mask_idc))
                for offset in range(lengths[j])
            ],
            dtype=np.int64,
        )
    else:
        mask_idc = np.concatenate([mask_idc + i for i in range(mask_multiple_length)])
    mask_idc = mask_idc[mask_idc < len(ww_mask)]
    try:
        ww_mask[mask_idc] = True
    except:  # something wrong
        logging.error(f"Assigning mask indexes {mask_idc} to mask {mask} failed!")
        raise

    sorted_indices = np.append(sorted_indices, len(bpe))
    for idx in mask_idc:
        mask[sorted_indices[idx] : sorted_indices[idx + 1]] = True
    total_mask = deepcopy(mask)

    # BERT like masking: random token replacement and unmasking
    # decide unmasking and random replacement
    rand_or_unmask_prob = random_token_prob + leave_unmasked_prob
    if rand_or_unmask_prob > 0.0:
        rand_or_unmask = mask & (rng.random(size) < rand_or_unmask_prob)
        if random_token_prob == 0.0:
            unmask = rand_or_unmask
            rand_mask = None
        elif leave_unmasked_prob == 0.0:
            unmask = None
            rand_mask = rand_or_unmask
        else:
            unmask_prob = leave_unmasked_prob / rand_or_unmask_prob
            decision = rng.random(size) < unmask_prob
            unmask = rand_or_unmask & decision
            rand_mask = rand_or_unmask & (~decision)
    else:
        unmask = rand_mask = None

    if unmask is not None:
        mask = mask ^ unmask

    new_seq = deepcopy(seq)
    new_seq[mask] = mask_idx
    if rand_mask is not None:
        num_rand = rand_mask.sum()
        if num_rand > 0:
            new_seq[rand_mask] = rng.choice(
                np.array(standard_toks_idx),
                num_rand,
                # p=self.weights, assume equal weights
            )
    return new_seq, total_mask, rand_mask


def transformerM_masking(
    item: dict,
    args: dict,
    seed: int,
    mask_idx: int,
    standard_toks: List[int],
):
    mask_prob = args.mask_ratio
    leave_unmasked_prob = args.leave_unmasked_prob
    random_token_prob = args.random_token_prob
    mask_multiple_length = args.mask_multiple_length
    mask_stdev = args.mask_stdev

    assert 0.0 <= mask_prob < 1.0
    assert 0.0 <= random_token_prob <= 1.0
    assert 0.0 <= leave_unmasked_prob <= 1.0
    assert random_token_prob + leave_unmasked_prob <= 1.0
    assert mask_multiple_length >= 1
    assert mask_stdev >= 0.0

    # decide elements to mask
    seq = item["aa"]
    size = len(seq)
    mask_type = np.full(size, False)
    mask_pos = np.full(size, False)

    # at least mask one element or one span, do not mask cls and eos
    num_mask = int(mask_prob * (size - 2) / float(mask_multiple_length) + 1)
    # FIXME: mask_multiple_length is not used
    assert (
        mask_multiple_length == 1
    ), "mask_multiple_length should be 1 for transformerM"

    # GLM like masking
    mask_type_idc = np.random.choice(size - 2, num_mask, replace=False)
    mask_pos_idc = np.random.choice(size - 2, num_mask, replace=False)

    # mask_type_idc = mask_type_idc[mask_type_idc < len(mask_type_idc)]
    # mask_pos_idc = mask_pos_idc[mask_pos_idc < len(mask_pos_idc)]
    try:
        mask_type[mask_type_idc + 1] = True
    except:  # something wrong
        logging.error(
            f"Assigning mask indexes {mask_type_idc} to mask {mask_type} failed!"
        )
        raise

    new_seq = deepcopy(seq)
    new_seq[mask_type] = mask_idx

    try:
        mask_pos[mask_pos_idc] = True
    except:  # something wrong
        logging.error(
            f"Assigning mask indexes {mask_pos_idc} to mask {mask_pos} failed!"
        )
        raise

    return new_seq, mask_type, mask_pos


def continuous_masking(
    item: dict,
    args: dict,
    seed: int,
    mask_idx: int,
    standard_toks: List[int],
):
    mask_prob = args.mask_ratio
    leave_unmasked_prob = args.leave_unmasked_prob
    random_token_prob = args.random_token_prob
    mask_multiple_length = args.mask_multiple_length
    mask_stdev = args.mask_stdev

    assert 0.0 <= mask_prob < 1.0
    assert 0.0 <= random_token_prob <= 1.0
    assert 0.0 <= leave_unmasked_prob <= 1.0
    assert random_token_prob + leave_unmasked_prob <= 1.0
    assert mask_multiple_length >= 1
    assert mask_stdev >= 0.0

    # decide elements to mask
    seq = item["aa"]
    size = len(seq)
    mask_type = np.full(size, False)
    mask_pos = np.full(size, False)

    # at least mask one element or one span, do not mask cls and eos
    num_mask = int(mask_prob * (size - 2) / float(mask_multiple_length) + 1)

    # WWM like masking
    mask_type_idc = np.random.choice(size - 2 - num_mask, 1, replace=False)[0]
    mask_pos_idc = np.random.choice(size - 2 - num_mask, 1, replace=False)[0]

    # mask_type_idc = mask_type_idc[mask_type_idc < len(mask_type_idc)]
    # mask_pos_idc = mask_pos_idc[mask_pos_idc < len(mask_pos_idc)]
    try:
        mask_type[mask_type_idc : mask_type_idc + num_mask] = True
    except:  # something wrong
        logging.error(
            f"Assigning mask indexes {mask_type_idc} to mask {mask_type} failed!"
        )
        raise

    new_seq = deepcopy(seq)
    new_seq[mask_type] = mask_idx

    try:
        mask_pos[mask_pos_idc : mask_pos_idc + num_mask] = True
    except:  # something wrong
        logging.error(
            f"Assigning mask indexes {mask_pos_idc} to mask {mask_pos} failed!"
        )
        raise

    return new_seq, mask_type, mask_pos


masking_registry = {
    "no": no_sequence_masking,
    "bert": BERT_sequence_masking,
    "bpe_bert": BPE_BERT_sequence_masking,
    "transformerM": transformerM_masking,
    "continuousMask": continuous_masking,
    # TODO: add more
}
