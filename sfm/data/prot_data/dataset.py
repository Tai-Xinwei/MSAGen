# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Any, List

import lmdb
import numpy as np
from collater import collate_fn
from process import bstr2obj
from sequence_masking import masking_registry
from spatial_noise import noise_registry
from vocalubary import Alphabet

from sfm.data.dataset import FoundationModelDataset

logger = logging.getLogger(__name__)


class ProteinLMDBDataset(FoundationModelDataset):
    """
    This is a dataset for protein information, including amino acid, position, angles and confidence score.
    All the information are raw data. Please ues other dataset to process the data, eg, tokenize, encode...

    The process pipeline will be changed in the future, but the interface will not change.
    """

    def __init__(self, args: Any) -> None:
        super().__init__()

        self.args = self.set_default_args(args)

        logger.info(args)

        self.lmdb_path = Path(args.lmdb_path)
        assert self.lmdb_path.is_dir(), f"Processed file not found: {self.lmdb_path}"

        self.vocab = Alphabet.from_architecture(args.vocab)

        self.seed = args.seed
        self.seq_masking_method = args.seq_masking_method

        self.noise_method = args.noise_method
        self.pos_noise = args.pos_noise
        self.ang_noise = args.ang_noise

        self.env = lmdb.open(
            str(self.lmdb_path), subdir=True, readonly=True, lock=False, readahead=False
        )
        self.txn = self.env.begin(write=False)

        metadata = bstr2obj(self.txn.get("metadata".encode()))
        self.sizes, self.names = metadata["sizes"], metadata["names"]
        self.comment = metadata["comment"]

    def set_default_args(self, args):
        args.lmdb_path = getattr(args, "lmdb_path", None)
        args.vocab = getattr(args, "vocab", "ESM-1b")

        args.seed = getattr(args, "seed", "2023")
        args.seq_masking_method = getattr(args, "seq_masking_method", "bert")

        args.mask_prob = getattr(args, "mask_prob", 0.15)
        args.leave_unmasked_prob = getattr(args, "leave_unmasked_prob", 0.1)
        args.random_token_prob = getattr(args, "random_token_prob", 0.1)
        args.mask_multiple_length = getattr(args, "mask_multiple_length", 1)
        args.mask_stdev = getattr(args, "mask_stdev", 0.0)

        args.noise_method = getattr(args, "noise_method", "normal")
        args.pos_noise = getattr(args, "pos_noise", True)
        args.ang_noise = getattr(args, "ang_noise", True)

        args.coord_noise_mean = getattr(args, "coord_noise_mean", 0.0)
        args.coord_noise_stdev = getattr(args, "coord_noise_stdev", 1.0)
        args.angle_noise_mean = getattr(args, "angle_noise_mean", 0.0)
        args.angle_noise_stdev = getattr(args, "angle_noise_stdev", 1.0)

        return args

    def __getitem__(self, index: int) -> dict:
        key = self.names[index]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = bstr2obj(value)
        item = {"id": index, **data}

        """
        - convert string sequence to int index
        """
        tokens = [self.vocab.tok_to_idx[tok] for tok in item["aa"]]
        if self.vocab.prepend_bos:
            tokens.insert(0, self.vocab.cls_idx)
        if self.vocab.append_eos:
            tokens.append(self.vocab.eos_idx)
        item["aa"] = np.array(tokens, dtype=np.int64)

        """
        - mask the sequence in different ways
        """
        seed = int(hash((self.seed, index)) % 1e6)
        seq = item["aa"]
        # {"id": index, 'aa': aa, 'pos': pos, 'ang': ang, 'conf': conf_score, "name": name}
        assert (
            "mask_idx" not in seq
        ), "Item already contains mask_idx key, this is not expected!"
        masked_seq, mask, replace_mask = masking_registry[self.seq_masking_method](
            item, self.args, seed, self.vocab.mask_idx, self.vocab.standard_toks
        )
        item["masked_aa"] = masked_seq
        item["mask"] = mask
        item["replace_mask"] = replace_mask

        """
        Add noise to the coordinate or angles, manupilate the item['pos']/item['ang']:
        - add noise to the coordinate
        - add noise to the angles
        """
        # keys in {"id", 'aa', 'pos', 'ang', 'conf', "name", "masked_aa", "mask", "replace_mask"}
        assert (
            "pos_noise" or "ang_noise" not in item
        ), "Item already contains mask_idx key, this is not expected!"
        pos_noise, ang_noise = noise_registry[self.noise_method](
            item, self.args, seed, self.pos_noise, self.ang_noise
        )
        item["pos_noise"] = pos_noise
        item["ang_noise"] = ang_noise

        return item

    def __len__(self) -> int:
        return len(self.names)

    def size(self, index: int) -> int:
        sz = self.sizes[index]
        if self.vocab.prepend_bos:
            sz += 1
        if self.vocab.append_eos:
            sz += 1
        raise sz

    def num_tokens(self, index: int) -> int:
        return self.sizes[index]

    def collater(self, samples: List[dict]) -> dict:
        return collate_fn(samples, self.vocab)


if __name__ == "__main__":

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Namespace()
    args.lmdb_path = (
        "/embedding/lihe/workspace/bio/pfm/data/sampled/downloads/pfm/48organism.lmdb"
    )

    print(args)
    print("=================")
    print("Test ProteinLMDBDataset")
    dataset = ProteinLMDBDataset(args)
    print(len(dataset))
    data = dataset[12]
    for k, v in data.items():
        print(k, v.shape if isinstance(v, np.ndarray) else v)
    # print(data)
