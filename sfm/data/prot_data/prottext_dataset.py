# -*- coding: utf-8 -*-
import logging

import torch

logging.getLogger().setLevel(logging.ERROR)

import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import lmdb
from torch.utils.data import Dataset

from sfm.data.prot_data.util import bstr2obj
from sfm.logging.loggers import logger

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class ProGPTCollator(object):
    def __init__(
        self, pad_token_id: int = 0, pp_mode: bool = True, protein_pad_id: int = 1
    ) -> None:
        self.pad_token_id = pad_token_id
        self.pp_mode = pp_mode
        self.protein_pad_id = protein_pad_id

    def pad_1d_unsqueeze(
        self, x: torch.Tensor, padlen: int, start: int, pad_num: Union[int, float]
    ):
        # insert 1 in the front of x as cls token and append 2 in the end as eos token
        x = torch.cat(
            [torch.tensor([0], dtype=x.dtype), x, torch.tensor([2], dtype=x.dtype)]
        )
        # (N) -> (1, padlen)
        xlen = x.size(0)
        assert (
            start + xlen <= padlen
        ), f"padlen {padlen} is too small for xlen {xlen} and start point {start}"
        new_x = x.new_full([padlen], pad_num, dtype=x.dtype)
        new_x[start : start + xlen] = x
        x = new_x

        return x.unsqueeze(0)

    def __call__(self, samples):
        input_ids = []
        labels = []
        proteins = []
        for sample in samples:
            input_ids.append(sample["input_ids"])
            labels.append(sample["labels"])
            proteins.extend(sample["proteins"])

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        max_tokens = max(len(p) for p in proteins) + 2  # add cls and eos token

        # convert proteins from list to torch tensor and pad it
        padded_proteins = torch.cat(
            [
                self.pad_1d_unsqueeze(
                    torch.tensor(p), max_tokens, 0, self.protein_pad_id
                )
                for p in proteins
            ]
        )

        if self.pp_mode:
            input_tuple = (
                input_ids,
                labels,
                input_ids.ne(self.pad_token_id),
                padded_proteins.long(),
            )
            label_tuple = (labels, input_ids.ne(self.pad_token_id))
            return (input_tuple, label_tuple)
        else:
            return dict(
                input_ids=input_ids,
                labels=labels,
                llm_mask=input_ids.ne(self.pad_token_id),
                proteins=padded_proteins.long(),
            )


class ProteinTextDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        model_max_length: int,
        protein_max_size: int,
        pad_token_id: int,
        max_pro_per_sample: int = 1,
        pool_mode: Optional[str] = "full",
        embedding_length: int = 20,
        num_token_id: int = 32003,
        protein_pad_id: int = 1,
        pp_mode: bool = True,
        local_rank: int = 0,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.model_max_length = model_max_length
        self.protein_max_size = protein_max_size
        self.pool_mode = pool_mode
        self.embedding_length = embedding_length
        self.num_token_id = num_token_id
        self.local_rank = local_rank

        self.max_pro_per_sample = max_pro_per_sample

        self.len = 0
        self.index_to_key_map = []
        self.read_txns = {}
        self.read_envs = {}
        self.dataset_count = {}
        self.dataset_filtered = {}

        self.env = lmdb.open(
            str(self.data_path), subdir=True, readonly=True, lock=False, readahead=False
        )
        self.txn = self.env.begin(write=False)
        metadata = bstr2obj(self.txn.get("metadata".encode()))
        self.len, self.keys = metadata["size"], metadata["keys"]

        self.collate_fn = ProGPTCollator(
            pad_token_id=pad_token_id, pp_mode=pp_mode, protein_pad_id=protein_pad_id
        )

        self.keys, self.len_filter = self.filter_dataset()

        logger.info(f"Dataset size: {self.len}, Filtered size: {self.len_filter}")
        self.len = self.len_filter

    def filter_dataset(self):
        filter_keys = []
        for key in self.keys:
            value = self.txn.get(str(key).encode())
            if value is None:
                raise IndexError(f"Name {key} has no data in the dataset")

            input_ids, proteins = pkl.loads(value)
            if len(proteins) > self.max_pro_per_sample:
                continue

            skip = 0
            sample_len = len(input_ids)
            for protein in proteins:
                if len(protein) > self.protein_max_size:
                    skip = 1
                    break

                sample_len += len(protein) - 1
                if sample_len > self.model_max_length:
                    skip = 1
                    break

            if not skip:
                filter_keys.append(key)

        return filter_keys, len(filter_keys)

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        key = self.keys[index]
        value = self.txn.get(str(key).encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")

        input_ids, proteins = pkl.loads(value)
        assert len(proteins) > 0, f"Protein list is empty for {key}"

        new_input_ids = []
        original_input_ids_len = len(input_ids)
        input_ids_len = len(input_ids)

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)

        if self.pool_mode == "full":
            mol_pos = torch.nonzero(input_ids < 0).squeeze(-1)
            mol_pos = torch.cat(
                [torch.tensor([0]), mol_pos, torch.tensor([len(input_ids)])]
            )

            for i in range(mol_pos.size(0) - 1):
                new_input_ids.extend(input_ids[mol_pos[i] : mol_pos[i + 1]])
                if i < len(mol_pos) - 2:
                    len_protein = len(proteins[i])
                    mol_idx = input_ids[mol_pos[i + 1]]
                    if len_protein > 1:
                        new_input_ids.extend(torch.ones([len_protein - 1]) * mol_idx)

                    if mol_pos[i + 1] < original_input_ids_len:
                        input_ids_len += len_protein - 1

        elif self.pool_mode == "qformer":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid pool_mode: {self.pool_mode}")

        input_ids = torch.tensor(new_input_ids).to(dtype=torch.int64)
        # input_ids = input_ids[: self.model_max_length]

        labels = input_ids.clone()
        # labels[:input_ids_len] = IGNORE_INDEX
        labels[labels < 0] = IGNORE_INDEX

        return dict(
            input_ids=input_ids,
            labels=labels,
            proteins=proteins,
        )

    def collate(self, samples):
        return self.collate_fn(samples)


if __name__ == "__main__":
    pass
