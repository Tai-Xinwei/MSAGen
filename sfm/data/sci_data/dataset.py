# -*- coding: utf-8 -*-
from collections import namedtuple

import numpy as np
import torch

from sfm.data.prot_data.dataset import DownstreamLMDBDataset
from sfm.data.sci_data import SFMDecTokenizer
from sfm.logging import logger

# we have to use named tuple to avoid being convreted to list by pytorch,
# but also compatible with DeepSpeed PP
SciTokenIdxAndMask = namedtuple("SciTokenIdxAndMask", ["input_ids", "padding_mask"])
SciDataTuple = namedtuple("SciDataTuple", ["input", "labels"])


def shuffle_sub_sequences(seq, eos_idx):
    """
    For a science_token sequence like [a, a, eos, b, eos, c]
    shuffle the sub sequences to [b, eos, a, a, eos, c]
    Note the last sequence is not shuffled if it's not complete.
    """
    only_contains_sci_tokens = np.all(seq <= 32000)
    if not only_contains_sci_tokens:
        # don't shuffle text or mixed sequences
        return seq

    indices = np.where(seq == eos_idx)[0] + 1
    split_arrays = np.split(seq, indices)
    last_seq_complete = seq[-1] == eos_idx
    if last_seq_complete:
        np.random.shuffle(split_arrays)
    else:
        to_shuffle, end = split_arrays[:-1], split_arrays[-1]
        np.random.shuffle(to_shuffle)
        split_arrays = to_shuffle + [end]
    return np.concatenate(split_arrays)


class ProcessedSciDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        padding_idx,
        max_len: int,
        eos_idx: int = -1,
        shuffle_subseq: bool = False,
    ):
        super().__init__()
        self.data = np.load(path, mmap_mode="r")
        processed_seq_len = self.data.shape[1]
        if processed_seq_len % max_len != 0:
            raise ValueError(
                f"processed_seq_len {processed_seq_len} is not divisible by max_len {max_len}"
            )
        self.replicate = processed_seq_len // max_len
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.eos_idx = eos_idx
        self.shuffle_subseq = shuffle_subseq

        if self.shuffle_subseq:
            assert self.eos_idx != -1, "eos_idx must be set if shuffle_subseq is True"

        logger.info(
            f"Loaded {path} with shape {self.data.shape}, max_len {max_len}, replicate {self.replicate}"
        )

    def __getitem__(self, index):
        index, offset = divmod(index, self.replicate)
        data = self.data[index][offset * self.max_len : (offset + 1) * self.max_len]
        if self.shuffle_subseq:
            data = shuffle_sub_sequences(data, self.eos_idx)
        return torch.from_numpy(data.astype(np.int64))

    def __len__(self):
        return self.data.shape[0] * self.replicate

    def collate(self, samples):
        input_ids = torch.stack(samples, dim=0)
        padding_mask = input_ids.ne(self.padding_idx)
        input = SciTokenIdxAndMask(input_ids, padding_mask)
        labels = input
        return SciDataTuple(input, labels)


class RawTextSciDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: SFMDecTokenizer,
        conditional_generation: bool = False,
        use_template: bool = False,
        max_len: int = 1024,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        with open(path, "r") as f:
            for line in f:
                self.data.append(line.strip())

        logger.info(f"Loaded {path} with {len(self.data)} lines")
        self.conditional_generation = conditional_generation
        self.use_template = use_template
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.data[index]
        if self.conditional_generation:
            prompt, target = text.split("\t")

            if self.use_template:
                prompt = f"Instruction: {prompt}\n\n\nResponse:"

            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
            tokens = (
                [self.tokenizer.bos_token_id]
                + prompt_tokens
                + target_tokens
                + [self.tokenizer.eos_token_id]
            )
            labels = tokens[:]
            labels[: len(prompt_tokens) + 1] = [-100] * (len(prompt_tokens) + 1)
        else:
            tokens = (
                [self.tokenizer.bos_token_id]
                + self.tokenizer.encode(text, add_special_tokens=False)
                + [self.tokenizer.eos_token_id]
            )
            labels = tokens[:]

        # keep the last max_len tokens, or there maybe no labels to pred
        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]

        return torch.tensor(tokens), torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def collate(self, samples):
        input_ids_list, labels_list = zip(*samples)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        padding_mask = input_ids.ne(self.tokenizer.pad_token_id)

        input = tuple([input_ids, padding_mask])
        return (input, labels)


class ProteinLmdbDataset(torch.utils.data.Dataset):
    def __init__(self, task_name, lmdb_dataset, lmdb_vocab, tokenizer):
        self.task_name = task_name
        self.task_type = DownstreamLMDBDataset.TASKINFO[task_name]["type"]
        self.lmdb_dataset = lmdb_dataset
        self.lmdb_vocab = lmdb_vocab
        self.tokenizer = tokenizer
        self.multi_seq_tasks = ["yeast_ppi", "human_ppi", "ppi_affinity"]
        self.idx_to_tok = {v: k for k, v in lmdb_vocab.tok_to_idx.items()}

        logger.info(f"Loaded {task_name} with {len(self)} lines, type {self.task_type}")

    def reverse2str(self, tokens):
        vocab = self.lmdb_vocab
        aa_seq = []
        for i in tokens:
            if i in [
                vocab.unk_idx,
                vocab.padding_idx,
                vocab.cls_idx,
                vocab.mask_idx,
                vocab.eos_idx,
            ]:
                continue
            aa_seq.append(self.idx_to_tok[i])
        return "".join(aa_seq)

    def __len__(self):
        return len(self.lmdb_dataset)

    def __getitem__(self, index):
        item = self.lmdb_dataset[index]

        sentence = ""

        if self.task_name in self.multi_seq_tasks:
            aa_seqs = [self.reverse2str(item[f"aa_{i}"]) for i in range(2)]
            sentence = (
                "<protein>"
                + aa_seqs[0]
                + "</protein><protein>"
                + aa_seqs[1]
                + "</protein>"
            )
        else:
            aa_seq = self.reverse2str(item["aa"])
            sentence = "<protein>" + aa_seq + "</protein>"

        target = item["target"]
        if self.task_type == "regression":
            assert target.shape == (1,)
            target = f"{target[0]:.8f}"
        elif self.task_type == "binary":
            assert target.shape == (1,)
            target = "true" if target[0] == 1 else "false"
        elif self.task_type == "classification":
            assert len(target) == 1
            target = str(target[0])
        else:  # multi_classification
            target = " , ".join([str(v) for v in item["target"]])

        sentence = sentence + " " + target
        # tokens = self.tokenizer.tokenize(sentence, add_special_tokens=False)
        tokens = self.tokenizer.encode(sentence, add_special_tokens=False)

        input_ids = (
            [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        )

        prot_end_token_idx = self.tokenizer.convert_tokens_to_ids("</protein>")
        last_prot_end = -1
        for i, token in enumerate(tokens):
            if token == prot_end_token_idx:
                last_prot_end = i

        labels = input_ids[:]
        labels[: last_prot_end + 2] = [-100] * (last_prot_end + 2)

        return torch.tensor(input_ids), torch.tensor(labels)

    def collate(self, samples):
        input_ids_list, labels_list = zip(*samples)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        padding_mask = input_ids.ne(self.tokenizer.pad_token_id)

        input = tuple([input_ids, padding_mask])
        return (input, labels)
