# -*- coding: utf-8 -*-
import bisect
import os
import pickle as pkl
from collections import namedtuple
from typing import Optional

import lmdb
import numpy as np
import torch
import torch.nn.functional as F

from sfm.data.prot_data.dataset import DownstreamLMDBDataset
from sfm.data.prot_data.util import bstr2obj
from sfm.data.sci_data import SFMDecTokenizer
from sfm.logging import logger
from sfm.pipeline.accelerator.dataclasses import TrainStrategy

# we have to use named tuple to avoid being convreted to list by pytorch,
# but also compatible with DeepSpeed PP
SciTokenIdxAndMask = namedtuple("SciTokenIdxAndMask", ["input_ids", "padding_mask"])
SciDataTuple = namedtuple("SciDataTuple", ["input", "labels"])


def shuffle_sub_sequences(seq, eos_idx, sci_idx_cutoff=32000):
    """
    For a science_token sequence like [a, a, eos, b, eos, c]
    shuffle the sub sequences to [b, eos, a, a, eos, c]
    Note the last sequence is not shuffled if it's not complete.
    """
    only_contains_sci_tokens = np.all(seq >= sci_idx_cutoff)
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
        data_list = []
        file_list = []
        if isinstance(path, str):
            if os.path.isfile(path):
                file_list.append(path)
            else:
                for file_name in os.listdir(path):
                    if file_name.endswith(".npy"):
                        file_list.append(os.path.join(path, file_name))
        elif isinstance(path, list):
            file_list = path
        else:
            raise ValueError(f"{path} error")
        logger.info(f"The dataset contains {len(file_list)} files.")
        for file_path in file_list:
            data = np.load(file_path, mmap_mode="r")
            logger.info(f"Load {file_path} into the dataset.")
            data_list.append(data)
        self.data_list = data_list

        processed_seq_len = self.data_list[0].shape[1]
        for i in range(len(data_list)):
            if processed_seq_len != self.data_list[i].shape[1]:
                raise ValueError(
                    f"{os.listdir(path)[i]} ({self.data_list[i].shape[1]}) is inconsistent with processed_seq_len in other files ({processed_seq_len})"
                )

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
        self.acc_data_size_list = []
        for i in range(len(self.data_list)):
            cur_num = self.data_list[i].shape[0] * self.replicate
            self.acc_data_size_list.append(self.acc_data_size_list[i] + cur_num)

        logger.info(
            f"Loaded {path} with shape ({int(self.__len__()/self.replicate), processed_seq_len}), max_len {max_len}, replicate {self.replicate}"
        )

    def __getitem__(self, index):
        list_index, data_index = self.get_real_index(index)
        data_index, offset = divmod(data_index, self.replicate)

        data = self.data_list[list_index][data_index][
            offset * self.max_len : (offset + 1) * self.max_len
        ]
        if self.shuffle_subseq:
            data = shuffle_sub_sequences(data, self.eos_idx)
        return torch.from_numpy(data.astype(np.int64))

    def get_real_index(self, index):
        list_index = bisect.bisect_right([0] + self.acc_data_size_list, index)
        return list_index - 1, index - self.acc_data_size_list[list_index - 1]

    def __len__(self):
        return self.acc_data_size_list[-1]

    def collate(self, samples):
        input_ids = torch.stack(samples, dim=0)
        padding_mask = input_ids.ne(self.padding_idx)
        input = SciTokenIdxAndMask(input_ids, padding_mask)
        labels = input
        return SciDataTuple(input, labels)


class ProcessedSciDatasetLmdb(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str | list,
        padding_idx: int,
        max_len: int,
        data_dir: Optional[str] = None,
        eos_idx: int = -1,
        shuffle_subseq: bool = False,
    ):
        super().__init__()
        env_list = []
        txn_list = []
        keys_list = []
        data_size_list = []
        acc_data_size_list = []
        processed_seq_len = None
        self.len = 0
        file_list = []
        self.dtype = None
        if isinstance(path, str):
            if path.find(",") != -1:
                logger.info(f"Multiple files in {path}")
                for pth in path.split(","):
                    if data_dir:
                        if os.path.isfile(os.path.join(data_dir, pth, "data.mdb")):
                            file_list.append(os.path.join(data_dir, pth))
                        else:
                            logger.warning(
                                f"File {os.path.join(data_dir, pth)} not found"
                            )
                    else:
                        file_list.append(pth)
            elif path.endswith(".lmdb") or path.endswith(".lmdb/"):
                logger.info(f"Single file {path}")
                if data_dir:
                    file_list.append(os.path.join(data_dir, path))
                else:
                    file_list.append(path)
            else:
                logger.info(f"Directory {path}")
                for file_name in os.listdir(path):
                    if file_name.endswith(".lmdb") or file_name.endswith(".lmdb/"):
                        file_list.append(os.path.join(path, file_name))
        elif isinstance(path, list):
            if data_dir is not None:
                for pth in path:
                    if os.path.isfile(os.path.join(data_dir, pth)):
                        file_list.append(os.path.join(data_dir, pth))
            else:
                file_list = path
        else:
            raise ValueError(f"{path} error")

        logger.info(f"The dataset {file_list} contains {len(file_list)} files.")

        for file_path in file_list:
            env = lmdb.open(
                file_path, subdir=True, readonly=True, lock=False, readahead=False
            )
            logger.info(f"Load {file_path} into the dataset.")
            env_list.append(env)
            txn = env.begin(write=False)
            txn_list.append(txn)
            metadata = bstr2obj(txn.get("metadata".encode()))
            cur_processed_seq_len = metadata["processed_seq_len"]
            cur_dtype = metadata["dtype"]
            if processed_seq_len is not None:
                if cur_processed_seq_len != processed_seq_len:
                    raise ValueError(
                        f"{file_path} ({cur_processed_seq_len}) is inconsistent with processed_seq_len in other files ({processed_seq_len})"
                    )
            else:
                processed_seq_len = cur_processed_seq_len
            if self.dtype is not None:
                if self.dtype != cur_dtype:
                    raise ValueError(
                        f"{file_path} ({cur_dtype}) is inconsistent with dtype in other files ({self.dtype})"
                    )
            else:
                self.dtype = cur_dtype

        if self.dtype == "uint16":
            self.dtype = np.uint16
        elif self.dtype == "uint32":
            self.dtype = np.uint32
        elif self.dtype == "uint64":
            self.dtype = np.uint64
        else:
            raise ValueError(f"current dtype {self.dtype} error")

        self.env_list = env_list
        self.txn_list = txn_list
        if processed_seq_len % max_len != 0:
            raise ValueError(
                f"processed_seq_len {processed_seq_len} is not divisible by max_len {max_len}"
            )

        self.replicate = processed_seq_len // max_len
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.eos_idx = eos_idx
        self.shuffle_subseq = shuffle_subseq
        for txn in self.txn_list:
            metadata = bstr2obj(txn.get("metadata".encode()))
            cur_len, cur_keys = metadata["size"], metadata["keys"]
            self.len += cur_len * self.replicate

            data_size_list.append(cur_len * self.replicate)
            acc_data_size_list.append(self.len)

            keys_list.append(cur_keys)
            cur_processed_seq_len = metadata["processed_seq_len"]
            if processed_seq_len is not None:
                if cur_processed_seq_len != processed_seq_len:
                    raise ValueError(
                        f"{os.path.join(path,file_name)} ({cur_processed_seq_len}) is inconsistent with processed_seq_len in other files ({processed_seq_len})"
                    )
            else:
                processed_seq_len = cur_processed_seq_len
        self.data_size_list = data_size_list
        self.acc_data_size_list = acc_data_size_list
        self.keys_list = keys_list
        if self.shuffle_subseq:
            assert self.eos_idx != -1, "eos_idx must be set if shuffle_subseq is True"

        logger.info(
            f"Loaded {path} with shape ({int(self.len/self.replicate), processed_seq_len}), max_len {max_len}, replicate {self.replicate}"
        )

    def __getitem__(self, index):
        list_index, data_index = self.get_real_index(index)
        data_index, offset = divmod(data_index, self.replicate)
        key = self.keys_list[list_index][data_index]
        value = self.txn_list[list_index].get(str(key).encode())
        data = np.frombuffer(value, dtype=self.dtype)[
            offset * self.max_len : (offset + 1) * self.max_len
        ]
        if self.shuffle_subseq:
            data = shuffle_sub_sequences(data, self.eos_idx)
        return torch.from_numpy(data.astype(np.int64))

    def get_real_index(self, index):
        list_index = bisect.bisect_right([0] + self.acc_data_size_list, index)
        return list_index - 1, index - self.acc_data_size_list[list_index - 1]

    def __len__(self):
        return self.len

    def collate(self, samples):
        input_ids = torch.stack(samples, dim=0)
        padding_mask = input_ids.ne(self.padding_idx)
        input = SciTokenIdxAndMask(input_ids, padding_mask)
        labels = input
        return SciDataTuple(input, labels)


class ProcessedSciWeightedDatasetLmdb(ProcessedSciDatasetLmdb):
    def __init__(
        self,
        args,
        data_dir: str,
        path: str,
        padding_idx: int,
        max_len: int,
        data_raito: float = None,
        eos_idx: int = -1,
        shuffle_subseq: bool = False,
    ):
        super(ProcessedSciWeightedDatasetLmdb, self).__init__(
            path, padding_idx, max_len, data_dir, eos_idx, shuffle_subseq
        )

        self.data_raio = [float(r) for r in data_raito.split(",")]
        ratio_sum = sum(self.data_raio)
        self.data_raio = [r / ratio_sum for r in self.data_raio]
        assert len(self.data_raio) == len(
            self.acc_data_size_list
        ), f"data_raito must be equal to the number of files, but got {len(self.data_raio)} and {len(self.acc_data_size_list)}"

        logger.info(f"data_raio {self.data_raio}")
        logger.info(f"data_size_list {self.data_size_list}")
        logger.info(f"acc_data_size_list {self.acc_data_size_list}")

        self.args = args
        self.__init_seed(args)

    def __init_seed(self, args):
        if args.strategy == TrainStrategy.ThreeD:
            from deepspeed.runtime.pipe.topology import (
                PipelineParallelGrid,
                PipeModelDataParallelTopology,
            )

            from megatron.core import mpu

            topology = PipeModelDataParallelTopology(
                num_pp=mpu.get_pipeline_model_parallel_world_size(),
                num_mp=mpu.get_tensor_model_parallel_world_size(),
                num_dp=mpu.get_data_parallel_world_size(),
            )
            self.dp_rank = PipelineParallelGrid(
                topology=topology
            ).get_data_parallel_rank()
            logger.warning(f"global_rank {args.rank}, dp_rank {self.dp_rank}")
        else:
            self.dp_rank = args.rank // args.pipeline_model_parallel_size

        self.global_index = 0

    def __getitem__(self, index):
        # get data from the corresponding dataset with the probability of data_raio
        np.random.seed(self.dp_rank + self.global_index)
        self.global_index += 1
        if self.global_index >= 2147480000:
            self.global_index = 0

        list_index = np.random.choice(len(self.data_raio), p=self.data_raio)
        data_index = np.random.randint(0, self.data_size_list[list_index])

        data_index, offset = divmod(data_index, self.replicate)
        key = self.keys_list[list_index][data_index]
        value = self.txn_list[list_index].get(str(key).encode())
        data = np.frombuffer(value, dtype=self.dtype)[
            offset * self.max_len : (offset + 1) * self.max_len
        ]
        if self.shuffle_subseq:
            data = shuffle_sub_sequences(data, self.eos_idx)
        return torch.from_numpy(data.astype(np.int64))


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


class RawTextSciDatasetwithAltTokenizer(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: SFMDecTokenizer,
        tokenize,
        conditional_generation: bool = False,
        use_template: bool = False,
        max_len: int = 1024,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenize = tokenize
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

            prompt_tokens = self.tokenize(prompt)[1:-1]
            target_tokens = self.tokenize(target)[1:-1]
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
                + self.tokenize(text)
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


class LMDBInstDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        padding_idx: int,
        max_len: int = 8192,
    ):
        super().__init__()
        env = lmdb.open(
            str(path), subdir=True, readonly=True, lock=False, readahead=False
        )
        self.txn = env.begin(write=False)
        metadata = bstr2obj(self.txn.get("metadata".encode()))
        self.size, self.keys = metadata["size"], metadata["keys"]

        logger.info(f"Loaded {path} with {self.size} lines")
        self.max_len = max_len
        self.padding_idx = padding_idx

    def __getitem__(self, index):
        value = self.txn.get(str(self.keys[index]).encode())
        tokens, labels = pkl.loads(value)

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        return torch.tensor(tokens), torch.tensor(labels)

    def __len__(self):
        return self.size

    def collate(self, samples):
        input_ids_list, labels_list = zip(*samples)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.padding_idx
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )

        padding_mask = input_ids.ne(self.padding_idx)

        input = tuple([input_ids, padding_mask])
        return (input, labels)
