# -*- coding: utf-8 -*-
import copy
import itertools
from functools import lru_cache
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# import pytorch_forecasting
import torch
from ogb.graphproppred import Evaluator
from ogb.lsc import PCQM4Mv2Evaluator
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from sfm.utils.chemical_tokens import CHEMICAL_TOKENS
from sfm.utils.move_to_device import move_to_device

from .collator import collator
from .moltext_dataset import smiles2graph_removeh
from .wrapper import preprocess_item, smiles2graph

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class MolTokenizer:
    def __init__(self, args):
        self.args = args

        text_tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model_name_or_path,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        special_tokens_dict = dict()
        if text_tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if text_tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if text_tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if text_tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        special_tokens_dict["additional_special_tokens"] = CHEMICAL_TOKENS
        text_tokenizer.add_special_tokens(special_tokens_dict)

        self.text_tokenizer = text_tokenizer

    def split_text_and_mol(self, list_data: List[str]) -> Tuple[List, List, List]:
        """Split data into text and mol."""
        text_list = []
        mol_list = []
        smile_idx_list = []
        for data in list_data:
            if data.find("<mol>") == -1:
                text_list.append([data])
                # mol_list.append("")
            else:
                split_text = []
                pos1 = []
                pos2 = []

                pos = 0
                while data.find("<mol>", pos) != -1:
                    pos1.append(data.find("<mol>", pos))
                    pos2.append(data.find("</mol>", pos))

                    pos = pos2[-1] + 1

                if len(pos1) > 0:
                    split_text.append(data[: pos1[0] + 5])
                    # split_text.append(data[: pos1[0]])

                    for i in range(len(pos1)):
                        smile = data[pos1[i] : pos2[i]]
                        mol_list.append(smile.replace("<mol>", "").replace(" ", ""))
                        if i < len(pos1) - 1:
                            split_text.append(data[pos2[i] : pos1[i + 1] + 5])

                    split_text.append(data[pos2[-1] :])
                    # split_text.append(data[pos2[-1]+6 :])

                text_list.append(split_text)
            # smiles_idx = -smiles_dict.get(smiles[idx], 1) - 1
            # smile_idx_list.append(smiles_idx)
        return text_list, mol_list, smile_idx_list

    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text."""
        input_ids = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        return input_ids

    def _process_text(
        self, text_list: List[str], batched_smile_data
    ) -> List[List[int]]:
        """Tokenize text."""
        if batched_smile_data is None:
            nnodes = [0] * len(text_list)
        else:
            nnodes = batched_smile_data["nnodes"]

        input_ids_list = []
        for i, text in enumerate(text_list):
            if len(text) == 1:
                input_ids = self._tokenize_text(text).input_ids[0]
            elif self.args.pool_mode == "qformer":
                input_ids = []
                for j, t in enumerate(text):
                    if j == 0:
                        input_ids.append(self._tokenize_text(t).input_ids[0])
                        input_ids.append(
                            torch.tensor(
                                [-j - 1 for i in range(self.args.embedding_length)]
                            ).to(torch.long)
                        )
                    elif j < len(text) - 1:
                        input_ids.append(self._tokenize_text(t).input_ids[0][1:])
                        input_ids.append(
                            torch.tensor(
                                [-j - 1 for i in range(self.args.embedding_length)]
                            ).to(torch.long)
                        )
                    else:
                        input_ids.append(self._tokenize_text(t).input_ids[0][1:])

                input_ids = torch.cat(input_ids)

            elif self.args.pool_mode == "full":
                input_ids = torch.cat(
                    [
                        self._tokenize_text(text[0]).input_ids[0],
                        torch.tensor([-1 for i in range(nnodes[i])]).to(torch.long),
                        self._tokenize_text(text[1]).input_ids[0][1:],
                    ]
                )
            input_ids_list.append(input_ids)

        return input_ids_list

    def _process_smile(self, smile_list: List[str]):
        """Process smile."""
        smile_items = []
        for i, smile in enumerate(smile_list):
            graph = smiles2graph_removeh(smile)
            data = Data()
            data.idx = i
            data.__num_nodes__ = torch.tensor([int(graph["num_nodes"])]).to(torch.int64)
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["x"]).to(torch.int64)
            data.y = torch.tensor([0]).to(torch.float32)

            smile_items.append(preprocess_item(data))

        batched_smile_data = collator(smile_items)
        return batched_smile_data

    def _process_input_ids(
        self, input_ids_list: List[List[int]], pad_token_id: int = 32000
    ):
        max_seq_len = max(len(i) for i in input_ids_list)
        input_ids = torch.cat(
            [
                torch.cat(
                    [
                        i,
                        torch.ones(max_seq_len - len(i), dtype=i.dtype).fill_(
                            pad_token_id
                        ),
                    ]
                ).unsqueeze(0)
                for i in input_ids_list
            ]
        )
        llm_mask = torch.cat([i.ne(pad_token_id).unsqueeze(0) for i in input_ids_list])
        return input_ids, llm_mask

    def tokenize(self, input_data: List[str]):
        text_list, smile_list, _ = self.split_text_and_mol(input_data)
        if len(smile_list) > 0:
            batched_smile_data = self._process_smile(smile_list)
        else:
            batched_smile_data = None

        input_ids_list = self._process_text(text_list, batched_smile_data)
        input_ids_list, llm_mask = self._process_input_ids(input_ids_list)
        return input_ids_list, batched_smile_data, llm_mask


if __name__ == "__main__":
    pass
