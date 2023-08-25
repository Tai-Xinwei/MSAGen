# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union

import torch
from transformers import AutoTokenizer

from sfm.data.dataset import Batch, FoundationModelDataset
from sfm.data.tamgent2.tokenizer import MolxptTokenizer
from sfm.logging import logger


class TokenType(Enum):
    Text = 0  # Natural language text
    Entity = 1  # Any kind of scientific entity
    # SMILES = 2
    # FASTA = 3


@dataclass
class TextSpan:
    text: str
    type: TokenType


class MixedTokenData(Batch):
    """
    This represent a text mixed with entities (e.g., SMILES, FASTA, etc.).
    However, everying is in the form of text.
    """

    def __init__(
        self,
        token_seq: torch.Tensor,
        token_seq_len: Union[int, torch.Tensor],
        token_type_mask: torch.Tensor,
    ) -> None:
        super().__init__(batch_size=MixedTokenData.compute_batch_size(token_seq))

        self.token_seq = token_seq
        self.token_seq_len = token_seq_len
        self.token_type_mask = token_type_mask

    @staticmethod
    def compute_batch_size(token_seq: torch.Tensor) -> int:
        if len(token_seq.shape) == 1:
            return 1
        else:
            return token_seq.shape[0]


class MixedTokenDataset(FoundationModelDataset):
    def __init__(
        self,
        sents: List[List[TextSpan]],
        text_tokenizer: str,
        entity_tokenizer: str,  # TODO: support multiple entity tokenizers
        max_text_len: int,
        max_entity_len: int,
    ) -> None:
        super().__init__()
        self.sents = sents

        self.init_tokenziers(text_tokenizer, entity_tokenizer)

        self.max_text_len = max_text_len
        self.max_entity_len = max_entity_len
        self.pad_idx = 0  # TODO: use tokenizer.pad_token_id

    def init_tokenziers(self, text_tokenizer: str, entity_tokenizer: str):
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_tokenizer, use_fast=False
        )
        self.entity_tokenizer = MolxptTokenizer.from_pretrained(
            entity_tokenizer, use_fast=False
        )

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index) -> MixedTokenData:
        sent = self.sents[index]

        token_seq = []
        token_type_seq = []
        for span in sent:
            if span.type == TokenType.Text:
                tokens = self.text_tokenizer(
                    span.text,
                    return_tensors="pt",
                    padding="do_not_pad",
                    truncation=True,
                    max_length=self.max_text_len,
                )
            elif span.type == TokenType.Entity:
                tokens = self.entity_tokenizer(
                    span.text,
                    return_tensors="pt",
                    padding="do_not_pad",
                    truncation=True,
                    max_length=self.max_entity_len,
                )

            else:
                raise ValueError(f"Unknown token type: {span.type}")

            token_seq.append(tokens["input_ids"])
            token_type_seq.append(
                torch.ones_like(tokens["input_ids"], dtype=torch.int8) * span.type.value
            )

        token_seq = torch.cat(token_seq, dim=1)
        token_seq_len = token_seq.shape[0]
        token_type_mask = torch.cat(token_type_seq, dim=1)

        return MixedTokenData(token_seq, token_seq_len, token_type_mask)

    def collate(self, batch: List[MixedTokenData]) -> MixedTokenData:
        """
        Collate a batch of MixedTokenData.
        """

        # pad the token_seq
        batched_tokens = torch.nn.utils.rnn.pad_sequence(
            [text.token_seq[0] for text in batch],
            batch_first=True,
            padding_value=self.pad_idx,
        )

        token_seq_len = torch.tensor(
            [text.token_seq_len for text in batch], dtype=torch.int64
        )

        batched_token_type_mask = torch.nn.utils.rnn.pad_sequence(
            [text.token_type_mask[0] for text in batch],
            batch_first=True,
            padding_value=self.pad_idx,
        )

        return MixedTokenData(batched_tokens, token_seq_len, batched_token_type_mask)

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        text_tokenizer: str,
        entity_tokenizer: str,
        max_text_len: int,
        max_entity_len: int,
    ):
        """
        Read a jsonl file, and return a MixedTokenDataset.
        """
        data = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                spans = json.loads(line)
                data.append([TextSpan(**span) for span in spans])
        return cls(data, text_tokenizer, entity_tokenizer, max_text_len, max_entity_len)

    @classmethod
    def from_text_to_mol(
        cls,
        mol_path: str,
        text_path,
        text_tokenizer: str,
        entity_tokenizer: str,
        max_text_len: int,
        max_entity_len: int,
        show_example: bool = False,
    ):
        """
        A special case that read from "text to mol" dataset.
        """

        with open(mol_path, "r") as f:
            mol_lines = f.read().splitlines()

        with open(text_path, "r") as f:
            text_lines = f.read().splitlines()

        assert len(mol_lines) == len(text_lines)
        data_size = len(mol_lines)

        data = []
        logger.info(
            "Loading data from {} and {}. Data size {}", mol_path, text_path, data_size
        )
        for mol_line, text_line in zip(mol_lines, text_lines):
            data.append(
                [
                    TextSpan(text_line, TokenType.Text),
                    TextSpan(
                        mol_line.replace("<m>", "").replace(" ", ""), TokenType.Entity
                    ),
                ]
            )

        logger.info("Loaded {}/{} data", len(data), data_size)
        if show_example:
            logger.info("First example:\n {}", data[0])

        return cls(data, text_tokenizer, entity_tokenizer, max_text_len, max_entity_len)
