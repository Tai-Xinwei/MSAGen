# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union

import torch
from transformers import AutoTokenizer

from sfm.data.dataset import Batch, FoundationModelDataset
from sfm.data.dec_data.SFMDecTokenizer import SFMDecTokenizer
from sfm.logging import logger

ENTITY_MARKERS = [
    "[M]",
    "[/M]",
    "[P]",
    "[/P]",
    "[A]",
    "[/A]",
    "[T]",
    "[/T]",
    "[R]",
]


class TokenType(Enum):
    Text = 0  # Natural language text
    Entity = 1  # Any kind of scientific entity
    # SMILES = 2
    # FASTA = 3


@dataclass
class TextSpan:
    text: str
    type: TokenType


@dataclass
class MixedTokenData(Batch):
    """
    This represent a text mixed with entities (e.g., SMILES, FASTA, etc.).
    However, everying is in the form of text.
    """

    token_seq: torch.Tensor
    token_type_mask: torch.Tensor
    label_seq: torch.Tensor
    pad_idx: int

    @property
    def non_padding_mask(self):
        return self.token_seq.eq(self.pad_idx).logical_not()

    def to_tuple(self):
        t = (
            self.token_seq,
            self.token_type_mask,
            self.label_seq,
            torch.LongTensor([self.pad_idx]),
            torch.LongTensor([self.batch_size]),
        )
        return (t, t)

    @classmethod
    def from_tuple(cls, t):
        if len(t) == 2:
            t = t[0]

        return cls(
            token_seq=t[0],
            token_type_mask=t[1],
            label_seq=t[2],
            pad_idx=t[3].item(),
            batch_size=t[4].item(),
        )


class MixedTokenDataset(FoundationModelDataset):
    def __init__(
        self,
        sents: List[List[TextSpan]],
        text_tokenizer: str,
        entity_tokenizer: str,  # TODO: support multiple entity tokenizers
        max_text_len: int,
        max_entity_len: int,
        return_tuple: bool = False,
        pad_left: bool = False,
    ) -> None:
        super().__init__()
        self.sents = sents

        self.init_tokenziers(text_tokenizer, entity_tokenizer)

        self.max_text_len = max_text_len
        self.max_entity_len = max_entity_len
        self.pad_idx = self.text_tokenizer.pad_token_id
        self.return_tuple = return_tuple
        self.pad_left = pad_left

    def init_tokenziers(self, text_tokenizer: str, entity_tokenizer: str):
        self.entity_tokenizer = SFMDecTokenizer.from_pretrained(
            entity_tokenizer, use_fast=False
        )

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_tokenizer, use_fast=False
        )

        self.text_tokenizer.add_special_tokens(
            {
                "pad_token": "[PAD]",
            },
        )

        self.text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ENTITY_MARKERS}
        )

    def __len__(self):
        return len(self.sents)

    def entity_marker_from_entity_id_to_text_id(self, marker_id):
        token_str = self.entity_tokenizer.convert_ids_to_tokens([marker_id])[0].rstrip(
            "</w>"
        )

        assert (
            token_str[0] == "[" and token_str[-1] == "]"
        ), f"Invalid entity marker: {token_str}"

        return self.text_tokenizer.convert_tokens_to_ids([token_str])[0]

    def __getitem__(self, index) -> MixedTokenData:
        """
        The data will be in the following format like:
        raw text: "text [M] m<C> [/M] text"
        Input:  <bos> text | [M]  <m>C | [/M] text
        Label:  text  [M]  | <m>C [/M] | text <eos>
        Type:   text  text | ent  ent  | text  text

        Note:
        - The LLM is responsible for generating the first token of the molecule like [M].
            It can also geerate <eos> to terminate the generaton.
            Thus, we need to add those special tokens to LLM vocab.
        - The Entity decoder is responsible for generating the last token of the molecule like [/M]
        """

        if isinstance(index, slice):
            batch = []
            step = index.step if index.step is not None else 1
            for i in range(index.start, index.stop, step):
                batch.append(self[i])

            return self.collate(batch)

        sent = self.sents[index]

        span_token_ids = []
        for span in sent:
            if span.type == TokenType.Text:
                tokens = self.text_tokenizer(
                    span.text,
                    truncation=True,
                    max_length=self.max_text_len,
                    add_special_tokens=False,
                )["input_ids"]
            elif span.type == TokenType.Entity:
                tokens = self.entity_tokenizer(
                    span.text,
                    truncation=True,
                    max_length=self.max_entity_len,
                    add_special_tokens=False,
                )["input_ids"]
            else:
                raise ValueError(f"Unknown token type: {span.type}")

            span_token_ids.append(tokens)

        token_seq = []
        token_type_seq = []
        label_seq = []
        entity_end_token = None

        # token_seq always starts with <bos> and handled by LLM
        token_seq.append(self.text_tokenizer.bos_token_id)
        token_type_seq.append(TokenType.Text.value)

        for i in range(len(span_token_ids)):
            tokens = span_token_ids[i][:]
            labels = span_token_ids[i][:]
            span_type = sent[i].type

            if span_type == TokenType.Text:
                if entity_end_token is None:  # The first span is text
                    token_seq.extend(tokens)
                    label_seq.extend(labels)
                    token_type_seq.extend([TokenType.Text.value] * len(tokens))
                else:
                    token_seq.extend([entity_end_token] + tokens)
                    entity_end_token = None

                    label_seq.extend(labels)
                    token_type_seq.extend([TokenType.Text.value] * (len(tokens) + 1))

            elif span_type == TokenType.Entity:
                if entity_end_token is not None:
                    token_seq.append(entity_end_token)
                    token_type_seq.append(TokenType.Text.value)
                    entity_end_token = None

                entity_begin_token = self.entity_marker_from_entity_id_to_text_id(
                    tokens[0]
                )
                entity_end_token = self.entity_marker_from_entity_id_to_text_id(
                    tokens[-1]
                )
                token_seq.extend(tokens[:-1])

                label_seq.extend([entity_begin_token] + tokens[1:])
                token_type_seq.extend([TokenType.Entity.value] * (len(tokens) - 1))

        if entity_end_token is not None:
            token_seq.append(entity_end_token)
            token_type_seq.append(TokenType.Text.value)
            entity_end_token = None

        label_seq.append(self.text_tokenizer.eos_token_id)

        assert len(token_seq) == len(label_seq), f"{len(token_seq)} != {len(label_seq)}"
        assert len(token_seq) == len(
            token_type_seq
        ), f"{len(token_seq)} != {len(token_type_seq)}"

        token_seq = torch.IntTensor(token_seq)
        token_type_mask = torch.ByteTensor(token_type_seq)
        label_seq = torch.IntTensor(label_seq)

        data = MixedTokenData(
            token_seq=token_seq,
            token_type_mask=token_type_mask,
            label_seq=label_seq,
            pad_idx=self.pad_idx,
            batch_size=1,
        )

        if self.return_tuple:
            data = data.to_tuple()

        return data

    def pad_sequence(self, batch: List[torch.Tensor], pad_idx):
        if self.pad_left:
            batch = [t.flip(0) for t in batch]

        batch = torch.nn.utils.rnn.pad_sequence(
            batch,
            batch_first=True,
            padding_value=pad_idx,
        )

        if self.pad_left:
            batch = batch.flip(1)

        return batch

    def collate(self, batch: List[MixedTokenData]) -> MixedTokenData:
        """
        Collate a batch of MixedTokenData.
        """

        if self.return_tuple:
            batch = [MixedTokenData.from_tuple(t) for t in batch]

        # pad the token_seq
        batched_tokens = self.pad_sequence(
            [text.token_seq for text in batch],
            self.pad_idx,
        )

        batched_token_type_mask = self.pad_sequence(
            [text.token_type_mask for text in batch],
            torch.iinfo(batch[0].token_type_mask.dtype).max,
        )

        batched_label_seq = self.pad_sequence(
            [text.label_seq for text in batch],
            self.pad_idx,
        )

        batch = MixedTokenData(
            token_seq=batched_tokens,
            token_type_mask=batched_token_type_mask,
            label_seq=batched_label_seq,
            pad_idx=self.pad_idx,
            batch_size=len(batch),
        )

        if self.return_tuple:
            batch = batch.to_tuple()

        return batch

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
                        mol_line.replace("<m>", "")
                        .replace(" ", "")
                        .replace("<start-of-mol>", "[M]")
                        .replace("<end-of-mol>", "[/M]"),
                        TokenType.Entity,
                    ),
                ]
            )

        logger.info("Loaded {}/{} data", len(data), data_size)
        if show_example:
            logger.info("First example:\n {}", data[0])

        return cls(data, text_tokenizer, entity_tokenizer, max_text_len, max_entity_len)
