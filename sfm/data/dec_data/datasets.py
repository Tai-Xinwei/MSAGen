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
    token_seq_len: Union[int, torch.Tensor]
    token_type_mask: torch.Tensor
    label_seq: torch.Tensor
    pad_idx: int

    @property
    def non_padding_mask(self):
        return self.token_seq.eq(self.pad_idx).logical_not()


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
        self.pad_idx = self.text_tokenizer.pad_token_id

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
            {
                "additional_special_tokens": [
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
            }
        )

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index) -> MixedTokenData:
        """
        The data will be in the following format like:
        Input:  <bos> text | [M]  <m>C | [/M] text
        Output: text  [M]  | <m>C [/M] | text <eos>
        Type:   text  text | ent  ent  | text  text

        Note:
        - We assume that the first span is always text. i.e, we can't generate a molecule without text.
        - The LLM is responsible for generating the first token of the molecule like [M].
            It can also geerate <eos> to terminate the generaton.
            Thus, we need to add those special tokens to LLM vocab.
        - The Entity decoder is responsible for generating the last token of the molecule like [/M]
        """

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
        eos_added = False
        for i in range(len(span_token_ids)):
            tokens = span_token_ids[i][:]
            labels = span_token_ids[i][:]
            span_type = sent[i].type

            if span_type == TokenType.Text:
                if i == 0:
                    tokens = [self.text_tokenizer.bos_token_id] + tokens
                else:
                    assert entity_end_token is not None
                    tokens = [entity_end_token] + tokens
                    entity_end_token = None

                if i == len(sent) - 1:
                    labels = labels + [self.text_tokenizer.eos_token_id]
                    eos_added = True
                else:
                    next_span_start_str = self.entity_tokenizer.convert_ids_to_tokens(
                        span_token_ids[i + 1][:1]
                    )[0].rstrip("</w>")
                    labels = labels + self.text_tokenizer.convert_tokens_to_ids(
                        [next_span_start_str]
                    )
            elif span_type == TokenType.Entity:
                entity_end_token = tokens[-1]
                tokens = tokens[:-1]  # remove [/X] in the end
                labels = labels[1:]  # remove [X] in the beginning

            assert len(tokens) == len(labels)
            token_seq.extend(tokens)
            token_type_seq.extend([span_type.value] * len(tokens))
            label_seq.extend(labels)

        if not eos_added:
            # This means the last span is an entity
            assert entity_end_token is not None
            token_seq.append(entity_end_token)
            token_type_seq.append(TokenType.Text.value)
            label_seq.append(self.text_tokenizer.eos_token_id)

        token_seq = torch.IntTensor(token_seq)
        token_seq_len = token_seq.shape[0]
        token_type_mask = torch.ShortTensor(token_type_seq)
        label_seq = torch.IntTensor(label_seq)

        return MixedTokenData(
            token_seq=token_seq,
            token_seq_len=token_seq_len,
            token_type_mask=token_type_mask,
            label_seq=label_seq,
            pad_idx=self.pad_idx,
            batch_size=1,
        )

    def collate(self, batch: List[MixedTokenData]) -> MixedTokenData:
        """
        Collate a batch of MixedTokenData.
        """

        # pad the token_seq
        batched_tokens = torch.nn.utils.rnn.pad_sequence(
            [text.token_seq for text in batch],
            batch_first=True,
            padding_value=self.pad_idx,
        )

        token_seq_len = torch.tensor(
            [text.token_seq_len for text in batch], dtype=torch.int64
        )

        batched_token_type_mask = torch.nn.utils.rnn.pad_sequence(
            [text.token_type_mask for text in batch],
            batch_first=True,
            padding_value=self.pad_idx,
        )

        batched_label_seq = torch.nn.utils.rnn.pad_sequence(
            [text.label_seq for text in batch],
            batch_first=True,
            padding_value=self.pad_idx,
        )

        return MixedTokenData(
            token_seq=batched_tokens,
            token_seq_len=token_seq_len,
            token_type_mask=batched_token_type_mask,
            label_seq=batched_label_seq,
            pad_idx=self.pad_idx,
            batch_size=len(batch),
        )

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
