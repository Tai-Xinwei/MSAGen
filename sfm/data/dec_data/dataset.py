# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union

import torch
from transformers import AutoTokenizer

from sfm.data.dataset import Batch, FoundationModelDataset
from sfm.data.tamgent2.tokenizer import MolxptTokenizer


class TokenType(Enum):
    Text = 0  # Natural language text
    Entity = 1  # Any kind of scientific entity
    # SMILES = 2
    # FASTA = 3


@dataclass
class TokenIdRange:
    """
    For entities, we use special tokens to represent them, e.g., <m>C rather than C.
    Thus, for each type of entity, we assign [strat, end) token ids to it.
    """

    start: int
    end: int


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
        entity_id_rage: Dict[TokenType, TokenIdRange],
    ) -> None:
        super().__init__(batch_size=MixedTokenData.compute_batch_size(token_seq))

        self.token_seq = token_seq
        self.entity_id_rage = entity_id_rage
        self.token_seq_len = token_seq_len

    @staticmethod
    def compute_batch_size(token_seq: torch.Tensor) -> int:
        if len(token_seq.shape) == 1:
            return 1
        else:
            return token_seq.shape[0]

    @property
    def token_type_mask(self) -> torch.Tensor:
        ret = torch.zeros_like(self.token_seq, dtype=torch.int8)
        for entity_type in TokenType:
            ret[self.entity_mask(entity_type)] = entity_type.value

        return ret

    def entity_mask(self, entity_type: TokenType) -> torch.Tensor:
        """
        Return a mask of the entity_type.
        """
        return (self.token_seq >= self.entity_id_rage[entity_type].start) & (
            self.token_seq < self.entity_id_rage[entity_type].end
        )


class MixedTokenDataset(FoundationModelDataset):
    def __init__(
        self,
        sents: List[List[TextSpan]],
        text_tokenizer: str,
        entity_tokenizer: str,  # TODO: support multiple entity tokenizers
        max_text_len: int,
    ) -> None:
        super().__init__()
        self.sents = sents

        self.init_tokenziers(text_tokenizer, entity_tokenizer)

        self.max_text_len = max_text_len

        self.entity_id_rage = {
            TokenType.Text: TokenIdRange(0, self.text_tokenizer.vocab_size),
            TokenType.Entity: TokenIdRange(
                self.text_tokenizer.vocab_size,
                self.text_tokenizer.vocab_size + self.entity_tokenizer.vocab_size,
            ),
        }

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
        for span in sent:
            if span.type == TokenType.Text:
                tokens = self.text_tokenizer(
                    span.text,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_text_len,
                )
            elif span.type == TokenType.Entity:
                tokens = self.entity_tokenizer(
                    span.text,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_text_len,
                )

                tokens["input_ids"] = (
                    tokens["input_ids"] + self.entity_id_rage[TokenType.Entity].start
                )

            else:
                raise ValueError(f"Unknown token type: {span.type}")
            token_seq.append(tokens["input_ids"])

        token_seq = torch.cat(token_seq, dim=0)
        token_seq_len = token_seq.shape[0]

        return MixedTokenData(token_seq, token_seq_len, self.entity_id_rage)

    def collate(self, batch: List[MixedTokenData]) -> MixedTokenData:
        """
        Collate a batch of MixedTokenData.
        """
        # make sure that the entity_id_rage is the same for all the texts in the batch
        entity_id_rage = batch[0].entity_id_rage
        for item in batch:
            if item.entity_id_rage != entity_id_rage:
                raise ValueError(
                    "The entity_id_rage is not the same for all the texts in the batch."
                )

        # pad the token_seq
        pad_idx = self.text_tokenizer.pad_token_id
        batched_tokens = torch.nn.utils.rnn.pad_sequence(
            [text.token_seq for text in batch],
            batch_first=True,
            padding_value=pad_idx,
        )

        token_seq_len = torch.tensor(
            [text.token_seq_len for text in batch], dtype=torch.int64
        )

        return MixedTokenData(batched_tokens, token_seq_len, entity_id_rage)

    @classmethod
    def from_jsonl(
        cls, path: str, text_tokenizer: str, entity_tokenizer: str, max_text_len: int
    ):
        """
        Read a jsonl file, and return a MixedTokenDataset.
        """
        data = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                spans = json.loads(line)
                data.append([TextSpan(**span) for span in spans])
        return cls(data, text_tokenizer, entity_tokenizer, max_text_len)
