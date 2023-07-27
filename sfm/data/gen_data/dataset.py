# -*- coding: utf-8 -*-
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch

from sfm.data.dataset import InMemoryFoundationModelDataset
from sfm.data.text import Text


class EntityType(Enum):
    SMILES = 1
    FASTA = 2


@dataclass
class TokenIdRange:
    """
    For entities, we use special tokens to represent them, e.g., <m>C rather than C.
    Thus, for each type of entity, we assign [strat, end) token ids to it.
    """

    start: int
    end: int


class TextMixedWithEntityData(Text):
    """
    This represent a text mixed with entities (e.g., SMILES, FASTA, etc.).
    However, everying is in the form of text.
    """

    def __init__(
        self, token_seq: torch.Tensor, entity_id_rage: Dict[EntityType, TokenIdRange]
    ) -> None:
        super().__init__(token_seq)
        self.entity_id_rage = entity_id_rage

    def entity_mask(self, entity_type: EntityType) -> torch.Tensor:
        """
        Return a mask of the entity_type.
        """
        return (self.token_seq >= self.entity_id_rage[entity_type].start) & (
            self.token_seq < self.entity_id_rage[entity_type].end
        )


class TextMixedWithEntityDataset(InMemoryFoundationModelDataset):
    def __init__(self, texts: List[TextMixedWithEntityData], pad_idx: int) -> None:
        super().__init__(texts)
        self.pad_idx = pad_idx

    def collate(self, batch: List[TextMixedWithEntityData]) -> TextMixedWithEntityData:
        """
        Collate a batch of TextMixedWithEntityData.
        """
        # make sure that the entity_id_rage is the same for all the texts in the batch
        entity_id_rage = batch[0].entity_id_rage
        for item in batch:
            if item.entity_id_rage != entity_id_rage:
                raise ValueError(
                    "The entity_id_rage is not the same for all the texts in the batch."
                )

        # pad the token_seq
        batched_tokens = torch.nn.utils.rnn.pad_sequence(
            [text.token_seq for text in batch],
            batch_first=True,
            padding_value=self.pad_idx,
        )

        return TextMixedWithEntityData(batched_tokens, entity_id_rage)

    @classmethod
    def from_jsonl(cls, path):
        """
        Read a jsonl file, and return a TextMixedWithEntityDataset.
        """
        raise NotImplementedError
