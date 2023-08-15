# -*- coding: utf-8 -*-
import unittest

import torch

from sfm.data.dec_data.dataset import (
    TextMixedWithEntityData,
    TextMixedWithEntityDataset,
    TokenIdRange,
    TokenType,
)
from sfm.data.molecule import Molecule


class TestCreateMolecule(unittest.TestCase):
    def test_create_molecule(self):
        Molecule()


class TestTextMixedWithEntityData(unittest.TestCase):
    def test_text_mixed_with_entity_data(self):
        token_seqs = [
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 101, 102]),
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 100]),
        ]

        entity_id_rage = {
            TokenType.SMILES: TokenIdRange(100, 103),
        }

        texts = [
            TextMixedWithEntityData(token_seq, entity_id_rage)
            for token_seq in token_seqs
        ]

        dataset = TextMixedWithEntityDataset(texts, pad_idx=0)

        batch = dataset.collate(texts)

        expected_batched_tokens = torch.LongTensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 101, 102],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 100, 0],
            ]
        )

        assert torch.equal(batch.token_seq, expected_batched_tokens)

        expected_entity_mask = torch.BoolTensor(
            [
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    False,
                ],
            ]
        )

        assert torch.equal(batch.entity_mask(TokenType.SMILES), expected_entity_mask)


if __name__ == "__main__":
    unittest.main()
