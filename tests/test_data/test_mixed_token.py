# -*- coding: utf-8 -*-
import unittest
from typing import Any
from unittest.mock import patch

import torch

from sfm.data.dec_data.datasets import (
    MixedTokenData,
    MixedTokenDataset,
    TextSpan,
    TokenType,
)


class MockTokenizer(object):
    def __init__(self, tokenizer_type: str):
        self.tokenizer_type = tokenizer_type

        if tokenizer_type == "text":
            self.vocab_size = 1000
            self.pad_token_id = 999
            self.bos_token_id = 0
            self.eos_token_id = 1

            self.text_base = 10
            self.entity_base = 800
        else:
            self.vocab_size = 100
            self.pad_token_id = 99
            self.bos_token_id = 0
            self.eos_token_id = 1

            self.text_base = 10
            self.entity_base = 50

    def token_to_id(self, token):
        token = token.rstrip("</w>")
        if token == "<s>":
            return self.bos_token_id
        elif token == "</s>":
            return self.eos_token_id
        elif token == "<pad>":
            return self.pad_token_id
        elif token[0] == "[" and token[-1] == "]":
            if token[1] == "/":
                return ord(token[2:-1]) - ord("A") + self.entity_base + 10
            else:
                return ord(token[1:-1]) - ord("A") + self.entity_base
        else:
            return ord(token) - ord("a") + self.text_base

    def id_to_token(self, id):
        if id == self.bos_token_id:
            return "<s>"
        elif id == self.eos_token_id:
            return "</s>"
        elif id == self.pad_token_id:
            return "<pad>"
        elif id >= self.entity_base:
            return "[" + chr(id - self.entity_base + ord("A")) + "]"
        else:
            return chr(id - self.text_base + ord("a"))

    def __call__(self, text, *args, **kwargs) -> Any:
        token_ids = [self.token_to_id(x) for x in text.split()]
        return {
            "input_ids": token_ids,
        }

    def convert_ids_to_tokens(self, token_ids):
        return [self.id_to_token(x) for x in token_ids]

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id(x) for x in tokens]


def init_tokenziers(self, *args, **kwargs):
    self.text_tokenizer = MockTokenizer("text")
    self.entity_tokenizer = MockTokenizer("ent")


MAX_LEN = 100

INIT_TOK_PATH = "sfm.data.dec_data.datasets.MixedTokenDataset.init_tokenziers"


class TestMixedToken(unittest.TestCase):
    def assert_data(self, data, expected):
        self.assertEqual(
            data.pad_idx, expected.pad_idx, f"{data.pad_idx} != {expected.pad_idx}"
        )
        self.assertEqual(
            data.batch_size,
            expected.batch_size,
            f"{data.batch_size} != {expected.batch_size}",
        )

        self.assertTrue(
            torch.equal(data.token_seq, expected.token_seq),
            f"{data.token_seq} != {expected.token_seq}",
        )
        self.assertTrue(
            torch.equal(data.token_type_mask, expected.token_type_mask),
            f"{data.token_type_mask} != {expected.token_type_mask}",
        )
        self.assertTrue(
            torch.equal(data.label_seq, expected.label_seq),
            f"{data.label_seq} != {expected.label_seq}",
        )

    def _test_case(self, sents, expected):
        dataset = MixedTokenDataset(sents, "", "", MAX_LEN, MAX_LEN)

        data = dataset[0]

        self.assert_data(data, expected)

    @patch(INIT_TOK_PATH, init_tokenziers)
    def test_text_only(self):
        sents = [
            [
                TextSpan("a b c", TokenType.Text),
            ]
        ]

        expected = MixedTokenData(
            token_seq=torch.IntTensor([0, 10, 11, 12]),  # [<s> a b c]
            token_type_mask=torch.ByteTensor([0, 0, 0, 0]),
            label_seq=torch.IntTensor([10, 11, 12, 1]),  # [a b c </s>]
            pad_idx=999,
            batch_size=1,
        )

        self._test_case(sents, expected)

    @patch(INIT_TOK_PATH, init_tokenziers)
    def test_entity_only(self):
        sents = [
            [
                TextSpan("[A] a b c [/A]", TokenType.Entity),
            ]
        ]

        expected = MixedTokenData(
            token_seq=torch.IntTensor([0, 50, 10, 11, 12, 810]),  # [<s> [A] a b c [/A]]
            token_type_mask=torch.ByteTensor([0, 1, 1, 1, 1, 0]),
            label_seq=torch.IntTensor(
                [800, 10, 11, 12, 60, 1]
            ),  # [[A] a b c [/A] </s>]
            pad_idx=999,
            batch_size=1,
        )

        self._test_case(sents, expected)

    @patch(INIT_TOK_PATH, init_tokenziers)
    def test_text_then_entity(self):
        sents = [
            [
                TextSpan("a b c", TokenType.Text),
                TextSpan("[A] d e [/A]", TokenType.Entity),
            ]
        ]

        expected = MixedTokenData(
            token_seq=torch.IntTensor(
                [0, 10, 11, 12, 50, 13, 14, 810]
            ),  # [<s> a b c [A] d e [/A]
            token_type_mask=torch.ByteTensor([0, 0, 0, 0, 1, 1, 1, 0]),
            label_seq=torch.IntTensor(
                [10, 11, 12, 800, 13, 14, 60, 1]
            ),  # [a b c [A] d e [/A] </s>]
            pad_idx=999,
            batch_size=1,
        )

        self._test_case(sents, expected)

    @patch(INIT_TOK_PATH, init_tokenziers)
    def test_entity_then_text(self):
        sents = [
            [
                TextSpan("[A] a b [/A]", TokenType.Entity),
                TextSpan("c d e", TokenType.Text),
            ]
        ]

        expected = MixedTokenData(
            token_seq=torch.IntTensor(
                [0, 50, 10, 11, 810, 12, 13, 14]
            ),  # [<s> [A] a b [/A] c d e]
            token_type_mask=torch.ByteTensor([0, 1, 1, 1, 0, 0, 0, 0]),
            label_seq=torch.IntTensor(
                [800, 10, 11, 60, 12, 13, 14, 1]
            ),  # [[A] a b [/A] c d e </s>]
            pad_idx=999,
            batch_size=1,
        )

        self._test_case(sents, expected)

    @patch(INIT_TOK_PATH, init_tokenziers)
    def test_mixed_sentences(self):
        sents = [
            [
                TextSpan("a b c", TokenType.Text),
                TextSpan("[A] d e [/A]", TokenType.Entity),
                TextSpan("f g h", TokenType.Text),
                TextSpan("[B] i j [/B]", TokenType.Entity),
                TextSpan("k l m", TokenType.Text),
            ]
        ]

        expected = MixedTokenData(
            token_seq=torch.IntTensor(
                [
                    0,
                    10,
                    11,
                    12,
                    50,
                    13,
                    14,
                    810,
                    15,
                    16,
                    17,
                    51,
                    18,
                    19,
                    811,
                    20,
                    21,
                    22,
                ]
            ),  # [<s> a b c [A] d e [/A] f g h [B] i j [/B] k l m]
            token_type_mask=torch.ByteTensor(
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
            ),
            label_seq=torch.IntTensor(
                [
                    10,
                    11,
                    12,
                    800,
                    13,
                    14,
                    60,
                    15,
                    16,
                    17,
                    801,
                    18,
                    19,
                    61,
                    20,
                    21,
                    22,
                    1,
                ]
            ),  # [a b c [A] d e [/A] f g h [B] i j [/B] k l m </s>]
            pad_idx=999,
            batch_size=1,
        )

        self._test_case(sents, expected)

    @patch(INIT_TOK_PATH, init_tokenziers)
    def test_two_entity(self):
        sents = [
            [
                TextSpan("[A] a b [/A]", TokenType.Entity),
                TextSpan("[B] c d [/B]", TokenType.Entity),
            ]
        ]

        expected = MixedTokenData(
            token_seq=torch.IntTensor(
                [0, 50, 10, 11, 810, 51, 12, 13, 811]
            ),  # [<s> [A] a b [/A] [B] c d [/B]
            token_type_mask=torch.ByteTensor([0, 1, 1, 1, 0, 1, 1, 1, 0]),
            label_seq=torch.IntTensor(
                [800, 10, 11, 60, 801, 12, 13, 61, 1]
            ),  # [[A] a b [/A] [B] c d [/B] </s>]
            pad_idx=999,
            batch_size=1,
        )

        self._test_case(sents, expected)


if __name__ == "__main__":
    unittest.main()
