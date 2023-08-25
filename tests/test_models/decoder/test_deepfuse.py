# -*- coding: utf-8 -*-
import unittest
from typing import Any
from unittest.mock import patch

import torch

from sfm.data.dec_data.datasets import MixedTokenDataset, TextSpan, TokenType
from sfm.models.decoder.deepfuse.config import DecDeepFuseConfig
from sfm.models.decoder.deepfuse.model import DecDeepFuseModel


class MockTokenizer:
    def __init__(self):
        self.vocab_size = ord("z") + 1
        self.pad_token_id = 46
        self.bos_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, *args, **kwargs) -> Any:
        token_ids = [ord(x) for x in text.split()]
        return {
            "input_ids": token_ids,
        }

    def convert_ids_to_tokens(self, token_ids):
        return [chr(x) for x in token_ids]

    def convert_tokens_to_ids(self, tokens):
        return [ord(x) for x in tokens]


class TestDeepFuse(unittest.TestCase):
    def tok(self, *args, **kwargs):
        self.text_tokenizer = MockTokenizer()
        self.entity_tokenizer = MockTokenizer()

    def load_from_pretrained(self):
        pass

    @patch("sfm.data.dec_data.datasets.MixedTokenDataset.init_tokenziers", tok)
    @patch(
        "sfm.models.decoder.deepfuse.model.DecDeepFuseModel.load_from_pretrained",
        load_from_pretrained,
    )
    def test_deepfuse(self):
        config = DecDeepFuseConfig(
            layer_usage="NMSNMS",
            vocab_size=1000,
            entity_vocab_size=1000,
            num_hidden_layers=6,
            hidden_size=256,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=4,
            hidden_act="gelu",
            entity_hidden_size=256,
            entity_intermediate_size=256,
            entity_num_attention_heads=4,
            entity_num_hidden_layers=4,
            entity_hidden_act="gelu",
            adapter_hidden_size=256,
        )

        model = DecDeepFuseModel(config)

        sents = [
            [
                TextSpan("a b c", TokenType.Text),
                TextSpan("d e f", TokenType.Entity),
            ],
            [TextSpan("g h", TokenType.Entity), TextSpan("i j k", TokenType.Entity)],
        ]

        dataset = MixedTokenDataset(sents, "", "", 10, 10)

        data = [dataset[i] for i in range(len(dataset))]
        batch = dataset.collate(data)

        assert batch.batch_size == 2
        assert batch.token_seq.shape == (2, 7), f"{batch.token_seq.shape}"

        output = model(batch)
        assert output is not None

        loss = model.compute_loss(output, batch)
        assert loss is not None


if __name__ == "__main__":
    unittest.main()
