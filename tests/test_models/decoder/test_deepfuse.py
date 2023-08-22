# -*- coding: utf-8 -*-
import unittest
from typing import Any
from unittest.mock import patch

import torch

from sfm.data.dec_data.dataset import MixedTokenDataset, TextSpan, TokenType
from sfm.models.decoder.deepfuse.config import DecDeepFuseConfig
from sfm.models.decoder.deepfuse.model import DecDeepFuse


class MockTokenizer:
    def __init__(self):
        self.vocab_size = ord("z") + 1
        self.pad_token_id = 0

    def __call__(self, text, *args, **kwargs) -> Any:
        token_ids = [ord(x) for x in text.split()]
        return {
            "input_ids": torch.tensor(token_ids),
        }


class TestDeepFuse(unittest.TestCase):
    def tok(self, *args, **kwargs):
        self.text_tokenizer = MockTokenizer()
        self.entity_tokenizer = MockTokenizer()

    def load_from_pretrained(self):
        pass

    @patch("sfm.data.dec_data.dataset.MixedTokenDataset.init_tokenziers", tok)
    @patch(
        "sfm.models.decoder.deepfuse.model.DecDeepFuse.load_from_pretrained",
        load_from_pretrained,
    )
    def test_deepfuse(self):
        config = DecDeepFuseConfig(layer_usage="NMSNMS", vocab_size=1000)

        model = DecDeepFuse(config)

        sents = [
            [
                TextSpan("a b c", TokenType.Text),
                TextSpan("d e f", TokenType.Entity),
            ],
            [TextSpan("g h", TokenType.Entity), TextSpan("i j k", TokenType.Entity)],
        ]

        dataset = MixedTokenDataset(sents, "", "", 10)

        data = [dataset[i] for i in range(len(dataset))]
        batch = dataset.collate(data)

        assert batch.batch_size == 2
        assert batch.token_seq.shape == (2, 6)

        output = model(batch)
        assert output is not None

        loss = model.compute_loss(output, batch)
        assert loss is not None


if __name__ == "__main__":
    unittest.main()
