# -*- coding: utf-8 -*-
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch

from sfm.data.tamgent2.datasets import TextToMolData, TextToMolDataset
from sfm.models.tamgent.model import Tamgent2, Tamgent2Config


@dataclass
class MockTextEncodeResult:
    hidden_states: list[torch.Tensor]


class MockTextEncoder(object):
    def __init__(self):
        self.device = "cpu"

    def __call__(self, input_ids, attention_mask, output_hidden_states=True):
        bsz, seq_len = input_ids.shape
        return MockTextEncodeResult(hidden_states=[torch.zeros(bsz, seq_len, 4096)])


class MockBioGPT(object):
    def embed_tokens(self, tokens):
        bsz, seq_len = tokens.shape
        return torch.zeros(bsz, seq_len, 4096)


class MockSmiDecoder(object):
    def __init__(self):
        self.device = "cpu"

        self.biogpt = MockBioGPT()

    def __call__(self, inputs_embeds, attention_mask, return_dict, labels):
        return {"loss": 0.0}


class Test_Tamgent2(unittest.TestCase):
    @patch("sfm.models.tamgent.model.Tamgent2.init_llama")
    @patch("sfm.models.tamgent.model.Tamgent2.init_smi_decoder")
    def test_tamgent2(self, mock_init_llama, mock_init_smi_decoder):
        config = Tamgent2Config(
            molxpt_model="./tests/test_models/tamgent2/molxpt",
            llama_model="./tests/test_models/tamgent2/llama",
        )
        model = Tamgent2(config)
        model.text2mol_proj = torch.nn.Linear(768, 4096)

        model.text_encoder = MockTextEncoder()

        model.smi_decoder = MockSmiDecoder()

        data = [
            TextToMolData(
                smiles="CC", text="Improve the moleclue <m>"
            ),  # TODO: This is correct?
        ]

        dataset = TextToMolDataset(data)
        batch = dataset.collate(dataset.data)

        output = model(batch)

        assert output is not None

        loss = model.compute_loss(output, batch)

        assert loss is not None


if __name__ == "__main__":
    unittest.main()
