# -*- coding: utf-8 -*-
import unittest
from unittest.mock import Mock, patch

from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer


class MockSPM(Mock):
    def encode(self, text, out_type=str):
        return text.split(" ")

    def get_piece_size(self):
        return 1000


def get_spm_processor(self, from_slow: bool = False):
    return MockSPM()


class TestSFMDecTokenizer(unittest.TestCase):
    @patch(
        "transformers.models.llama.tokenization_llama.LlamaTokenizer.get_spm_processor",
        get_spm_processor,
    )
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.tokenizer = SFMDecTokenizer(vocab_file=None)

    def check_tokenizer(self, text, expected):
        self.assertEqual(self.tokenizer.tokenize(text), expected)

    def test_tok_text(self):
        self.check_tokenizer("a", ["a"])
        self.check_tokenizer("a b", ["a", "b"])

    def test_tok_mol(self):
        self.check_tokenizer("<mol> C </mol>", ["<mol>", "<m>C", "</mol>"])
        self.check_tokenizer(
            "<mol> CC[Mg+] </mol>", ["<mol>", "<m>C", "<m>C", "<m>[Mg+]", "</mol>"]
        )

    def test_tok_fasta(self):
        self.check_tokenizer(
            "<protein> A </protein>", ["<protein>", "<a>A", "</protein>"]
        )
        self.check_tokenizer(
            "<protein> AB </protein>", ["<protein>", "<a>A", "<a>B", "</protein>"]
        )

    def test_tok_material(self):
        self.check_tokenizer(
            "<material> C </material>", ["<material>", "<i>C", "</material>"]
        )
        self.check_tokenizer(
            "<material> S N <sg123> </material>",
            ["<material>", "<i>S", "<i>N", "<sg123>", "</material>"],
        )

    def test_tok_mixed(self):
        self.check_tokenizer(
            "AA <mol> C </mol> BB", ["AA", "<mol>", "<m>C", "</mol>", "BB"]
        )

        self.check_tokenizer(
            "A <mol> C </mol> B <protein> A </protein> C <material> C </material> D",
            [
                "A",
                "<mol>",
                "<m>C",
                "</mol>",
                "B",
                "<protein>",
                "<a>A",
                "</protein>",
                "C",
                "<material>",
                "<i>C",
                "</material>",
                "D",
            ],
        )

    def test_tok_prompt(self):
        self.check_tokenizer("AA <mol>", ["AA", "<mol>"])

        self.check_tokenizer("AA <mol> BB", ["AA", "<mol>", "<m>B", "<m>B"])


if __name__ == "__main__":
    unittest.main()
