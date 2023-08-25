# -*- coding: utf-8 -*-
import re

from transformers import BioGptTokenizer


class SFMDecTokenizer(BioGptTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        pad_token="<pad>",
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

    def _tokenize_mol(self, s):
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(s)]
        tokens_tagged = ["<m>" + t + "</w>" for t in tokens]
        return " ".join(tokens_tagged)

    def _tokenize_protein(self, s):
        tokens_tagged = ["<a>" + t + "</w>" for t in s]
        return " ".join(tokens_tagged)

    def _tokenize_antibody(self, s):
        tokens_tagged = ["<a>" + t + "</w>" for t in s]
        return " ".join(tokens_tagged)

    def _tokenize_retrosyntehsis(self, s):
        mol1, mol2 = s.split("[R]")
        mol1_tokens = self._tokenize_mol(mol1)
        mol2_tokens = self._tokenize_mol(mol2)
        return mol1_tokens + " [R]</w> " + mol2_tokens

    def _tokenize_material(self, s):
        tokens_tagged = [t + "</w>" for t in s.split()]
        return " ".join(tokens_tagged)

    def _tokenize(self, text, bypass_tokenizer=False):
        """Returns a tokenized string."""
        if "[R]" in text:
            # print(text)
            if text[-4:] == "[/M]":
                text = (
                    "[M]</w> " + self._tokenize_retrosyntehsis(text[3:-4]) + " [/M]</w>"
                )
            else:
                text = "[M]</w> " + self._tokenize_mol(text[3:-3]) + " [R]</w>"
        elif text[:3] == "[M]":
            if text[-4:] == "[/M]":
                text = "[M]</w> " + self._tokenize_mol(text[3:-4]) + " [/M]</w>"
            else:
                text = "[M]</w> " + self._tokenize_mol(text[3:])
        elif text[:3] == "[P]":
            if text[-4:] == "[/P]":
                text = "[P]</w> " + self._tokenize_protein(text[3:-4]) + " [/P]</w>"
            else:
                text = "[P]</w> " + self._tokenize_protein(text[3:])
        elif text[:3] == "[A]":
            if text[-4:] == "[/A]":
                text = "[A]</w> " + self._tokenize_antibody(text[3:-4]) + " [/A]</w>"
            else:
                text = "[A]</w> " + self._tokenize_antibody(text[3:])
        elif text[:3] == "[T]":
            if text[-4:] == "[/T]":
                text = "[T]</w> " + self._tokenize_material(text[4:-5]) + " [/T]</w>"
            else:
                text = "[T]</w> " + self._tokenize_material(text[4:])
        return text.split()

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # remove BPE
        if tokens[0] == "[T]</w>":
            text = " ".join(tokens).replace("</w>", "")
        else:
            text = (
                "".join(tokens)
                .replace(" ", "")
                .replace("</w>", "")
                .replace("<m>", "")
                .replace("<a>", "")
                .replace("<i>", "")
            )
        return text
