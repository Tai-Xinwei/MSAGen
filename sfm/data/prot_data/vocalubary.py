# -*- coding: utf-8 -*-


import itertools
from typing import List, Sequence, Tuple

from sfm.logging import logger

# from https://github.com/facebookresearch/esm/blob/main/esm/constants.py
proteinseq_toks = {
    "toks": [
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
    ],
}

# Chemical polarity, Net charge at pH 7.4, Hydropathy index, Molecular mass
# [Chemical polarity]
### 0 - Polar
### 1 - Nonpolar
### 2 - Brønsted base
### 3 - Brønsted acid
### 4 - Brønsted acid and base
### 5 - Basic polar
### 6 - Unknown
# [Net charge at pH 7.4]
### 0 - Neutral
### 1 - Positive
### 2 - Negative
### 3 - Unknown
# [	Hydropathy index]
### real value
# [	Molecular mass]
### real value
PROP_FEAT = {
    "L": (1, 0, 3.8, 131.175 / 136.901),
    "A": (1, 0, 1.8, 89.094 / 136.901),
    "G": (1, 0, -0.4, 75.067 / 136.901),
    "V": (1, 0, 4.2, 117.148 / 136.901),
    "S": (0, 0, -0, 8, 105.093 / 136.901),
    "E": (2, 2, -3.5, 147.131 / 136.901),
    "R": (5, 1, -4.5, 174.203 / 136.901),
    "T": (0, 0, -0.7, 119.119 / 136.901),
    "I": (1, 0, 4.5, 131.175 / 136.901),
    "D": (2, 2, -3.5, 133.104 / 136.901),
    "P": (1, 0, -1.6, 115.132 / 136.901),
    "K": (3, 1, -3.9, 146.189 / 136.901),
    "Q": (0, 0, -3.5, 146.146 / 136.901),
    "N": (0, 0, -3.5, 132.119 / 136.901),
    "F": (1, 0, 2.8, 165.192 / 136.901),
    "Y": (3, 0, -1.3, 181.191 / 136.901),
    "M": (1, 0, 1.9, 149.208 / 136.901),
    "H": (4, 0, -3.2, 155.156 / 136.901),
    "W": (1, 0, -0.9, 204.228 / 136.901),
    "C": (3, 0, 2.5, 121.154 / 136.901),
    "Unknown": (6, 3, -0.9, 136.901 / 136.901),
}


# from https://github.com/facebookresearch/esm/blob/main/esm/data.py
class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str] = proteinseq_toks["toks"],
        prepend_toks: Sequence[str] = ("<cls>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<mask>",),
        prepend_bos: bool = True,
        append_eos: bool = True,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ["<eos>", "<unk>", "<pad>", "<cls>", "<mask>"]
        self.unique_no_split_tokens = self.all_toks

        self.standard_toks_idx = [self.tok_to_idx[tok] for tok in self.standard_toks]

        self.unk_prop_feat = PROP_FEAT["Unknown"]
        self.idx_prop_feat = dict(
            (self.tok_to_idx[k], PROP_FEAT[k])
            if k in PROP_FEAT
            else (self.tok_to_idx[k], self.unk_prop_feat)
            for k in self.all_toks
        )

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]

    def _feat_text(self, text, feat_id):
        assert feat_id < 4
        return [self.idx_prop_feat[idx][feat_id] for idx in self.encode(text)]

    def _feat_idx(self, indices, feat_id):
        assert feat_id < 4
        return [self.idx_prop_feat[idx][feat_id] for idx in indices]

    def feat_text(self, text):
        return {
            "chem_polar": self._feat_text(text, 0),
            "net_charge": self._feat_text(text, 1),
            "hydropathy": self._feat_text(text, 2),
            "mol_mass": self._feat_text(text, 3),
        }

    def feat_idx(self, indices):
        return {
            "chem_polar": self._feat_idx(indices, 0),
            "net_charge": self._feat_idx(indices, 1),
            "hydropathy": self._feat_idx(indices, 2),
            "mol_mass": self._feat_idx(indices, 3),
        }


if __name__ == "__main__":
    alphabet = Alphabet()
    print(alphabet.all_toks)
    print(alphabet.tok_to_idx)
    print(alphabet.tokenize("AAAALMLMLMLM<mask>AAA"))
    print()
    print(alphabet.feat_text("AAAALMLMLMLM<mask>AAA"))
    print(alphabet.feat_idx(alphabet.encode("AAAALMLMLMLM<mask>AAA")))
