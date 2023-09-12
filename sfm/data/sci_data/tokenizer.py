# -*- coding: utf-8 -*-

import itertools
from typing import List, Sequence, Tuple

import numpy as np


class SciTokenizer(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
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

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def _tokenize(self, text) -> List[str]:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        # TODO: implement a better tokenizer instead of splitting tokens by blankets
        return self._tokenize(text)

    def encode(self, text):
        return np.array([self.get_idx(tok) for tok in self.tokenize(text)])

    @classmethod
    def from_file(cls, filename):
        tokens = []
        with open(filename, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token = line.split()[0]
                tokens.append(token)
        return cls(tokens)


if __name__ == "__main__":
    tokenizer = SciTokenizer(["<a>A", "<a>M", "<a>L"])
    print(tokenizer.all_toks)
    print(tokenizer.tok_to_idx)
    print(
        tokenizer.tokenize(
            "<a>A <a>A <a>A <a>A <a>L <a>M <a>L <a>M <a>L <a>M <a>L <a>M <a>A <a>A <a>A"
        )
    )
    print(
        tokenizer.encode(
            "<a>A <a>A <a>A <a>A <a>L <a>M <a>L <a>M <a>L <a>M <a>L <a>M <a>A <a>A <a>A"
        )
    )
    print()

    tokenizer2 = SciTokenizer.from_file("/mnt/protein/scigpt/dict.txt")
    print(len(tokenizer2))
    print(
        tokenizer2.tokenize(
            "<a>A <a>A <a>A <a>A <a>L <a>M <a>L <a>M <a>L <a>M <a>L <a>M <a>A <a>A <a>A"
        )
    )
    print(
        tokenizer2.encode(
            "<a>A <a>A <a>A <a>A <a>L <a>M <a>L <a>M <a>L <a>M <a>L <a>M <a>A <a>A <a>A"
        )
    )
    print()
    print(
        tokenizer2.encode(
            "[A] <a>P <a>S <a>E <a>T <a>L <a>S <a>L <a>T <a>C <a>A <a>V <a>Y <a>G <a>G <a>S <a>F <a>S <a>G <a>Y <a>Y <a>W <a>S <a>W <a>I <a>R <a>Q <a>P <a>P <a>G <a>K <a>G <a>L <a>E <a>W <a>I <a>G <a>E <a>I <a>N <a>H <a>S <a>G <a>S <a>T <a>N <a>Y <a>N <a>P <a>S <a>L <a>K <a>S <a>R <a>V <a>T <a>I <a>S <a>V <a>D <a>T <a>S <a>K <a>N <a>Q <a>F <a>S <a>L <a>N <a>L <a>S <a>S <a>V <a>T <a>A <a>A <a>D <a>T <a>A <a>V <a>Y <a>Y <a>C <a>A <a>R <a>G <a>S <a>N <a>S <a>V <a>A <a>Y <a>W <a>G <a>R <a>G <a>T <a>L <a>V <a>T <a>V <a>S <a>S [/A]"
        )
    )
    print(
        tokenizer2.encode(
            "[M] <m>Cl <m>c <m>1 <m>c <m>c <m>c <m>c <m>c <m>1 <m>C <m>N <m>C <m>c <m>1 <m>n <m>n <m>[nH] <m>n <m>1 [/M]"
        )
    )
    print(
        tokenizer2.encode(
            "[P] <a>M <a>A <a>F <a>S <a>A <a>E <a>D <a>V <a>L <a>K <a>E <a>Y <a>D <a>R <a>R <a>R <a>R <a>M <a>E <a>A <a>L <a>L <a>L <a>S <a>L <a>Y <a>V <a>E <a>S <a>A <a>H <a>R <a>M <a>R <a>Q <a>G <a>H <a>M <a>I <a>N <a>V <a>K <a>Y <a>I <a>L <a>Y <a>Q <a>L <a>L <a>K <a>K <a>H <a>G <a>H <a>G <a>P <a>D <a>G <a>P <a>D <a>I <a>L <a>T <a>V <a>K <a>T <a>G <a>S <a>K <a>G <a>V <a>L <a>Y <a>D <a>D <a>S <a>F <a>R <a>K <a>I <a>Y <a>T <a>D <a>L <a>G <a>W <a>K <a>F <a>T <a>P <a>L [/P]"
        )
    )
    print(tokenizer2.encode("[T] <i>In <i>4 <i>Li <i>8 <i>Y <i>4 <sg62> [/T]"))
    print(
        tokenizer2.encode(
            "[F] <m>c <m>1 <m>( <m>N <m>c <m>2 <m>n <m>c <m>( <m>N <m>3 <m>C <m>C <m>C <m>[C@H] <m>3 <m>C <m>( <m>N <m>c <m>3 <m>c <m>n <m>c <m>( <m>F <m>) <m>c <m>c <m>3 <m>) <m>= <m>O <m>) <m>n <m>c <m>3 <m>c <m>2 <m>C <m>C <m>C <m>3 <m>) <m>n <m>[nH] <m>c <m>( <m>C <m>( <m>C <m>) <m>( <m>C <m>) <m>O <m>C <m>C <m>= <m>C <m>) <m>c <m>1 [FR] <m>c <m>1 <m>( <m>N <m>c <m>2 <m>n <m>c <m>( <m>N <m>3 <m>C <m>C <m>C <m>[C@H] <m>3 <m>C <m>( <m>N <m>c <m>3 <m>c <m>n <m>c <m>( <m>F <m>) <m>c <m>c <m>3 <m>) <m>= <m>O <m>) <m>n <m>c <m>3 <m>c <m>2 <m>C <m>C <m>C <m>3 <m>) <m>n <m>[nH] <m>c <m>( <m>C <m>( <m>C <m>) <m>( <m>C <m>) <m>O <m>) <m>c <m>1 [/F]"
        )
    )
    print(
        tokenizer2.encode(
            "[B] <m>c <m>1 <m>( <m>N <m>c <m>2 <m>n <m>c <m>( <m>N <m>3 <m>C <m>C <m>C <m>[C@H] <m>3 <m>C <m>( <m>N <m>c <m>3 <m>c <m>n <m>c <m>( <m>F <m>) <m>c <m>c <m>3 <m>) <m>= <m>O <m>) <m>n <m>c <m>3 <m>c <m>2 <m>C <m>C <m>C <m>3 <m>) <m>n <m>[nH] <m>c <m>( <m>C <m>( <m>C <m>) <m>( <m>C <m>) <m>O <m>) <m>c <m>1 [BR] <m>c <m>1 <m>( <m>N <m>c <m>2 <m>n <m>c <m>( <m>N <m>3 <m>C <m>C <m>C <m>[C@H] <m>3 <m>C <m>( <m>N <m>c <m>3 <m>c <m>n <m>c <m>( <m>F <m>) <m>c <m>c <m>3 <m>) <m>= <m>O <m>) <m>n <m>c <m>3 <m>c <m>2 <m>C <m>C <m>C <m>3 <m>) <m>n <m>[nH] <m>c <m>( <m>C <m>( <m>C <m>) <m>( <m>C <m>) <m>O <m>C <m>C <m>= <m>C <m>) <m>c <m>1 [/B]"
        )
    )
