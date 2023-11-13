# -*- coding: utf-8 -*-

import itertools
import re
from typing import List, Sequence, Tuple

import numpy as np


def flatten_formula(formula, sites=None, site_count=None):
    site_count = site_count if site_count is not None else {}
    if sites is not None:
        for s in sites:
            if s["element"] not in site_count:
                site_count[s["element"]] = 1
            else:
                site_count[s["element"]] += 1

    output = []

    i = 0
    while i < len(formula):
        char = formula[i]

        if char.isupper():
            elem_start = i
            i += 1
            while i < len(formula) and formula[i].islower():
                i += 1
            elem = formula[elem_start:i]

            if i < len(formula) and formula[i].isdigit():
                num_start = i
                while i < len(formula) and formula[i].isdigit():
                    i += 1
                num = int(formula[num_start:i])
            else:
                num = 1

            if elem not in site_count:
                repeat = 1
            else:
                repeat = site_count[elem] / num
            output.extend([elem] * int(num * repeat))

        elif char == "(":
            group_elem = []
            group_elem_num = []
            i += 1

            while i < len(formula) and formula[i] != ")":
                if formula[i].isupper():
                    elem_start = i
                    i += 1
                    while (
                        i < len(formula) and formula[i] != ")" and formula[i].islower()
                    ):
                        i += 1
                    elem = formula[elem_start:i]

                if i < len(formula) and formula[i] != ")" and formula[i].isdigit():
                    num_start = i
                    while (
                        i < len(formula) and formula[i] != ")" and formula[i].isdigit()
                    ):
                        i += 1
                    num = int(formula[num_start:i])
                else:
                    num = 1
                group_elem.append(elem)
                group_elem_num.append(num)
            i += 1
            if i < len(formula) and formula[i].isdigit():
                num_start = i
                while i < len(formula) and formula[i].isdigit():
                    i += 1
                group_num = int(formula[num_start:i])
            else:
                group_num = 1

            if group_elem[0] not in site_count:
                repeat = 1
            else:
                repeat = site_count[group_elem[0]] // (group_elem_num[0] * group_num)

            for _ in range(group_num * repeat):
                for elem, num in zip(group_elem, group_elem_num):
                    output.extend([elem] * num)

    return output


class ThreeDimTokenizer(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<pad>", "<bos>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<mask>", "<gen>"),
        unk_tok: str = "<unk>",
        pad_tok: str = "<pad>",
        bos_tok: str = "<bos>",
        eos_tok: str = "<eos>",
        gen_tok: str = "<gen>",
        mask_tok: str = "<mask>",
        # cls_tok: str = "<cls>",
        args=None,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_tok = unk_tok
        self.pad_tok = pad_tok
        self.bos_tok = bos_tok
        self.eos_tok = eos_tok
        self.gen_tok = gen_tok
        self.mask_tok = mask_tok
        # self.cls_tok = cls_tok
        self.unk_idx = self.tok_to_idx[self.unk_tok]
        self.padding_idx = self.get_idx(self.pad_tok)
        # self.cls_idx = self.get_idx(self.cls_tok)
        self.mask_idx = self.get_idx(self.mask_tok)
        self.eos_idx = self.get_idx(self.eos_tok)
        self.bos_idx = self.get_idx(self.bos_tok)
        self.gen_idx = self.get_idx(self.gen_tok)
        self.all_special_tokens = [
            self.pad_tok,
            self.bos_tok,
            self.eos_tok,
            self.unk_tok,
            self.mask_tok,
            self.gen_tok,
        ]
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def tokenize(
        self, text, sites=None, prepend_bos=True, append_eos=False, append_gen=False
    ) -> List[str]:
        res = flatten_formula(text, sites)
        if prepend_bos:
            res = [self.bos_tok] + res
        if append_gen:
            res = res + [self.gen_tok]
        if append_eos:
            res = res + [self.eos_tok]
        return res

    def encode(
        self, text, sites=None, prepend_bos=True, append_eos=False, append_gen=False
    ):
        return np.array(
            [
                self.get_idx(tok)
                for tok in self.tokenize(
                    text, sites, prepend_bos, append_eos, append_gen
                )
            ]
        )

    def decode(self, tokens, coordindates, mask, digit_scale=None) -> str:
        seq_len = tokens.shape[0]
        mask = mask[:seq_len]
        ret = []
        coordinates_index = 0
        digit_scale = digit_scale if digit_scale else 1
        for i in range(seq_len):
            if mask[i] == 1:
                x, y, z = coordindates[coordinates_index]
                if coordinates_index < 3:
                    ret.extend(list(map(str, [x, y, z])))
                else:
                    ret.extend(
                        list(
                            map(
                                str, [x / digit_scale, y / digit_scale, z / digit_scale]
                            )
                        )
                    )
                coordinates_index += 1
            else:
                if tokens[i] not in [self.bos_idx, self.eos_idx, self.padding_idx]:
                    ret.append(self.get_tok(tokens[i]))
                if tokens[i] == self.eos_idx:
                    break
        return " ".join(ret)

    def get_ele_num(self, text):
        res = flatten_formula(text)
        return res

    @classmethod
    def from_file(cls, filename, args=None):
        tokens = []
        with open(filename, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token = line.split()[0]
                tokens.append(token)
        return cls(tokens, args=args)


if __name__ == "__main__":
    tokenizer = ThreeDimTokenizer(
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Li", "O", "Mn", "Co"]
    )
    print(tokenizer.all_toks)
    print(tokenizer.tok_to_idx)
    print(tokenizer.tokenize("Li7Mn4CoO12", prepend_bos=True, append_eos=False))
    print(tokenizer.encode("Li7Mn4CoO12", prepend_bos=True, append_eos=False))
    print()

    tokenizer2 = ThreeDimTokenizer.from_file("dict.txt")
    print(len(tokenizer2))
    print(tokenizer.tokenize("Li7Mn4CoO12", prepend_bos=True, append_eos=False))
    print(tokenizer.encode("Li7Mn4CoO12", prepend_bos=True, append_eos=False))
    print(tokenizer.tokenize("LiCaPb", prepend_bos=True, append_eos=False))
    print(tokenizer.tokenize("Ho2(Ni5B3)3", prepend_bos=True, append_eos=False))
    print()
