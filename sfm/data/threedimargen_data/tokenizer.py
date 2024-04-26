# -*- coding: utf-8 -*-

import itertools
import re
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import numpy as np

# def flatten_formula(formula, sites=None, site_count=None):
#     site_count = site_count if site_count is not None else {}
#     if sites is not None:
#         for s in sites:
#             if s["element"] not in site_count:
#                 site_count[s["element"]] = 1
#             else:
#                 site_count[s["element"]] += 1

#     output = []

#     i = 0
#     while i < len(formula):
#         char = formula[i]

#         if char.isupper():
#             elem_start = i
#             i += 1
#             while i < len(formula) and formula[i].islower():
#                 i += 1
#             elem = formula[elem_start:i]

#             if i < len(formula) and formula[i].isdigit():
#                 num_start = i
#                 while i < len(formula) and formula[i].isdigit():
#                     i += 1
#                 num = int(formula[num_start:i])
#             else:
#                 num = 1

#             if elem not in site_count:
#                 repeat = 1
#             else:
#                 repeat = site_count[elem] / num
#             output.extend([elem] * int(num * repeat))

#         elif char == "(":
#             group_elem = []
#             group_elem_num = []
#             i += 1

#             while i < len(formula) and formula[i] != ")":
#                 if formula[i].isupper():
#                     elem_start = i
#                     i += 1
#                     while (
#                         i < len(formula) and formula[i] != ")" and formula[i].islower()
#                     ):
#                         i += 1
#                     elem = formula[elem_start:i]

#                 if i < len(formula) and formula[i] != ")" and formula[i].isdigit():
#                     num_start = i
#                     while (
#                         i < len(formula) and formula[i] != ")" and formula[i].isdigit()
#                     ):
#                         i += 1
#                     num = int(formula[num_start:i])
#                 else:
#                     num = 1
#                 group_elem.append(elem)
#                 group_elem_num.append(num)
#             i += 1
#             if i < len(formula) and formula[i].isdigit():
#                 num_start = i
#                 while i < len(formula) and formula[i].isdigit():
#                     i += 1
#                 group_num = int(formula[num_start:i])
#             else:
#                 group_num = 1

#             if group_elem[0] not in site_count:
#                 repeat = 1
#             else:
#                 repeat = site_count[group_elem[0]] // (group_elem_num[0] * group_num)

#             for _ in range(group_num * repeat):
#                 for elem, num in zip(group_elem, group_elem_num):
#                     output.extend([elem] * num)

#     return output


def flatten_formula(formula):
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

            output.extend([elem] * num)

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

            for elem, num in zip(group_elem, group_elem_num):
                output.extend([elem] * num * group_num)

    return output


def normalize_frac_coordinate(x: float, margin: float = 1e-4):
    if x < 0:
        x = x + abs(int(x)) + 1
    if x > 1:
        x = x - int(x)
    # adjust value near 0 or 1 to 0
    if min(abs(x - 0), abs(x - 1)) < margin:
        x = float(0.0)
    x = round(x, 6)
    return x


def tokenize_float(f, frac=False):
    # if frac, f should be in [0, 1]
    if frac and f < 0 and f > -0.0001:
        f = float(0.0)
    if f == -0.0:  # in case f == -0.0
        f = float(0.0)
    str_f = format(f, ".4f")
    return [char for char in str_f]


def tokenize_float_v2(f, frac=False):
    # if frac, f should be in [0, 1], can only reserve the decimal part
    if frac and f < 0 and f > -0.0001:
        f = float(0.0)
    if f == -0.0:  # in case f == -0.0
        f = float(0.0)

    is_negative = f < 0

    # reserve 3 digits for the decimal part
    str_f = format(f, ".3f")
    int_part, dec_part = str_f.split(".")
    if is_negative:
        int_part = int_part[1:]
    int_part = "0" * ((3 - len(int_part) % 3) % 3) + int_part
    dec_part = dec_part + "0" * ((3 - len(dec_part) % 3) % 3)
    grouped_int_part = [int_part[i - 3 : i] for i in range(len(int_part), 0, -3)][::-1]
    grouped_dec_part = [dec_part[i : i + 3] for i in range(0, len(dec_part), 3)]

    if is_negative:
        ret = ["-"]
    else:
        ret = []
    ret = ret + grouped_int_part + ["."] + grouped_dec_part
    return ret


def list2float(num_list):
    if len(num_list) == 0:
        return float(0)
    # remove non digit chars, remove duplicate '.' and '-'
    _num_list = []
    for i, char in enumerate(num_list):
        if char == "." and "." not in _num_list:
            _num_list.append(char)
        elif i == 0 and char == "-":
            _num_list.append(char)
        elif char.isdigit():
            _num_list.append(char)
    try:
        return float("".join(_num_list))
    except ValueError:
        print("Error:", _num_list)
        return float(0)


class ThreeDimARGenTokenizer(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Dict[str, str] = OrderedDict(
            {
                "pad": "<pad>",
                "bos": "<bos>",
                "eos": "<eos>",
                "unk": "<unk>",
            }
        ),
        append_toks: Dict[str, str] = OrderedDict(
            {
                "mask": "<mask>",
                "coord": "<coord>",
            }
        ),
        args=None,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks.values())
        self.append_toks = list(append_toks.values())

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        for key, value in prepend_toks.items():
            setattr(self, f"{key}_tok", value)
            setattr(self, f"{key}_idx", self.tok_to_idx[value])
        for key, value in append_toks.items():
            setattr(self, f"{key}_tok", value)
            setattr(self, f"{key}_idx", self.tok_to_idx[value])
        setattr(self, "padding_idx", self.tok_to_idx[self.pad_tok])

        self.args = args

        self.reorder = getattr(args, "reorder", False)
        if self.reorder:
            self.order_tokens = [
                "<orderxyz>",
                "<orderxzy>",
                "<orderyxz>",
                "<orderyzx>",
                "<orderzxy>",
                "<orderzyx>",
            ]
            self.order_token_ids = [self.add_tok(token) for token in self.order_tokens]

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def add_tok(self, tok):
        if tok in self.tok_to_idx:
            return self.tok_to_idx[tok]
        self.all_toks.append(tok)
        self.tok_to_idx[tok] = len(self.all_toks) - 1
        return self.tok_to_idx[tok]

    def tokenize(
        self, text, prepend_bos=True, append_eos=False, append_gen=False
    ) -> List[str]:
        res = flatten_formula(text)
        if prepend_bos:
            res = [self.bos_tok] + res
        if append_gen:
            res = res + [self.coord_tok]
        if append_eos:
            res = res + [self.eos_tok]
        return res

    def encode(self, text, prepend_bos=True, append_eos=False, append_gen=False):
        return np.array(
            [
                self.get_idx(tok)
                for tok in self.tokenize(text, prepend_bos, append_eos, append_gen)
            ]
        )

    def decode(self, args, **kwargs):
        return NotImplementedError

    def decode_batch(self, args, **kwargs):
        return NotImplementedError

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
        if args.tokenizer == "num":
            append_toks = OrderedDict(
                {
                    "mask": "<mask>",
                    "coord": "<coord>",
                    "sg": "<sg>",
                }
            )
            return ThreeDimARGenNumTokenizer(tokens, append_toks=append_toks, args=args)
        elif args.tokenizer == "lan":
            append_toks = OrderedDict(
                {
                    "mask": "<mask>",
                    "coord": "<coord>",
                    "sg": "<sg>",
                    "cs": "<cs>",
                }
            )
            return ThreeDimARGenLanTokenizer(tokens, append_toks=append_toks, args=args)
        elif args.tokenizer == "lanv2":
            append_toks = OrderedDict(
                {
                    "mask": "<mask>",
                    "coord": "<coord>",
                    "sg": "<sg>",
                    "cs": "<cs>",
                }
            )
            return ThreeDimARGenLanV2Tokenizer(
                tokens, append_toks=append_toks, args=args
            )
        elif args.tokenizer == "slices":
            append_toks = OrderedDict(
                {
                    "mask": "<mask>",
                    "gen": "<gen>",
                }
            )
            return ThreeDimARGenSlicesTokenizer(
                tokens, append_toks=append_toks, args=args
            )
        else:
            raise ValueError(f"tokenizer {args.tokenizer} not supported")


class ThreeDimARGenEnergyTokenizer(ThreeDimARGenTokenizer):
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
        if args.tokenizer == "num":
            append_toks = OrderedDict(
                {
                    "mask": "<mask>",
                    "coord": "<coord>",
                    "energy": "<energy>",
                }
            )
            return ThreeDimARGenNumEnergyTokenizer(
                tokens, append_toks=append_toks, args=args
            )
        elif args.tokenizer == "lan":
            append_toks = OrderedDict(
                {
                    "mask": "<mask>",
                    "coord": "<coord>",
                    "energy": "<energy>",
                    "cs": "<cs>",
                }
            )
            return ThreeDimARGenLanEnergyTokenizer(
                tokens, append_toks=append_toks, args=args
            )
        elif args.tokenizer == "lanv2":
            append_toks = OrderedDict(
                {
                    "mask": "<mask>",
                    "coord": "<coord>",
                    "energy": "<energy>",
                    "cs": "<cs>",
                }
            )
            return ThreeDimARGenLanV2EnergyTokenizer(
                tokens, append_toks=append_toks, args=args
            )
        else:
            raise ValueError(f"tokenizer {args.tokenizer} not supported")


class ThreeDimARGenNumTokenizer(ThreeDimARGenTokenizer):
    def decode(self, tokens, coordinates, mask) -> str:
        scale_coords = getattr(self.args, "scale_coords", None)
        seq_len = tokens.shape[0]
        mask = mask[:seq_len]
        sent = []
        lattice = []
        atom_coordinates = []
        coordinates_index = 0
        for i in range(seq_len):
            if mask[i] == 1:
                x, y, z = coordinates[coordinates_index]
                if coordinates_index < 3:
                    lattice.append([x, y, z])
                else:
                    atom_coordinates.append([x, y, z])
                sent.extend(map(str, [x, y, z]))
                coordinates_index += 1
            else:
                if tokens[i] not in [self.bos_idx, self.eos_idx, self.padding_idx]:
                    sent.append(self.get_tok(tokens[i]))
                if tokens[i] == self.eos_idx:
                    break
        sent = " ".join(sent)
        if scale_coords:
            atom_coordinates = [
                [x / scale_coords for x in vec] for vec in atom_coordinates
            ]
        return sent, lattice, atom_coordinates

    def decode_batch(self, tokens, coordindates, masks) -> List[str]:
        ret = []
        bs = tokens.shape[0]
        coords_start = 0
        for i in range(bs):
            num_coords = np.sum(masks[i] != 0)
            sent, lattice, atom_coordinates = self.decode(
                tokens[i],
                coordindates[coords_start : coords_start + num_coords],
                masks[i],
            )
            ret.append((sent, lattice, atom_coordinates))
            coords_start += num_coords
        return ret


class ThreeDimARGenLanTokenizer(ThreeDimARGenTokenizer):
    def tokenize_coords(self, coords):
        return [tokenize_float(f, frac=True) for f in coords]

    def decode(self, tokens) -> str:  # coordinates, mask,
        seq_len = tokens.shape[0]
        sent = []
        tmp_num = []
        numbers = []
        number_flag = False
        for i in range(seq_len):
            if tokens[i] == self.eos_idx:
                number = list2float(tmp_num)
                numbers.append(number)
                sent.append(format(number, ".4f"))
                break
            elif tokens[i] == self.coord_idx:
                number_flag = True
                sent.append(self.get_tok(tokens[i]))
                tmp_num = []
            elif tokens[i] == self.cs_idx:
                number = list2float(tmp_num)
                sent.append(format(number, ".4f"))
                tmp_num = []
                numbers.append(number)
            elif number_flag:
                tmp_num.append(self.get_tok(tokens[i]))
            else:
                if tokens[i] not in [
                    self.bos_idx,
                    self.eos_idx,
                    self.padding_idx,
                ]:
                    sent.append(self.get_tok(tokens[i]))

        coords = [numbers[i : i + 3] for i in range(0, len(numbers), 3)]
        lattice = coords[:3]
        atom_coordinates = coords[3:]

        sent = " ".join(sent)
        return sent, lattice, atom_coordinates

    def decode_batch(self, tokens) -> List[str]:
        ret = []
        bs = tokens.shape[0]
        for i in range(bs):
            output = self.decode(tokens[i])
            ret.append(output)
        return ret


class ThreeDimARGenLanV2Tokenizer(ThreeDimARGenLanTokenizer):
    def tokenize_coords(self, coords):
        return [tokenize_float_v2(f, frac=True) for f in coords]


class ThreeDimARGenSlicesTokenizer(ThreeDimARGenTokenizer):
    def tokenize(
        self, text, prepend_bos=True, append_eos=False, append_gen=False
    ) -> List[str]:
        res = text.strip().split()
        if prepend_bos:
            res = [self.bos_tok] + res
        if append_gen:
            res = res + [self.gen_tok]
        if append_eos:
            res = res + [self.eos_tok]
        return res

    def encode(self, text, prepend_bos=True, append_eos=False, append_gen=False):
        return np.array(
            [
                self.get_idx(tok)
                for tok in self.tokenize(text, prepend_bos, append_eos, append_gen)
            ]
        )

    def decode(self, tokens) -> str:
        seq_len = tokens.shape[0]
        sent = []
        for i in range(seq_len):
            if tokens[i] == self.eos_idx:
                break
            if tokens[i] not in [self.bos_idx, self.padding_idx, self.gen_idx]:
                sent.append(self.get_tok(tokens[i]))

        sent = " ".join(sent)
        return sent

    def decode_batch(self, tokens) -> List[str]:
        ret = []
        bs = tokens.shape[0]
        for i in range(bs):
            output = self.decode(tokens[i])
            ret.append(output)
        return ret


class ThreeDimARGenNumEnergyTokenizer(ThreeDimARGenEnergyTokenizer):
    def decode(self, tokens, coordinates, mask) -> str:
        scale_energy = getattr(self.args, "scale_energy", None)
        seq_len = tokens.shape[0]
        mask = mask[:seq_len]
        sent = []
        coordinates_index = 0
        for i in range(seq_len):
            if mask[i] == 1:
                x, y, z = coordinates[coordinates_index]
                sent.extend(map(str, [x, y, z]))
                coordinates_index += 1
            elif mask[i] == 2:
                energy = np.mean([x, y, z])
                if scale_energy:
                    energy = energy / scale_energy
                sent.append(str(energy))
            else:
                if tokens[i] not in [self.bos_idx, self.eos_idx, self.padding_idx]:
                    sent.append(self.get_tok(tokens[i]))
                if tokens[i] == self.eos_idx:
                    break
        sent = " ".join(sent)

        return sent, energy

    def decode_batch(self, tokens, coordindates, masks) -> List[str]:
        ret = []
        bs = tokens.shape[0]
        coords_start = 0
        for i in range(bs):
            num_coords = np.sum(masks[i] != 0)
            sent, energy = self.decode(
                tokens[i],
                coordindates[coords_start : coords_start + num_coords],
                masks[i],
            )
            ret.append((sent, energy))
            coords_start += num_coords
        return ret


class ThreeDimARGenLanEnergyTokenizer(ThreeDimARGenEnergyTokenizer):
    def decode(self, tokens) -> str:
        seq_len = tokens.shape[0]
        sent = []
        tmp_num = []
        number_flag = False
        for i in range(seq_len):
            if tokens[i] == self.eos_idx:
                break
            elif tokens[i] in [self.coord_idx, self.energy_idx]:
                number_flag = 1
                sent.append(self.get_tok(tokens[i]))
                tmp_num = []
            elif tokens[i] == self.cs_idx:
                number = list2float(tmp_num)
                sent.append(format(number, ".4f"))
                tmp_num = []
            elif number_flag:
                tmp_num.append(self.get_tok(tokens[i]))
            else:
                if tokens[i] not in [
                    self.bos_idx,
                    self.eos_idx,
                    self.padding_idx,
                ]:
                    sent.append(self.get_tok(tokens[i]))

        number = list2float(tmp_num)
        energy = number
        sent.append(format(number, ".4f"))

        sent = " ".join(sent)
        return sent, energy

    def decode_batch(self, tokens) -> List[str]:
        ret = []
        bs = tokens.shape[0]
        for i in range(bs):
            output = self.decode(tokens[i])
            ret.append(output)
        return ret


class ThreeDimARGenLanV2EnergyTokenizer(ThreeDimARGenLanEnergyTokenizer):
    def decode(self, tokens) -> str:
        return super().decode(tokens)


if __name__ == "__main__":
    tokenizer = ThreeDimARGenNumTokenizer(
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Li", "O", "Mn", "Co"]
    )
    print(tokenizer.all_toks)
    print(tokenizer.tok_to_idx)
    print(tokenizer.tokenize("Li7Mn4CoO12", prepend_bos=True, append_eos=False))
    print(tokenizer.encode("Li7Mn4CoO12", prepend_bos=True, append_eos=False))
    print()

    tokenizer2 = ThreeDimARGenNumTokenizer.from_file("dict.txt")
    print(len(tokenizer2))
    print(tokenizer.tokenize("Li7Mn4CoO12", prepend_bos=True, append_eos=False))
    print(tokenizer.encode("Li7Mn4CoO12", prepend_bos=True, append_eos=False))
    print(tokenizer.tokenize("LiCaPb", prepend_bos=True, append_eos=False))
    print(tokenizer.tokenize("Ho2(Ni5B3)3", prepend_bos=True, append_eos=False))
    print()

    tokenizer3 = ThreeDimARGenLanTokenizer.from_file("dict_.txt")
