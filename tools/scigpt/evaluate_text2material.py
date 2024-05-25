# -*- coding: utf-8 -*-
#%%
import numpy as np
import re
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Type, Union
from tqdm import tqdm
import itertools

import smact
from smact.screening import pauling_test

## from https://github.com/msr-ai4science/feynman/blob/818ea614ba15b7107d7c1789961c840402243557/projects/materials/explorers/common/utils/data_utils.py
OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]
EPSILON = 1e-5
chemical_symbols = [
    # 0
    "X",
    # 1
    "H",
    "He",
    # 2
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    # 3
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    # 4
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    # 5
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    # 6
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    # 7
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

def parse_material_string(line):
    segs = line.strip().split()
    if segs[0] != "<material>" or segs[-1] != "</material>":
        return None
    segs.pop()
    segs.pop(0)

    if not segs[-1].startswith(r"<sg"):
        return None

    sg = segs.pop()
    composition = defaultdict(int)
    current_ele = None
    for s in segs:
        if '<i>' not in s:
            return None
        atom = s.replace('<i>', '').strip()
        composition[atom] += 1
    return composition, sg


def check_contains_elements(instruction, composition):
    m = re.search(r"The material contains (?P<elements>[^\.]+)", instruction)
    if m is None:
        return None

    exact_match = not "and other elements" in instruction
    segs = m.group("elements").replace("and other elements", "").split(",")
    elements = [e.strip() for e in segs]

    for e in elements:
        if e not in composition:
            return False

    if exact_match:
        return True, len(composition) == len(segs)
    return True


def smact_validity(
    comp: Union[Tuple[int, ...], Tuple[str, ...]],
    count: Tuple[int, ...],
    use_pauling_test: bool = True,
    include_alloys: bool = True,
    include_cutoff: bool = False,
    use_element_symbol: bool = False,
) -> bool:
    """Computes SMACT validity.

    Args:
        comp: Tuple of atomic number or element names of elements in a crystal.
        count: Tuple of counts of elements in a crystal.
        use_pauling_test: Whether to use electronegativity test. That is, at least in one
            combination of oxidation states, the more positive the oxidation state of a site,
            the lower the electronegativity of the element for all pairs of sites.
        include_alloys: if True, returns True without checking charge balance or electronegativity
            if the crystal is an alloy (consisting only of metals) (default: True).
        include_cutoff: assumes valid crystal if the combination of oxidation states is more
            than 10^6 (default: False).

    Returns:
        True if the crystal is valid, False otherwise.
    """
    assert len(comp) == len(count)
    if use_element_symbol:
        elem_symbols = comp
    else:
        elem_symbols = tuple([chemical_symbols[elem] for elem in comp])  # type:ignore
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    n_comb = np.prod([len(ls) for ls in ox_combos])
    # If the number of possible combinations is big, it'd take too much time to run the smact checker
    # In this case, we assum that at least one of the combinations is valid
    if n_comb > 1e6 and include_cutoff:
        return True
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False

# #%%
# import pickle as pkl
# fn = r'instructv1_mat_beam.pkl'
# fr = open(fn, 'rb')
# DB = pkl.load(fr)
# fr.close()

# #%%
# smact_valid_correct, smact_valid_wrong = 0, 0
# format_incorrect, format_correct = 0, 0

# for k, values in tqdm(DB.items(),total=len(DB)):
#     desc = values[0]
#     for hyp in values[1]:
#         material = parse_material_string(hyp)
#         if material is None:
#             format_incorrect += 1
#             continue
#         format_correct += 1
#         # ret = check_contains_elements(row['description'], material[0])
#         # if ret is None:
#         #    not_related += 1
#         #elif ret == False:
#         #    unhit += 1
#         #else:
#         #    hit += 1

#         comp, count = [], []
#         for k,v in material[0].items():
#             comp.append(k)
#             count.append(v)

#         if smact_validity(tuple(comp), tuple(count),use_element_symbol=True):
#             smact_valid_correct += 1
#         else:
#             smact_valid_wrong += 1

# print(smact_valid_correct, smact_valid_wrong, smact_valid_correct / (smact_valid_correct + smact_valid_wrong))
# print(format_correct, format_incorrect, format_correct / (format_correct + format_incorrect))

# #%%
# unique_elements = defaultdict(lambda : defaultdict(int))
# smact_valid_material = []

# for k, values in tqdm(DB.items(),total=len(DB)):
#     for hyp in values[1]:
#         material = parse_material_string(hyp)
#         if material is None:
#             continue
#         t = [(k,v) for (k,v) in material[0].items()]
#         t = sorted(t, key=lambda e: e[0])
#         unique_elements[tuple(t)][material[1]] += 1

# smact_valid, smact_invalid = 0, 0
# for (composition, sg) in tqdm(unique_elements.items(),total=len(unique_elements)):
#     comp, count = [], []
#     for e in composition:
#         comp.append(e[0])
#         count.append(e[1])
#     if smact_validity(tuple(comp), tuple(count),use_element_symbol=True):
#         smact_valid += 1
#         smact_valid_material.append((composition,sg))
#     else:
#         smact_invalid += 1

# print(smact_valid, smact_invalid, smact_valid / len(unique_elements))
# #%%
# with open(fn.replace('.pkl', '.unique.smactvalid.pkl'), 'wb') as fw:
#     pkl.dump(smact_valid_material, fw)


# #%%
# with open(r"/home/yinxia/blob1.v2/shufxi/data/scigpt/materials_project_data/train_x10.txt", 'r', encoding='utf8') as fr:
#     all_lines1 = [e.strip() for e in fr]


# with open(r"/home/yinxia/blob1.v2/shufxi/data/scigpt/CrystalLLM/train.txt", 'r', encoding='utf8') as fr:
#     all_lines2 = [e.strip() for e in fr]


# with open(r"/home/yinxia/blob1.v2/shufxi/data/scigpt/text2material/train.txt", 'r', encoding='utf8') as fr:
#     all_lines3 = [e.strip() for e in fr]

# with open(r"/home/yinxia/blob1.v2/yinxia/wu2/shared/SFM/SFM.overall.data/instruction_tuning/train.instruct_text2mat.tsv", 'r', encoding='utf8') as fr:
#     all_lines4 = [e.strip() for e in fr]

# with open(r"/home/yinxia/blob1.v2/yinxia/wu2/shared/SFM/SFM.overall.data/instruction_tuning/valid.instruct_text2mat.tsv", 'r', encoding='utf8') as fr:
#     all_lines5 = [e.strip() for e in fr]


# all_lines = all_lines1 + all_lines2 + all_lines3 + all_lines4 + all_lines5


# #%%
# trainSet_material = defaultdict(lambda : defaultdict(int))
# for line in tqdm(all_lines,total=len(all_lines)):
#     m = re.search(r'<material>(.*?)</material>', line)
#     if not m:
#         continue
#     s = m.group(1).strip()
#     S = s.split()
#     if not S[-1].startswith('<sg'):
#         continue

#     ret = defaultdict(int)
#     for s in S[:-1]:
#         ret[s] += 1

#     S2 = sorted(ret.items(), key=lambda e: e[0])
#     trainSet_material[tuple(S2)][S[-1]] += 1

# #%%
# unique_material = []
# for k in unique_elements.keys():
#     if k not in trainSet_material:
#         unique_material.append(k)
# u = len(unique_material)
# t = len(unique_elements)
# print(u,t,u/t)
# cnt = 0
# for k,v in unique_elements.items():
#     cnt += len(v)
# print(cnt)

# #%%
# with open(fn.replace('.pkl', '.unique.pkl'), 'wb') as fw:
#     pkl.dump(unique_elements, fw)
