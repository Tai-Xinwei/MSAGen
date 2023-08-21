# -*- coding: utf-8 -*-
import json
import pickle
import zlib
from io import StringIO
from typing import List

import numpy as np
from Bio.Data import IUPACData
from Bio.PDB import MMCIFParser

AA32AA1 = {k.upper(): v.upper() for k, v in IUPACData.protein_letters_3to1.items()}


def obj2bstr(obj):
    return zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def bstr2obj(bstr):
    return pickle.loads(zlib.decompress(bstr))


def angle_helper(res, defi: str) -> float:
    ang = res.internal_coord.get_angle(defi)
    if ang is None:
        return np.inf
    return ang


def process_cif(content: str, angle_strs: List[str]) -> dict:
    parser = MMCIFParser()
    structure = parser.get_structure("", StringIO(content))
    structure.atom_to_internal_coordinates()  # verbose=True)
    # assert only one chain, it should be true
    assert len(structure.child_list[0].child_dict) == 1
    chain = structure.child_list[0].child_list[0]
    aa, pos, ang = [], [], []
    for res in chain.child_list:
        aa.append(AA32AA1[res.resname.upper()])
        pos.append(res["CA"].coord)
        ang.append([angle_helper(res, defi) for defi in angle_strs])
    aa = np.array(aa)
    pos = np.array(pos, dtype=np.float32)
    ang = np.array(ang, dtype=np.float32)
    # aa here is a character list, not a string or a np.array
    return {"aa": aa, "pos": pos, "ang": ang}


def process_conf(content: str) -> List[float]:
    if not content:
        return {"conf": None}
    conf_score = json.loads(content)["confidenceScore"]
    conf_score = np.array(conf_score, dtype=np.float32)
    return {"conf": conf_score}
