# -*- coding: utf-8 -*-
import itertools
import json
import re
import signal
import sys
from argparse import ArgumentParser
from collections import Counter
from multiprocessing import Pool, TimeoutError

import numpy as np
import smact
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from smact.screening import pauling_test
from tqdm import tqdm

LTOL = 0.3
STOL = 0.5
ANGLE_TOL = 10

arg_parser = ArgumentParser()
arg_parser.add_argument("input", type=str, help="input file")
arg_parser.add_argument("--valid", type=bool, default=False)
arg_parser.add_argument("--output", type=str, default=None)
args = arg_parser.parse_args()


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


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    # elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    elem_symbols = comp
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
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold
        )
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


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


def get_pmg_structure(sites, lattice, coordinates):
    structure = Structure(lattice, sites, coordinates)
    return structure


def get_pred_structure_from_slices(data, backend):
    slices = data["prediction"]["slices"]
    reconstructed_structure, final_energy_per_atom_IAP = backend.SLICES2structure(
        slices
    )
    return reconstructed_structure


def get_pred_structure_from_coords(data):
    pred_strucutre = Structure(
        lattice=Lattice(data["prediction"]["lattice"]),
        species=[site["element"] for site in data["sites"]],
        coords=data["prediction"]["coordinates"],
    )
    return pred_strucutre


def get_pred_structure(data, backend=None):
    if "slices" in data["prediction"]:
        return get_pred_structure_from_slices(data, backend)
    else:
        return get_pred_structure_from_coords(data)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out")


def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Disable the alarm
            return result

        return wrapper

    return decorator


@timeout(1)
def get_rms_dist(matcher, pred_structure, gt_structure):
    rms_dist = matcher.get_rms_dist(pred_structure, gt_structure)
    return rms_dist


@timeout(2)
def evaluate_singe(data, backend, matcher):
    sites = [site["element"] for site in data["sites"]]
    try:
        # print("constructing")
        pred_structure = get_pred_structure(data, backend)
        gt_strucutre = Structure(
            lattice=Lattice(data["lattice"]),
            species=sites,
            coords=[site["fractional_coordinates"] for site in data["sites"]],
        )
        constructed = True
    except:
        constructed = False
    if args.valid:
        elem_counter = Counter(sites)
        composition = [
            (elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())
        ]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        comps = tuple(counts.astype("int").tolist())
        # print("smact ing")
        comp_valid = smact_validity(elems, comps)
        if constructed:
            try:
                # print("structing validiting")
                struct_valid = structure_validity(pred_structure)
            except:
                struct_valid = False
        else:
            struct_valid = False
        is_valid = comp_valid and struct_valid
    else:
        is_valid = None
        comp_valid = None
        struct_valid = None
    if args.valid and not is_valid:
        rms_dist = None
    else:
        try:
            # print("getting rms")
            rms_dist = get_rms_dist(matcher, pred_structure, gt_strucutre)
            rms_dist = None if rms_dist is None else rms_dist[0]
        except:
            rms_dist = None
    return rms_dist, is_valid, comp_valid, struct_valid


def evaluate(fname):
    if args.valid:
        print("Using SMACt and structure validity check")
    matcher = StructureMatcher(stol=STOL, angle_tol=ANGLE_TOL, ltol=LTOL)
    rms_dists = []
    valids = []
    smact_valids = []
    struct_valids = []
    backend = None
    with open(fname, "r") as f:
        lines = f.readlines()
        if "slices" in json.loads(lines[0])["prediction"]:
            from invcryrep.invcryrep import InvCryRep

            backend = InvCryRep()
        for line in tqdm(lines):
            data = json.loads(line)
            try:
                rms_dist, is_valid, comp_valid, struct_valid = evaluate_singe(
                    data, backend, matcher
                )
            except:
                rms_dist = None
                is_valid = None
                comp_valid = None
                struct_valid = None
            rms_dists.append(rms_dist)
            valids.append(is_valid)
            smact_valids.append(comp_valid)
            struct_valids.append(struct_valid)

            rms_dists_np = np.array(rms_dists)
            match_rate = sum(rms_dists_np != None) / len(rms_dists_np)  # noqa
            mean_rms_dist = rms_dists_np[rms_dists_np != None].mean()  # noqa
            # print(f"Match rate: {match_rate:.4f}")
            # print(f"Average RMS distance: {mean_rms_dist:.4f}")

        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(rms_dists)  # noqa
        mean_rms_dist = rms_dists[rms_dists != None].mean()  # noqa

        logging_str = f"{args.input}\n"

        if args.valid:
            valid_rate = sum(valids) / len(valids)
            smact_valid_rate = sum(smact_valids) / len(smact_valids)
            struct_valid_rate = sum(struct_valids) / len(struct_valids)
            logging_str += f"SMACt valid rate: {smact_valid_rate:.4f}\n"
            logging_str += f"Valid rate: {valid_rate:.4f}\n"
            logging_str += f"Structure valid rate: {struct_valid_rate:.4f}\n"

        logging_str += f"Match rate: {match_rate:.4f}\n"
        logging_str += f"Average RMS distance: {mean_rms_dist:.4f}\n"
        print(logging_str)
        if args.output:
            with open(args.output, "a+") as fw:
                fw.write(logging_str)


if __name__ == "__main__":
    evaluate(args.input)
