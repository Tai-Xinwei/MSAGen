# -*- coding: utf-8 -*-
import copy
import gc
import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

##################################################################
# WHEN pretrain with molecule energy, the system/atom reference need
# to be removed for better NN learning

VOCAB = {
    "<pad>": 0,  # padding
    "<unk>": 1,
    # "2"-"128": 1-127, # atom type
    "<cell_corner>": 129,  # use for pbc material
    "L": 130,
    "A": 131,
    "G": 132,
    "V": 133,
    "S": 134,
    "E": 135,
    "R": 136,
    "T": 137,
    "I": 138,
    "D": 139,
    "P": 140,
    "K": 141,
    "Q": 142,
    "N": 143,
    "F": 144,
    "Y": 145,
    "M": 146,
    "H": 147,
    "W": 148,
    "C": 149,
    "X": 150,
    "B": 151,
    "U": 152,
    "Z": 153,
    "O": 154,
    "-": 155,
    ".": 156,
    "<mask>": 157,
    "<cls>": 158,
    "<eos>": 159,
}

MSAVOCAB = {
    "<pad>": 0,  # padding
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "X": 21,
    "B": 22,
    "U": 23,
    "Z": 24,
    "O": 25,
    "-": 26,
    "<mask>": 27,
    "<cls>": 28,
    "<eos>": 29,
}

AF3_MSAVOCAB = {
    "A": 0,
    "B": 25,  # Same as D.
    "C": 4,
    "D": 3,
    "E": 6,
    "F": 13,
    "G": 7,
    "H": 8,
    "I": 9,
    # 'J': 25,  # Same as unknown (X).
    "K": 11,
    "L": 10,
    "M": 12,
    "N": 2,
    "O": 24,  # Same as unknown (X).
    "P": 14,
    "Q": 5,
    "R": 1,
    "S": 15,
    "T": 16,
    "U": 23,  # Same as C.
    "V": 19,
    "W": 17,
    "X": 20,
    "Y": 18,
    "Z": 22,  # Same as E.
    "-": 21,
}


DATA_SPLIT_RATIO = {
    "buckyball_catcher": [600.0 / 6102, 50.0 / 6102, 1 - 650.0 / 6102],
    "double_walled_nanotube": [800.0 / 5032, 100.0 / 5032, 1 - 900.0 / 5032],
    "AT_AT": [3000.0 / 19990, 200.0 / 19990, 1 - 3200.0 / 19990],
    "AT_AT_CG_CG": [2000.0 / 10153, 200.0 / 10153, 1 - 2200.0 / 10153],
    "stachyose": [8000.0 / 27138, 800.0 / 27138, 1 - 8800.0 / 27138],
    "DHA": [8000.0 / 69388, 800.0 / 69388, 1 - 8800.0 / 69388],
    "Ac_Ala3_NHMe": [6000.0 / 85109, 600.0 / 85109, 1 - 6600.0 / 85109],
}
SYSTEM_REF = {
    "Ac_Ala3_NHMe": -620662.75,
    "AT_AT": -1154896.6,
    "AT_AT_CG_CG": -2329950.2,
    "DHA": -631480.2,
    "stachyose": -1578839.0,
    "buckyball_catcher": -2877475.2,  # buckyball_catcher/radius3_broadcast_kmeans
    "double_walled_nanotube": -7799787.0,  # double_walled_nanotube/radius3_broadcast_kmeans
}

PM6_ATOM_REFERENCE = {
    1: -16.369653388041115,
    2: -2038.9032192465093,
    3: -204.69851593083877,
    4: -400.14692331028346,
    5: -679.0854269648153,
    6: -1036.9721554229873,
    7: -1489.7474706309004,
    8: -2046.7225809457013,
    9: -2716.664083129007,
    10: 9.344613136131247e-08,
    11: -4411.522046504277,
    12: -5445.027794252672,
    13: -6598.42237277425,
    14: -7878.595170132059,
    15: -9289.90283680892,
    16: -10834.645744486206,
    17: -12522.430131665715,
    18: -14377.801550004617,
    19: -16324.984854767115,
    20: -18433.482314760826,
    21: -20694.025878493154,
    22: -23120.578878738335,
    23: -25684.926535826566,
    24: -28416.540789365128,
    25: 12693.362010622226,
    26: 29432.75757436947,
    27: 23191.034847208277,
    28: 26678.07937161134,
    29: -44633.752266805706,
    30: -48408.86025358135,
    31: -52372.44082900867,
    32: -56512.1086703412,
    33: -60834.79958507835,
    34: -65342.08279274979,
    35: -70038.55556303446,
    36: -77818.83360448085,
    37: 23422.517109503377,
    38: 30335.465950950675,
    39: 11438.346359699213,
    40: 54831.79791585401,
    41: 3157.2127044979316,
    42: 17680.815901486254,
    43: 58812.13541364664,
    44: 47486.67444641509,
    45: 24015.778943982295,
    46: 33235.438975447134,
    47: 15612.96573520193,
    48: 25612.495497244025,
    49: -10719.68805011228,
    50: 17069.56347634128,
    51: -30093.20575884392,
    52: -31686.01754336678,
    53: -52917.991568412486,
    54: 5789.405818086347,
    55: 26641.29593078723,
    56: 33375.350872970615,
    57: -1.7440895029925985e-10,
    58: -4.403669011008699e-10,
    59: 2.4427221309771238e-11,
    60: -1.611272847669496e-10,
    61: 1.7448591922814817e-11,
    62: -1.5765862135024423e-11,
    63: -2.7381310604358195e-11,
    64: -147.6126453802135,
    65: -3.182965922014105e-12,
    66: 7.648592216225725e-12,
    67: 4.089542656347698e-12,
    68: -3.3053675014069327e-12,
    69: 6.169714627282976e-12,
    70: 0.0,
    71: -325910.94246112177,
    72: 36047.27087028866,
    73: 34403.00501823302,
    74: 21382.92623941272,
    75: 10351.142127485778,
    76: 12753.03823275541,
    77: 36927.56880202808,
    78: 48581.5792850649,
    79: 37692.21889108131,
    80: 22642.089876440186,
    81: 19699.368159305486,
    82: 19386.18078737853,
    83: 32329.08247818586,
    84: 0.0,
    85: 0.0,
    86: 0.0,
    87: 0.0,
    88: 0.0,
    89: 0.0,
    90: 0.0,
    91: 0.0,
    92: 0.0,
    93: 0.0,
    94: 0.0,
    95: 0.0,
    96: 0.0,
    97: 0.0,
    98: 0.0,
    99: 0.0,
    100: 0.0,
    101: 0.0,
    102: 0.0,
    103: 0.0,
    104: 0.0,
    105: 0.0,
    106: 0.0,
    107: 0.0,
    108: 0.0,
    109: 0.0,
    110: 0.0,
    111: 0.0,
    112: 0.0,
    113: 0.0,
    114: 0.0,
    115: 0.0,
    116: 0.0,
    117: 0.0,
    118: 0.0,
    119: 0.0,
    120: 0.0,
    121: 0.0,
    122: 0.0,
    123: 0.0,
    124: 0.0,
    125: 0.0,
    126: 0.0,
    127: 0.0,
    128: 0.0,
    129: 0.0,
}
PM6_ATOM_REFERENCE_LIST = list(PM6_ATOM_REFERENCE.values())

PM6_ATOM_ENERGY_OUTLIER_LIST = [
    2,
    3,
    4,
    10,
    11,
    12,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
]

WB97XD3_ATOM_ENERGY_OUTLIER_LIST = [
    2,
    3,
    4,
    10,
    11,
    12,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
]


# when train ratio is -1, we can use this pre-defined split ratio
def get_data_defult_config(data_name):
    # train ratio , val ratio,test ratio can be int or float.
    has_energy, has_forces, is_pbc = 1, 1, 0
    unit = 0.0433634  # from kcal/mol - > eV
    # unit = 1
    train_ratio, val_ratio, test_ratio = None, None, None
    if data_name.lower() == "qh9":
        train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
        atom_reference = np.zeros([20])
        system_ref = 0.0
    elif data_name.lower() == "pubchem5w":
        train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
        atom_reference = np.array(
            [
                0.0000,
                -376.3395,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                -23905.9824,
                -34351.3164,
                -47201.4062,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                -214228.1250,
                -249841.3906,
            ]
        )
        system_ref = 0.0
    elif data_name.lower() == "oc20":
        unit = 1
        atom_reference = np.array(
            [
                0.0000,
                -3.3995,
                0.0000,
                0.0000,
                0.0000,
                -6.0821,
                -8.2925,
                -8.4169,
                -4.9008,
                0.0000,
                0.0000,
                -1.5417,
                0.0000,
                -3.6177,
                -5.3100,
                -5.2911,
                -4.6126,
                -3.0819,
                0.0000,
                -1.5323,
                -2.4426,
                -6.7297,
                -7.7538,
                -8.3827,
                -8.4383,
                -7.8442,
                -7.1843,
                -6.2366,
                -4.8882,
                -2.9629,
                -0.6659,
                -2.5736,
                -4.0672,
                -4.4946,
                -3.8119,
                0.0000,
                0.0000,
                -1.3493,
                -2.3484,
                -7.0278,
                -8.5368,
                -9.5898,
                -9.8338,
                -9.2727,
                -8.3798,
                -6.7955,
                -4.7875,
                -1.8926,
                -0.2102,
                -2.0652,
                -3.3848,
                -3.6490,
                -3.0334,
                0.0000,
                0.0000,
                -1.4970,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                -9.9214,
                -11.1467,
                -11.6672,
                -11.0532,
                -10.1330,
                -8.2885,
                -5.7752,
                -2.6926,
                0.2878,
                -1.8088,
                -3.0232,
                -3.3307,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ]
        )
        system_ref = 0.0
        has_energy, has_forces, is_pbc = 1, 1, 1
    elif data_name.lower() == "mattersim":
        # unit = 1 / 0.0433634  # use kcal/mol as unit
        atom_reference = np.zeros([20])
        system_ref = 0.0
        has_energy, has_forces, is_pbc = 1, 1, 1
    elif data_name.lower() == "spice":
        # unit = 1
        atom_reference = np.array(
            [
                0.0000e00,
                -3.7092e02,
                0.0000e00,
                -4.6022e03,
                0.0000e00,
                -1.5665e04,
                -2.3935e04,
                -3.4383e04,
                -4.7234e04,
                -6.2692e04,
                0.0000e00,
                -1.0179e05,
                -1.2511e05,
                0.0000e00,
                -1.8178e05,
                -2.1428e05,
                -2.4990e05,
                -2.8884e05,
                0.0000e00,
                -3.7648e05,
                -4.2486e05,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                -1.6153e06,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                -1.8675e05,
            ]
        )
        system_ref = 0.0
    elif data_name.lower() in ["deshaw", "deshaw_120", "deshaw_400", "deshaw_650"]:
        # unit = 1
        atom_reference = np.array(
            [
                0.0000,
                -366.5363,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                -23896.9922,
                -34311.2500,
                -47173.9297,
                0.0000,
                0.0000,
                -101742.2188,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                -249763.8750,
                -288754.4062,
            ]
        )
        system_ref = 0.0
    elif data_name.lower() == "gems":
        # unit = 1
        # atom_reference = np.zeros([20])
        atom_reference = np.array(
            [
                0.0000,
                -53.37939453,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                -174.86627197,
                -93.69719696,
                -121.22570801,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                -70.425354,
                0.0000,
            ]
        )
        system_ref = 0.0
        train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    elif data_name.lower() == "oc20":
        unit = 1
        atom_reference = np.array(
            [
                0.0000,
                -3.3995,
                0.0000,
                0.0000,
                0.0000,
                -6.0821,
                -8.2925,
                -8.4169,
                -4.9008,
                0.0000,
                0.0000,
                -1.5417,
                0.0000,
                -3.6177,
                -5.3100,
                -5.2911,
                -4.6126,
                -3.0819,
                0.0000,
                -1.5323,
                -2.4426,
                -6.7297,
                -7.7538,
                -8.3827,
                -8.4383,
                -7.8442,
                -7.1843,
                -6.2366,
                -4.8882,
                -2.9629,
                -0.6659,
                -2.5736,
                -4.0672,
                -4.4946,
                -3.8119,
                0.0000,
                0.0000,
                -1.3493,
                -2.3484,
                -7.0278,
                -8.5368,
                -9.5898,
                -9.8338,
                -9.2727,
                -8.3798,
                -6.7955,
                -4.7875,
                -1.8926,
                -0.2102,
                -2.0652,
                -3.3848,
                -3.6490,
                -3.0334,
                0.0000,
                0.0000,
                -1.4970,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                -9.9214,
                -11.1467,
                -11.6672,
                -11.0532,
                -10.1330,
                -8.2885,
                -5.7752,
                -2.6926,
                0.2878,
                -1.8088,
                -3.0232,
                -3.3307,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ]
        )
        system_ref = 0.0
        has_energy, has_forces, is_pbc = 1, 1, 1
    elif data_name.lower() == "mattersim":
        atom_reference = np.zeros([20])
        system_ref = 0.0
        has_energy, has_forces, is_pbc = 1, 1, 1
    else:
        atom_reference = np.zeros([20])
        system_ref = SYSTEM_REF[data_name]
        train_ratio, val_ratio, test_ratio = DATA_SPLIT_RATIO[data_name]
    return (
        atom_reference,
        system_ref,
        train_ratio,
        val_ratio,
        test_ratio,
        has_energy,
        has_forces,
        is_pbc,
        unit,
    )


##################################################################
# hamitonian related utility functions
def get_conv_variable(basis="def2-tzvp"):
    # str2order = {"s":0,"p":1,"d":2,"f":3}
    chemical_symbols = [
        "n",
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
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

    convention_dict = {
        "def2-tzvp": Namespace(
            atom_to_orbitals_map={
                1: "sssp",
                6: "ssssspppddf",
                7: "ssssspppddf",
                8: "ssssspppddf",
                9: "ssssspppddf",
                15: "ssssspppppddf",
                16: "ssssspppppddf",
                17: "ssssspppppddf",
            },
            orbital_idx_map={
                "s": [0],
                "p": [1, 2, 0],
                "d": [0, 1, 2, 3, 4],
                "f": [0, 1, 2, 3, 4, 5, 6],
            },
            orbital_sign_map={
                "s": [1],
                "p": [1, 1, 1],
                "d": [1, 1, 1, 1, 1],
                "f": [1, 1, 1, 1, 1, 1, 1],
            },
            max_block_size=37,
            orbital_order_map={
                1: [0, 1, 2, 3],
                6: list(range(11)),
                7: list(range(11)),
                8: list(range(11)),
                9: list(range(11)),
                15: list(range(13)),
                16: list(range(13)),
                17: list(range(13)),
            },
        ),
        # 'back2pyscf': Namespace(
        #     atom_to_orbitals_map={1: 'sssp', 6: 'ssssspppddf', 7: 'ssssspppddf', 8: 'ssssspppddf',
        #                            9: 'ssssspppddf', 15: 'ssssspppppddf', 16: 'ssssspppppddf',
        #                            17: 'ssssspppppddf'},
        #     orbital_idx_map={'s': [0], 'p': [2, 0, 1], 'd': [0, 1, 2, 3, 4], 'f': [0, 1, 2, 3, 4, 5, 6]},
        #     orbital_sign_map={'s': [1], 'p': [1, 1, 1], 'd': [1, 1, 1, 1, 1], 'f': [1, 1, 1, 1, 1, 1, 1]},
        #     orbital_order_map={
        #         1: [0, 1, 2, 3], 6: list(range(11)), 7: list(range(11)),
        #         8: list(range(11)), 9: list(range(11)), 15: list(range(13)),
        #         16: list(range(13)), 17: list(range(13)),
        #     },
        # ),
    }

    # orbital reference (for def2tzvp basis)
    orbitals_ref = {}
    mask = {}
    if basis == "def2-tzvp":
        orbitals_ref[1] = np.array([0, 0, 0, 1])  # H: 2s 1p
        orbitals_ref[6] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3])  # C: 3s 2p 1d
        orbitals_ref[7] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3])  # N: 3s 2p 1d
        orbitals_ref[8] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3])  # O: 3s 2p 1d
        orbitals_ref[9] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3])  # F: 3s 2p 1d
        orbitals_ref[15] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3])  # P
        orbitals_ref[16] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3])  # S
        orbitals_ref[17] = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3])  # Cl
        # 0,1,2,3,4,(5,6,7)(8,9,10)(11,12,13)(14,15,16)(17,18,19)(20,21,22,23,24)(25,26,27,28,29)(30,31,32,33,34,35,36)
        mask[1] = np.array([0, 1, 2, 5, 6, 7])
        mask[6] = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
            ]
        )
        mask[7] = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
            ]
        )
        mask[8] = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
            ]
        )
        mask[9] = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
            ]
        )
        mask[15] = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
            ]
        )
        mask[16] = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
            ]
        )
        mask[17] = np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
            ]
        )

    return convention_dict[basis], orbitals_ref, mask, chemical_symbols


def matrix_transform(hamiltonian, atoms, conv):
    orbitals = ""
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a.item()]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a.item()]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int32)
    transform_signs = np.concatenate(transform_signs)

    hamiltonian_new = hamiltonian[..., transform_indices, :]
    hamiltonian_new = hamiltonian_new[..., :, transform_indices]
    hamiltonian_new = hamiltonian_new * transform_signs[:, None]
    hamiltonian_new = hamiltonian_new * transform_signs[None, :]

    return hamiltonian_new


def split2blocks(H, C, Z, orbitals_ref, mask, block_size):
    local_orbitals = []
    local_orbitals_number = 0
    Z = Z.reshape(-1)
    for z in Z:
        local_orbitals.append(tuple((int(z), int(l)) for l in orbitals_ref[z.item()]))
        local_orbitals_number += sum(2 * l + 1 for _, l in local_orbitals[-1])

    orbitals = local_orbitals

    # atom2orbitals = dict()
    Norb = 0
    begin_index_dict = []
    end_index_dict = []
    for i in range(len(orbitals)):
        begin_index_dict.append(Norb)
        for z, l in orbitals[i]:
            Norb += 2 * l + 1
        end_index_dict.append(Norb)
        # if z not in atom2orbitals:
        #     atom2orbitals[z] = orbitals[i]

    # max_len = max(len(orbit) for orbit in atom2orbitals.values())
    # atom_orb = np.zeros((max(atom2orbitals.keys())+1, max_len), dtype=np.int64)
    # for key, value in atom2orbitals.items():
    #     atom_orb[key, :len(value)] = np.array([it[1] for it in value], dtype=np.int64)

    # block_size = 37#np.int32((atom_orb*2+1).sum(axis=1).max())
    matrix_diag = np.zeros((Z.shape[0], block_size, block_size), dtype=np.float32)
    matrix_non_diag = np.zeros(
        (Z.shape[0] * (Z.shape[0] - 1), block_size, block_size), dtype=np.float32
    )
    matrix_diag_init = np.zeros((Z.shape[0], block_size, block_size), dtype=np.float32)
    matrix_non_diag_init = np.zeros(
        (Z.shape[0] * (Z.shape[0] - 1), block_size, block_size), dtype=np.float32
    )
    mask_diag = np.zeros((Z.shape[0], block_size, block_size), dtype=np.float32)
    mask_non_diag = np.zeros(
        (Z.shape[0] * (Z.shape[0] - 1), block_size, block_size), dtype=np.float32
    )
    non_diag_index = 0
    for i in range(len(orbitals)):  # loop over rows
        for j in range(len(orbitals)):  # loop over columns
            z1 = orbitals[i][0][0]
            z2 = orbitals[j][0][0]
            mask1 = mask[z1]
            mask2 = mask[z2]
            if i == j:
                subblock_H = H[
                    begin_index_dict[i] : end_index_dict[i],
                    begin_index_dict[j] : end_index_dict[j],
                ]
                subblock_C = C[
                    begin_index_dict[i] : end_index_dict[i],
                    begin_index_dict[j] : end_index_dict[j],
                ]
                matrix_diag[i][np.ix_(mask1, mask2)] = subblock_H
                matrix_diag_init[i][np.ix_(mask1, mask2)] = subblock_C
                mask_diag[i] = matrix_diag[i] != 0

            else:
                subblock_H = H[
                    begin_index_dict[i] : end_index_dict[i],
                    begin_index_dict[j] : end_index_dict[j],
                ]
                subblock_C = C[
                    begin_index_dict[i] : end_index_dict[i],
                    begin_index_dict[j] : end_index_dict[j],
                ]
                matrix_non_diag[non_diag_index][np.ix_(mask1, mask2)] = subblock_H
                matrix_non_diag_init[non_diag_index][np.ix_(mask1, mask2)] = subblock_C
                mask_non_diag[non_diag_index] = matrix_non_diag[non_diag_index] != 0
                non_diag_index += 1
    return (
        matrix_diag,
        matrix_non_diag,
        matrix_diag_init,
        matrix_non_diag_init,
        mask_diag,
        mask_non_diag,
    )


# pyscf px py pz
# tp: py pz px
STR2ORDER = {"s": 0, "p": 1, "d": 2, "f": 3}

CHEMICAL_SYMBOLS = [
    "n",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
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

CONVENTION_DICT = {
    "def2-tzvp": Namespace(
        atom_to_orbitals_map={
            1: "sssp",
            6: "ssssspppddf",
            7: "ssssspppddf",
            8: "ssssspppddf",
            9: "ssssspppddf",
            15: "ssssspppppddf",
            16: "ssssspppppddf",
            17: "ssssspppppddf",
        },
        # as 17 is the atom with biggest orbitals, 5*1+5*3+2*5+1*7, thus s ~[0,5) p~[5,5+15) d~[20,20+2*5) f~[30,37)
        # thus max_block_size is 37
        str2idx={
            "s": 0,
            "p": 0 + 5 * 1,
            "d": 0 + 5 * 1 + 5 * 3,
            "f": 0 + 5 * 1 + 5 * 3 + 2 * 5,
        },
        max_block_size=37,
        orbital_idx_map={
            "s": np.array([0]),
            "p": np.array([2, 0, 1]),
            "d": np.array([0, 1, 2, 3, 4]),
            "f": np.array([0, 1, 2, 3, 4, 5, 6]),
        },
    ),
    "def2-svp": Namespace(
        atom_to_orbitals_map={
            1: "ssp",
            6: "sssppd",
            7: "sssppd",
            8: "sssppd",
            9: "sssppd",
        },
        # as 17 is the atom with biggest orbitals, 5*1+5*3+2*5+1*7, thus s ~[0,5) p~[5,5+15) d~[20,20+2*5) f~[30,37)
        # thus max_block_size is 37
        str2idx={"s": 0, "p": 0 + 3 * 1, "d": 0 + 3 * 1 + 2 * 3},
        max_block_size=14,
        orbital_idx_map={
            "s": np.array([0]),
            "p": np.array([2, 0, 1]),
            "d": np.array([0, 1, 2, 3, 4]),
        },
    ),
}


def get_conv_variable_lin(basis="def2-tzvp"):
    # str2order = {"s":0,"p":1,"d":2,"f":3}
    conv = CONVENTION_DICT[basis]
    mask = {}
    for atom in conv.atom_to_orbitals_map:
        mask[atom] = []
        orb_id = 0
        visited_orbital = set()
        for s in conv.atom_to_orbitals_map[atom]:
            if s not in visited_orbital:
                visited_orbital.add(s)
                orb_id = conv.str2idx[s]

            mask[atom].extend(conv.orbital_idx_map[s] + orb_id)
            orb_id += len(conv.orbital_idx_map[s])
    for key in mask:
        mask[key] = np.array(mask[key])
    return conv, None, mask, None


def matrixtoblock_lin(H, Z, mask_lin, max_block_size, sym=False):
    """_summary_

    Args:
        H (_type_): _description_
        Z (_type_): _description_
        mask_lin (_type_): _description_

    Returns:
        _type_: _description_
    """
    n_atom = len(Z)
    Z = Z.reshape(-1)
    new_H = np.zeros(
        (n_atom * max_block_size, n_atom * max_block_size), dtype=np.float32
    )
    new_mask = np.zeros(
        (n_atom * max_block_size, n_atom * max_block_size), dtype=np.float32
    )
    atom_orbitals = []
    for i in range(n_atom):
        atom_orbitals.append(i * max_block_size + mask_lin[Z[i]])
    atom_orbitals = np.concatenate(atom_orbitals, axis=0)
    new_H_tmp = np.zeros((n_atom * max_block_size, len(atom_orbitals)))
    new_mask_tmp = np.zeros((n_atom * max_block_size, len(atom_orbitals)))

    new_mask_tmp[atom_orbitals] = 1
    new_mask[:, atom_orbitals] = new_mask_tmp
    new_mask = new_mask.reshape(n_atom, max_block_size, n_atom, max_block_size)
    new_mask = new_mask.transpose(0, 2, 1, 3)

    # new_H[atom_orbitals][:,atom_orbitals] = H

    new_H_tmp[atom_orbitals] = H
    new_H[:, atom_orbitals] = new_H_tmp
    new_H = new_H.reshape(n_atom, max_block_size, n_atom, max_block_size)
    new_H = new_H.transpose(0, 2, 1, 3)
    if sym:
        unit_matrix = np.ones((n_atom, n_atom))
        # if up and down remove eye
        upper_triangular_matrix = unit_matrix - np.triu(unit_matrix)
        diag = new_H[np.eye(n_atom) == 1]
        non_diag = new_H[upper_triangular_matrix == 1]

        diag_mask = new_mask[np.eye(n_atom) == 1]
        non_diag_mask = new_mask[upper_triangular_matrix == 1]
        del new_H, new_mask, new_H_tmp, new_mask_tmp

        return diag, non_diag, diag_mask, non_diag_mask
    else:
        unit_matrix = np.ones((n_atom, n_atom))
        # if up and down remove eye
        upper_triangular_matrix = unit_matrix - np.eye(len(Z))
        # # if up remove eye
        # upper_triangular_matrix = np.triu(unit_matrix) - np.eye(len(Z))
        diag = new_H[np.eye(n_atom) == 1]
        non_diag = new_H[upper_triangular_matrix == 1]

        diag_mask = new_mask[np.eye(n_atom) == 1]
        non_diag_mask = new_mask[upper_triangular_matrix == 1]
        del new_H, new_mask, new_H_tmp, new_mask_tmp

        return diag, non_diag, diag_mask, non_diag_mask


import torch


def block2matrix(Z, diag, non_diag, mask_lin, max_block_size, sym=False):
    if isinstance(Z, torch.Tensor):
        if not isinstance(mask_lin[1], torch.Tensor):
            for key in mask_lin:
                mask_lin[key] = torch.from_numpy(mask_lin[key])
        Z = Z.reshape(-1)
        n_atom = len(Z)
        atom_orbitals = []
        for i in range(n_atom):
            atom_orbitals.append(i * max_block_size + mask_lin[Z[i].item()])
        atom_orbitals = torch.cat(atom_orbitals, dim=0)

        rebuild_fock = torch.zeros((n_atom, n_atom, max_block_size, max_block_size)).to(
            Z.device
        )

        if sym:
            ## down
            rebuild_fock[torch.eye(n_atom) == 1] = diag
            unit_matrix = torch.ones((n_atom, n_atom))
            down_triangular_matrix = unit_matrix - torch.triu(unit_matrix)
            rebuild_fock[down_triangular_matrix == 1] = 2 * non_diag
            rebuild_fock = (
                rebuild_fock + torch.permute(rebuild_fock, (1, 0, 3, 2))
            ) / 2
        else:
            # no sym
            rebuild_fock[torch.eye(n_atom) == 1] = diag
            unit_matrix = torch.ones((n_atom, n_atom))
            matrix_noeye = unit_matrix - torch.eye(len(Z))
            rebuild_fock[matrix_noeye == 1] = non_diag
            rebuild_fock = (
                rebuild_fock + torch.permute(rebuild_fock, (1, 0, 3, 2))
            ) / 2

        rebuild_fock = torch.permute(rebuild_fock, (0, 2, 1, 3))
        rebuild_fock = rebuild_fock.reshape(
            (n_atom * max_block_size, n_atom * max_block_size)
        )
        rebuild_fock = rebuild_fock[atom_orbitals][:, atom_orbitals]
        return rebuild_fock

    else:
        Z = Z.reshape(-1)
        n_atom = len(Z)
        atom_orbitals = []
        for i in range(n_atom):
            atom_orbitals.append(i * max_block_size + mask_lin[Z[i]])
        atom_orbitals = np.concatenate(atom_orbitals, axis=0)
        rebuild_fock = np.zeros((n_atom, n_atom, max_block_size, max_block_size))

        if sym:
            ## down
            rebuild_fock[np.eye(n_atom) == 1] = diag
            unit_matrix = np.ones((n_atom, n_atom))
            down_triangular_matrix = unit_matrix - np.triu(unit_matrix)
            rebuild_fock[down_triangular_matrix == 1] = 2 * non_diag
            rebuild_fock = (rebuild_fock + rebuild_fock.transpose(1, 0, 3, 2)) / 2
        else:
            # no sym
            rebuild_fock[np.eye(n_atom) == 1] = diag
            unit_matrix = np.ones((n_atom, n_atom))
            matrix_noeye = unit_matrix - np.eye(len(Z))
            rebuild_fock[matrix_noeye == 1] = non_diag

        rebuild_fock = rebuild_fock.transpose(0, 2, 1, 3)
        rebuild_fock = rebuild_fock.reshape(
            (n_atom * max_block_size, n_atom * max_block_size)
        )
        rebuild_fock = rebuild_fock[atom_orbitals][:, atom_orbitals]
        return rebuild_fock


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def plot_probability_heatmaps(true_prob, pred_prob, padding_mask, batched_data):
    """
    绘制真实概率分布、模型预测概率分布和它们的差异热图，并保存到磁盘。
    坐标轴已调整为：横轴表示序列位置，纵轴表示类别。
    """

    save_dir = "./output/vis"
    os.makedirs(save_dir, exist_ok=True)

    # 如果输入是 torch.Tensor，则转换为 numpy 数组
    if hasattr(true_prob, "detach"):
        true_prob = true_prob.detach().cpu().numpy()
    if hasattr(pred_prob, "detach"):
        pred_prob = pred_prob.detach().cpu().numpy()
    if hasattr(padding_mask, "detach"):
        padding_mask = padding_mask.detach().cpu().numpy()

    B, L, num_classes = true_prob.shape
    diff = pred_prob - true_prob

    class_labels = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
        "X",
        "B",
        "U",
        "Z",
        "O",
        "-",
    ]

    for i in range(B):
        unique_id = batched_data["unique_ids"][i]
        sample_true = true_prob[i]  # (L, 26)
        sample_pred = pred_prob[i]  # (L, 26)
        sample_diff = diff[i]  # (L, 26)
        sample_mask = np.broadcast_to(padding_mask[i][:, None], sample_true.shape)

        # 转置为 (26, L) 后，掩码也要跟着转置
        sample_true_T = sample_true.T
        sample_pred_T = sample_pred.T
        sample_diff_T = sample_diff.T
        sample_mask_T = sample_mask.T

        # 绘制真实概率分布热图
        plt.figure(figsize=(12, 6))
        ax = sns.heatmap(
            sample_true_T, mask=sample_mask_T, cmap="viridis", yticklabels=class_labels
        )
        ax.set_title(f"True Probability Distribution for Sample {unique_id}")
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Classes")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"heatmap_true_{unique_id}.png"))
        plt.close()

        # 绘制预测概率分布热图
        plt.figure(figsize=(12, 6))
        ax = sns.heatmap(
            sample_pred_T, mask=sample_mask_T, cmap="viridis", yticklabels=class_labels
        )
        ax.set_title(f"Predicted Probability Distribution for Sample {unique_id}")
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Classes")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"heatmap_pred_{unique_id}.png"))
        plt.close()

        # 绘制差异热图
        plt.figure(figsize=(12, 6))
        ax = sns.heatmap(
            sample_diff_T,
            mask=sample_mask_T,
            cmap="coolwarm",
            center=0,
            yticklabels=class_labels,
        )
        ax.set_title(f"Difference Heatmap for Sample {unique_id}")
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Classes")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"heatmap_diff_{unique_id}.png"))
        plt.close()

        print(f"Saved heatmaps for sample {unique_id} to {save_dir}")


def af3_to_msa_array(af3_array: np.ndarray) -> np.ndarray:
    """
    将 AF3 ID 矩阵（shape=(D,L)）映射为 MSAVOCAB ID 矩阵（shape=(D,L)）。
    """
    # 1. 反转 AF3_MSAVOCAB 得到 id -> token
    af3_id2token = {v: k for k, v in AF3_MSAVOCAB.items()}

    # 2. 构造一个映射数组，长度至少覆盖 AF3 ID 的最大值
    max_id = max(af3_id2token.keys())
    mapping = np.zeros(max_id + 1, dtype=np.int32)

    # 3. 填入映射：mapping[af3_id] = 对应的 MSAVOCAB ID
    for af3_id, token in af3_id2token.items():
        if token not in MSAVOCAB:
            raise KeyError(f"Token '{token}' 不在 MSAVOCAB 中")
        mapping[af3_id] = MSAVOCAB[token]

    # 4. 确保输入中没有越界 ID
    if af3_array.min() < 0 or af3_array.max() > max_id:
        raise ValueError(f"AF3 ID 超出映射范围 (0–{max_id})")

    # 5. 用 NumPy 索引一次性完成映射
    msa_array = mapping[af3_array]

    return msa_array
