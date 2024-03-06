# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass
from types import SimpleNamespace

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"


def recursive_simplenamespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = recursive_simplenamespace(v)
    return SimpleNamespace(**d)


@dataclass
class Lddt:
    c_in: int
    no_bins: int = 64
    c_hidden: int = 128


@dataclass
class Distogram:
    c_z: int
    no_bins: int = 64


@dataclass
class Tm:
    c_z: int
    no_bins: int = 64
    enabled: bool = False


@dataclass
class MaskedMsa:
    c_m: int = 128
    c_out: int = 23


@dataclass
class ExperimentallyResolved:
    c_s: int
    c_out: int = 37


@dataclass
class HeadsConfig:
    c_s: int = 384
    c_z: int = 128
    lddt: Lddt = None
    distogram: Distogram = None
    tm: Tm = None
    masked_msa: MaskedMsa = MaskedMsa()
    experimentally_resolved: ExperimentallyResolved = None

    def __post_init__(self):
        self.lddt = Lddt(c_in=self.c_s)
        self.distogram = Distogram(c_z=self.c_z)
        self.tm = Tm(c_z=self.c_z)
        self.experimentally_resolved = ExperimentallyResolved(c_s=self.c_s)


@dataclass
class StructureModuleConfig:
    c_s: int = 384
    c_z: int = 128
    c_ipa: int = 16
    c_resnet: int = 128
    no_heads_ipa: int = 12
    no_qk_points: int = 4
    no_v_points: int = 8
    dropout_rate: float = 0.1
    no_blocks: int = 8
    no_transition_layers: int = 1
    no_resnet_blocks: int = 2
    no_angles: int = 7
    trans_scale_factor: int = 10
    epsilon: float = 1e-8
    inf: float = 1e5


eps = 1e-8
# loss conifg
LossConfig = {
    "distogram": {
        "min_bin": 2.3125,
        "max_bin": 21.6875,
        "no_bins": 64,
        "eps": eps,  # 1e-6,
        "weight": 0.3,
    },
    "experimentally_resolved": {
        "eps": eps,  # 1e-8,
        "min_resolution": 0.1,
        "max_resolution": 3.0,
        "weight": 0.0,
    },
    "fape": {
        "backbone": {
            "clamp_distance": 10.0,
            "loss_unit_distance": 10.0,
            "weight": 0.5,
        },
        "sidechain": {
            "clamp_distance": 10.0,
            "length_scale": 10.0,
            "weight": 0.5,
        },
        "eps": 1e-4,
        "weight": 1.0,
    },
    "plddt_loss": {
        "min_resolution": 0.1,
        "max_resolution": 3.0,
        "cutoff": 15.0,
        "no_bins": 50,
        "eps": eps,  # 1e-10,
        "weight": 0.01,
    },
    "masked_msa": {
        "num_classes": 23,
        "eps": eps,  # 1e-8,
        "weight": 2.0,
    },
    "supervised_chi": {
        "chi_weight": 0.5,
        "angle_norm_weight": 0.01,
        "eps": eps,  # 1e-6,
        "weight": 1.0,
    },
    "violation": {
        "violation_tolerance_factor": 12.0,
        "clash_overlap_tolerance": 1.5,
        "average_clashes": False,
        "eps": eps,  # 1e-6,
        "weight": 0.0,
    },
    "tm": {
        "max_bin": 31,
        "no_bins": 64,
        "min_resolution": 0.1,
        "max_resolution": 3.0,
        "eps": eps,  # 1e-8,
        "weight": 0.0,
        "enabled": False,
    },
    "chain_center_of_mass": {
        "clamp_distance": -4.0,
        "weight": 0.0,
        "eps": eps,
        "enabled": False,
    },
    "eps": eps,
}
loss_config = recursive_simplenamespace(LossConfig)


# HeadsConfig = {
#     "lddt": {
#         "no_bins": 50,
#         "c_in": c_s,
#         "c_hidden": 128,
#     },
#     "distogram": {
#         "c_z": c_z,
#         "no_bins": 64,
#     },
#     "tm": {
#         "c_z": c_z,
#         "no_bins": aux_distogram_bins,
#         "enabled": False,
#     },
#     "masked_msa": {
#         "c_m": c_m,
#         "c_out": 23,
#     },
#     "experimentally_resolved": {
#         "c_s": c_s,
#         "c_out": 37,
#     },
# }
# heads_config = recursive_simplenamespace(HeadsConfig)
