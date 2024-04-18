# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass
from enum import Enum

from sfm.models.graphormer.graphormer_config import GraphormerConfig


class VecInitApproach(Enum):
    ZERO_CENTERED_POS: str = "ZERO_CENTERED_POS"
    RELATIVE_POS: str = "RELATIVE_POS"

    def __str__(self):
        return self.value


class DiffusionTrainingLoss(Enum):
    L1: str = "L1"
    MSE: str = "MSE"

    def __str__(self):
        return self.value


class DiffusionTimeStepEncoderType(Enum):
    DISCRETE_LEARNABLE: str = "DISCRETE_LEARNABLE"
    POSITIONAL: str = "POSITIONAL"

    def __str__(self):
        return self.value


@dataclass
class PSMConfig(GraphormerConfig):
    model_type: str = "psm"
    seq_masking_method: str = "transformerM"

    add_rope: bool = True
    num_residues: int = 32
    max_num_aa: int = 1024
    encoder_pair_embed_dim: int = 64
    task: str = "mae"
    sample_mode: bool = False

    train_data_path: str = ""
    valid_data_path: str = ""

    data_path_list: str = ""
    dataset_name_list: str = ""
    dataset_split_raito: str = ""

    lamb_pde: float = 0.01

    mode: str = "score"

    ode_mode: bool = False

    # for PBC
    pbc_expanded_token_cutoff: int = 256
    pbc_expanded_num_cell_per_direction: int = 5
    pbc_expanded_distance_cutoff: float = 20.0
    pbc_use_local_attention: bool = True
    pbc_multigraph_cutoff: float = 5.0
    diff_init_lattice_size: float = 4.0

    lattice_size: float = 4.0

    # for diffusion
    diffusion_sampling: str = "ode"
    diffusion_noise_std: float = 1.0
    ddim_eta: float = 0.0
    ddim_steps: int = 50
    clean_sample_ratio: float = 0.5
    diffusion_training_loss: DiffusionTrainingLoss = DiffusionTrainingLoss.L1
    diffusion_time_step_encoder_type: DiffusionTimeStepEncoderType = (
        DiffusionTimeStepEncoderType.POSITIONAL
    )

    # for equivariant part
    equivar_vec_init: VecInitApproach = VecInitApproach.ZERO_CENTERED_POS

    # for 2D information
    use_2d_atom_features: bool = False
    use_2d_bond_features: bool = False

    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(args)
        for k, v in asdict(self).items():
            if hasattr(args, k):
                setattr(self, k, getattr(args, k))
