# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

from sfm.models.graphormer.graphormer_config import GraphormerConfig


class VecInitApproach(Enum):
    ZERO_CENTERED_POS: str = "ZERO_CENTERED_POS"
    RELATIVE_POS: str = "RELATIVE_POS"
    AUGMENTED_RELATIVE_POS: str = "AUGMENTED_RELATIVE_POS"
    RELATIVE_POS_VEC_BIAS: str = "RELATIVE_POS_VEC_BIAS"

    def __str__(self):
        return self.value


class DiffusionTrainingLoss(Enum):
    L1: str = "L1"
    MSE: str = "MSE"
    SmoothL1: str = "SmoothL1"

    def __str__(self):
        return self.value


class ForceLoss(Enum):
    L1: str = "L1"
    MSE: str = "MSE"
    SmoothL1: str = "SmoothL1"
    NoiseTolerentL1: str = "NoiseTolerentL1"

    def __str__(self):
        return self.value


class DiffusionTimeStepEncoderType(Enum):
    DISCRETE_LEARNABLE: str = "DISCRETE_LEARNABLE"
    POSITIONAL: str = "POSITIONAL"

    def __str__(self):
        return self.value


class ForceHeadType(Enum):
    LINEAR: str = "LINEAR"
    GATED_EQUIVARIANT: str = "GATED_EQUIVARIANT"
    MLP: str = "MLP"

    def __str__(self) -> str:
        return self.value


class GaussianFeatureNodeType(Enum):
    EXCHANGABLE: str = "EXCHANGABLE"
    NON_EXCHANGABLE: str = "NON_EXCHANGABLE"

    def __str__(self) -> str:
        return self.value


@dataclass
class PSMConfig(GraphormerConfig):
    model_type: str = "psm"
    seq_masking_method: str = "transformerM"

    add_rope: bool = True
    num_residues: int = 32
    max_num_aa: int = 1024

    encoder_pair_embed_dim: int = 64
    decoder_ffn_dim: int = 1024

    task: str = "mae"
    sample_mode: bool = False

    train_data_path: str = ""
    valid_data_path: str = ""

    data_path_list: str = ""
    dataset_name_list: str = ""
    dataset_split_raito: str = ""
    dataset_micro_batch_size: str = ""

    lamb_pde: float = 0.01

    # for PBC
    pbc_expanded_token_cutoff: int = 512
    pbc_expanded_num_cell_per_direction: int = 5
    pbc_expanded_distance_cutoff: float = 20.0
    pbc_use_local_attention: bool = False
    pbc_multigraph_cutoff: float = 5.0
    diff_init_lattice_size: float = 4.0
    add_unit_cell_virtual_node: bool = False
    lattice_size: float = 4.0

    # for protein
    crop_radius: float = 25.0
    mode_prob: str = "0.1,0.4,0.5"

    # for diffusion
    diffusion_sampling: str = "ddpm"
    diffusion_mode: str = "epsilon"
    diffusion_noise_std: float = 1.0
    ddim_eta: float = 0.0
    ddim_steps: int = 50
    clean_sample_ratio: float = 0.5
    diffusion_training_loss: DiffusionTrainingLoss = DiffusionTrainingLoss.MSE
    diffusion_time_step_encoder_type: DiffusionTimeStepEncoderType = (
        DiffusionTimeStepEncoderType.POSITIONAL
    )
    align_x0_in_diffusion_loss: bool = True

    # for force
    force_loss_type: ForceLoss = ForceLoss.L1
    force_head_type: ForceHeadType = ForceHeadType.GATED_EQUIVARIANT
    node_type_edge_method: GaussianFeatureNodeType = (
        GaussianFeatureNodeType.NON_EXCHANGABLE
    )

    # for equivariant part
    equivar_vec_init: VecInitApproach = VecInitApproach.ZERO_CENTERED_POS
    equivar_use_linear_bias: bool = False
    equivar_use_attention_bias: bool = False
    use_smooth_softmax: bool = False
    smooth_factor: float = 20.0
    use_smooth_equviariant_norm: bool = False
    no_rotary_embedding_for_vector: bool = False
    mlm_from_decoder_feature: bool = True
    disable_data_aug: bool = False
    use_fp32_in_decoder: bool = False

    # for 2D information
    use_2d_atom_features: bool = False
    use_2d_bond_features: bool = False
    preprocess_2d_bond_features_with_cuda: bool = True

    # memory efficient attention
    use_memory_efficient_attention: bool = False

    # loss computation options
    rescale_loss_with_std: bool = False
    material_force_loss_ratio: float = 1.0
    material_energy_loss_ratio: float = 1.0
    molecule_energy_loss_ratio: float = 1.0
    energy_per_atom_label_scale: float = 1.0
    molecule_energy_per_atom_std_override: float = 1.0
    decoder_feat4energy: bool = True
    AutoGradForce: bool = False
    NoisePredForce: bool = False
    seq_only: bool = False
    freeze_backbone: bool = False
    hard_dist_loss_raito: float = 20.0
    use_hard_dist_loss: bool = False
    if_total_energy: bool = False

    # used in force and noise heads
    num_force_and_noise_head_layers: int = 2

    # used for finetuning and diffusion sampling
    psm_validation_mode: bool = False
    sample_in_validation: bool = False
    num_sampling_time: int = 1
    sampled_structure_output_path: Optional[str] = None
    psm_finetune_mode: bool = False
    psm_sample_structure_in_finetune: bool = False
    psm_finetune_reset_head: bool = False
    psm_finetune_noise_mode: str = "diffusion"
    psm_finetune_valid_noise_mode: str = "diffusion"
    only_use_rotary_embedding_for_protein: bool = False
    psm_validate_for_train_set: bool = False

    # for matbench finetuning
    psm_matbench_task_name: str = ""
    psm_matbench_fold_id: int = 0

    # dali pipeline
    use_dali_pipeline: bool = False

    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(args)
        for k, v in asdict(self).items():
            if hasattr(args, k):
                setattr(self, k, getattr(args, k))
