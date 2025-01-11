# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

import torch

from sfm.models.graphormer.graphormer_config import GraphormerConfig


class VecInitApproach(Enum):
    ZERO_CENTERED_POS: str = "ZERO_CENTERED_POS"
    RELATIVE_POS: str = "RELATIVE_POS"
    AUGMENTED_RELATIVE_POS: str = "AUGMENTED_RELATIVE_POS"
    RELATIVE_POS_VEC_BIAS: str = "RELATIVE_POS_VEC_BIAS"

    def __str__(self):
        return self.value


torch.serialization.add_safe_globals([VecInitApproach])


class DiffusionTrainingLoss(Enum):
    L1: str = "L1"
    L2: str = "L2"
    MSE: str = "MSE"
    SmoothL1: str = "SmoothL1"

    def __str__(self):
        return self.value


torch.serialization.add_safe_globals([DiffusionTrainingLoss])


class ForceLoss(Enum):
    L1: str = "L1"
    L2: str = "L2"
    MSE: str = "MSE"
    SmoothL1: str = "SmoothL1"
    NoiseTolerentL1: str = "NoiseTolerentL1"

    def __str__(self):
        return self.value


torch.serialization.add_safe_globals([ForceLoss])


class StressLoss(Enum):
    L1: str = "L1"
    L2: str = "L2"
    MSE: str = "MSE"
    SmoothL1: str = "SmoothL1"
    NoiseTolerentL1: str = "NoiseTolerentL1"

    def __str__(self):
        return self.value


torch.serialization.add_safe_globals([StressLoss])


class DiffusionTimeStepEncoderType(Enum):
    DISCRETE_LEARNABLE: str = "DISCRETE_LEARNABLE"
    POSITIONAL: str = "POSITIONAL"

    def __str__(self):
        return self.value


torch.serialization.add_safe_globals([DiffusionTimeStepEncoderType])


class ForceHeadType(Enum):
    LINEAR: str = "LINEAR"
    GATED_EQUIVARIANT: str = "GATED_EQUIVARIANT"
    MLP: str = "MLP"

    def __str__(self) -> str:
        return self.value


torch.serialization.add_safe_globals([ForceHeadType])


class GaussianFeatureNodeType(Enum):
    EXCHANGABLE: str = "EXCHANGABLE"
    NON_EXCHANGABLE: str = "NON_EXCHANGABLE"
    NON_EXCHANGABLE_DIFF_SELF_EDGE: str = "NON_EXCHANGABLE_DIFF_SELF_EDGE"

    def __str__(self) -> str:
        return self.value


torch.serialization.add_safe_globals([GaussianFeatureNodeType])


class SequenceEncoderOption(Enum):
    PAIR_PLAIN: str = "PAIR_PLAIN"
    NONE: str = "NONE"

    def __str__(self) -> str:
        return self.value


torch.serialization.add_safe_globals([SequenceEncoderOption])


class StructureEncoderOption(Enum):
    GRAPHORMER: str = "GRAPHORMER"
    DIT: str = "DIT"
    NONE: str = "NONE"

    def __str__(self) -> str:
        return self.value


torch.serialization.add_safe_globals([StructureEncoderOption])


class StructureDecoderOption(Enum):
    GEOMFORMER: str = "GEOMFORMER"
    NONE: str = "NONE"

    def __str__(self) -> str:
        return self.value


torch.serialization.add_safe_globals([StructureDecoderOption])


@dataclass
class PSMConfig(GraphormerConfig):
    model_type: str = "psm"
    seq_masking_method: str = "transformerM"

    add_rope: bool = True
    rope_theta: int = 10000
    num_residues: int = 32
    max_num_aa: int = 1024

    num_structure_encoder_layer: int = 4
    encoder_pair_embed_dim: int = 32
    structure_ffn_dim: int = 2028
    structure_hidden_dim: int = 512
    decoder_ffn_dim: int = 2048
    decoder_hidden_dim: int = 512

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
    diff_init_lattice_size_factor: float = 2.859496852322873
    diff_init_lattice_size: float = 10.0
    use_fixed_init_lattice_size: bool = False
    add_unit_cell_virtual_node: bool = False
    use_ddpm_for_material: bool = False

    # for protein and complex
    crop_radius: float = 50.0
    ligand_crop_size: float = 20.0
    max_residue_num: int = 768  # max token number in complex and multi-chain protein
    mode_prob: str = "0.1,0.4,0.5"
    complex_mode_prob: str = "0.1,0.4,0.5"
    sample_ligand_only: bool = False
    plddt_threshold: float = 70.0
    all_atom: bool = False

    # for molecule
    molecule_ref_energy_source: Optional[str] = None
    molecule_outlier_energy_atoms: str = "DEPRECATED_PM6_ATOM_ENERGY_OUTLIER_LIST"

    # for diffusion
    diffusion_sampling: str = "ddpm"
    num_timesteps_stepsize: int = -1
    diffusion_mode: str = "epsilon"
    diffusion_noise_std: float = 1.0
    use_adaptive_noise_std_for_periodic: bool = False
    periodic_diffusion_noise_std_factor: float = 1.0531306506190654
    periodic_lattice_diffusion_noise_std: float = 0.5
    diffusion_rescale_coeff: float = 1.0
    ddim_eta: float = 0.0
    ddim_steps: int = 50
    clean_sample_ratio: float = 0.5
    diffusion_training_loss: DiffusionTrainingLoss = DiffusionTrainingLoss.MSE
    diffusion_time_step_encoder_type: DiffusionTimeStepEncoderType = (
        DiffusionTimeStepEncoderType.POSITIONAL
    )
    align_x0_in_diffusion_loss: bool = True
    separate_noise_head: bool = False

    # EDM
    edm_P_mean: float = -1.2
    edm_P_std: float = 1.5
    edm_sigma_data: float = 16.0
    edm_sample_num_steps: int = 200
    edm_sample_sigma_min: float = 0.004
    edm_sample_sigma_max: float = 160.0
    edm_sample_rho: float = 7.0
    edm_sample_S_churn: float = 0.0
    edm_sample_S_min: float = 0.0
    edm_sample_S_max: float = 3.0e30
    edm_sample_S_noise: float = 1.0
    # for AF3
    af3_sample_gamma_0: float = 0.8
    af3_sample_gamma_min: float = 1.0
    af3_sample_step_scale: float = 1.5
    noise_embedding: str = "fourier"

    # for RL
    psm_finetuneRL_mode: bool = True
    diffusion_sampling_rl: str = "ddpm"
    num_timesteps_stepsize_rl: int = -1
    reward_model: str = "lddt"
    psm_value_step: int = 1
    perturbation_each_traj: int = 2
    reward_weight: float = 10.0
    kl_weight: float = 0.1
    ratio_clip: float = 1e-4

    # EDM
    edm_P_mean: float = -1.2
    edm_P_std: float = 1.5
    edm_sigma_data: float = 16.0
    edm_sample_num_steps: int = 200
    edm_sample_sigma_min: float = 4e-4  # 0.004
    edm_sample_sigma_max: float = 160.0
    edm_sample_rho: float = 7.0
    edm_sample_S_churn: float = 0.0
    edm_sample_S_min: float = 0.0
    edm_sample_S_max: float = 100  # 3.0e30
    edm_sample_S_noise: float = 1.0
    # for AF3
    af3_sample_gamma_0: float = 0.8
    af3_sample_gamma_min: float = 1.0
    af3_sample_step_scale: float = 1.5
    noise_embedding: str = "fourier"
    # for force and stress
    force_loss_type: ForceLoss = ForceLoss.L1
    force_head_type: ForceHeadType = ForceHeadType.GATED_EQUIVARIANT
    node_type_edge_method: GaussianFeatureNodeType = (
        GaussianFeatureNodeType.NON_EXCHANGABLE
    )
    stress_loss_type: StressLoss = StressLoss.L1
    stress_loss_factor: float = 0.1

    # for equivariant part
    equivar_vec_init: VecInitApproach = VecInitApproach.ZERO_CENTERED_POS
    equivar_use_linear_bias: bool = False
    equivar_use_attention_bias: bool = False
    use_smooth_softmax: bool = False
    use_no_pre_cutoff_softmax: bool = False
    smooth_factor: float = 20.0
    use_smooth_equviariant_norm: bool = False
    no_rotary_embedding_for_vector: bool = False
    mlm_from_decoder_feature: bool = False
    disable_data_aug: bool = False
    use_fp32_in_decoder: bool = False

    # for 2D information
    use_2d_atom_features: bool = False
    use_2d_bond_features: bool = False
    use_graphormer_path_edge_feature: bool = True
    preprocess_2d_bond_features_with_cuda: bool = True
    share_attention_bias: bool = False

    # memory efficient attention
    use_memory_efficient_attention: bool = False

    # loss computation options
    rescale_loss_with_std: bool = False
    material_force_loss_ratio: float = 1.0
    material_energy_loss_ratio: float = 1.0
    molecule_force_loss_ratio: float = 1.0
    molecule_energy_loss_ratio: float = 1.0
    energy_per_atom_label_scale: float = 1.0
    molecule_energy_per_atom_std_override: float = 1.0
    decoder_feat4energy: bool = True
    encoderfeat4noise: bool = False
    encoderfeat4mlm: bool = True
    AutoGradForce: bool = False
    supervise_force_from_head_when_autograd: bool = False
    supervise_autograd_stress: bool = False
    NoisePredForce: bool = False
    seq_only: bool = False
    freeze_backbone: bool = False
    hard_dist_loss_raito: float = 1.0
    use_hard_dist_loss: bool = False
    if_total_energy: bool = False
    group_optimizer: bool = False
    group_lr_ratio: float = 1.0

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
    psm_finetune_skip_ori_head: bool = False
    only_use_rotary_embedding_for_protein: bool = False
    psm_validate_for_train_set: bool = False
    # only for dpm solver
    algorithm_type: str = "dpmsolver++"
    solver_order: int = 2
    solver_type: str = "midpoint"

    # for matbench finetuning
    psm_matbench_task_name: str = ""
    psm_matbench_fold_id: int = 0

    # dali pipeline
    use_dali_pipeline: bool = False

    # for structure relaxation
    relax_after_sampling_structure: bool = False
    structure_relax_step_size: float = 0.01
    use_autograd_force_for_relaxation_and_md: bool = False
    relax_ase_steps: int = 8000
    relax_initial_cell_matrix: str = "1,1,1"
    relax_lower_deformation: int = 0
    relax_upper_deformation: int = 0
    relax_deformation_step: int = 5
    relax_fmax: float = 0.01

    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(args)
        for k, v in asdict(self).items():
            if hasattr(args, k):
                setattr(self, k, getattr(args, k))
        self.relax_initial_cell_matrix = [
            int(i) for i in self.relax_initial_cell_matrix.split(",")
        ]
