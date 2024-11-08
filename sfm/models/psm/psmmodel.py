# -*- coding: utf-8 -*-
# Copyright (c) Mircrosoft.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sfm.data.psm_data.utils import VOCAB
from sfm.logging import logger
from sfm.models.psm.equivariant.e2former import E2former
from sfm.models.psm.equivariant.equiformer.graph_attention_transformer import Equiformer
from sfm.models.psm.equivariant.equiformer_series import Equiformerv2SO2
from sfm.models.psm.equivariant.equivariant import EquivariantDecoder
from sfm.models.psm.equivariant.geomformer import EquivariantVectorOutput
from sfm.models.psm.equivariant.nodetaskhead import (
    ConditionVectorGatedOutput,
    DiffusionModule,
    DiffusionModule2,
    DiffusionModule3,
    ForceGatedOutput,
    ForceVecOutput,
    NodeTaskHead,
    ScalarGatedOutput,
    VectorGatedOutput,
    VectorOutput,
    VectorProjOutput,
)
from sfm.models.psm.equivariant.vectorVT import VectorVanillaTransformer
from sfm.models.psm.invariant.dit_encoder import PSMDiTEncoder
from sfm.models.psm.invariant.ditp_encoder import PSMPDiTPairEncoder
from sfm.models.psm.invariant.invariant_encoder import PSMEncoder
from sfm.models.psm.invariant.plain_encoder import PSMPairPlainEncoder, PSMPlainEncoder
from sfm.models.psm.modules.embedding import PSMMixEmbedding
from sfm.models.psm.modules.mixembedding import (
    ProteaEmbedding,
    PSMLightEmbedding,
    PSMLightPEmbedding,
    PSMMix3dDitEmbedding,
    PSMMix3dEmbedding,
    PSMMixSeqEmbedding,
    PSMSeqEmbedding,
)
from sfm.models.psm.modules.mixembedding_equiv import PSMMix3DEquivEmbedding
from sfm.models.psm.modules.pbc import CellExpander
from sfm.models.psm.psm_config import ForceHeadType, GaussianFeatureNodeType, PSMConfig
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model

from .modules.autograd import GradientHead
from .modules.confidence_model import lddt
from .modules.dataaug import uniform_random_rotation
from .modules.diffusion import DIFFUSION_PROCESS_REGISTER
from .modules.sampled_structure_converter import SampledStructureConverter
from .modules.timestep_encoder import (
    DiffNoise,
    DiffNoiseEDM,
    NoiseStepSamplerEDM,
    TimeStepSampler,
)


class PSMModel(Model):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(
        self,
        args,
        loss_fn=None,
        not_init=False,
        psm_finetune_head: nn.Module = None,
        molecule_energy_per_atom_std=1.0,
        periodic_energy_per_atom_std=1.0,
        molecule_force_std=1.0,
        periodic_force_std=1.0,
    ):
        """
        Initialize the PSMModel class.

        Args:
            args: Command line arguments.
            loss_fn: The loss function to use.
            data_mean: The mean of the label. For label normalization.
            data_std: The standard deviation of the label. For label normalization.
            not_init: If True, the model will not be initialized. Default is False.
            psm_finetune_head: head used to finetune psm
        """

        super().__init__()
        if not_init:
            return

        self.psm_config = PSMConfig(args)
        self.args = self.psm_config.args
        if args.rank == 0:
            logger.info(self.args)

        self.net = PSM(
            args,
            self.psm_config,
            molecule_energy_per_atom_std=molecule_energy_per_atom_std,
            periodic_energy_per_atom_std=periodic_energy_per_atom_std,
            molecule_force_std=molecule_force_std,
            periodic_force_std=periodic_force_std,
        )

        self.psm_finetune_head = psm_finetune_head
        self.checkpoint_loaded = self.reload_checkpoint()

        if self.psm_config.diffusion_mode == "edm":
            self.diffnoise = DiffNoiseEDM(self.psm_config)
            self.diffnoise.alphas_cumprod = None

            if self.psm_config.diffusion_sampling == "dpm_edm":
                self.diffusion_process = DIFFUSION_PROCESS_REGISTER[
                    self.psm_config.diffusion_sampling
                ](self.diffnoise.alphas_cumprod, self.psm_config)
            elif self.psm_config.diffusion_sampling == "edm":
                self.diffusion_process = None
        else:
            self.diffnoise = DiffNoise(self.psm_config)

            self.diffusion_process = DIFFUSION_PROCESS_REGISTER[
                self.psm_config.diffusion_sampling
            ](self.diffnoise.alphas_cumprod, self.psm_config)

        if self.psm_config.diffusion_mode == "edm":
            self.time_step_sampler = NoiseStepSamplerEDM()
        else:
            self.time_step_sampler = TimeStepSampler(self.psm_config.num_timesteps)

        self.loss_fn = loss_fn(args)

        if self.args.backbone in [
            "vanillatransformer",
            "dit",
            "e2dit",
            "ditp",
            "exp",
            "exp2",
            "exp3",
        ]:
            self.disable_data_aug = getattr(self.args, "disable_data_aug", False)
            # if self.psm_config.psm_finetune_mode:
            # self.disable_data_aug = True
            if self.disable_data_aug:
                logger.warning(
                    f"=== N O T E === Data augmentation is disabled for {self.args.backbone}"
                )

        if self.psm_config.sample_in_validation:
            self.sampled_structure_converter = SampledStructureConverter(
                self.psm_config.sampled_structure_output_path,
                self.psm_config,
                self,
            )

        try:
            mode_prob = [float(item) for item in self.psm_config.mode_prob.split(",")]
            assert len(mode_prob) == 3
            assert sum(mode_prob) == 1.0
        except:
            mode_prob = [0.2, 0.7, 0.1]

        if self.psm_config.diffusion_mode == "protea":
            mode_prob = [0.0, 1.0, 0.0]
            self.psm_config.mask_ratio = 0.0
            self.mode_prob = mode_prob
            logger.info(
                "Protein mode prob is set to [0.0, 1.0, 0.0] in protea mode, mask ratio is set to 0.0"
            )
        else:
            self.mode_prob = mode_prob
            logger.info(f"protein mode prob: {mode_prob}")

        try:
            complex_mode_prob = [
                float(item) for item in self.psm_config.complex_mode_prob.split(",")
            ]
            assert len(complex_mode_prob) == 4
            assert sum(complex_mode_prob) == 1.0
        except:
            complex_mode_prob = [1.0, 0.0, 0.0, 0.0]

        if self.psm_config.diffusion_mode == "protea":
            complex_mode_prob = [1.0, 0.0, 0.0, 0.0]
            self.complex_mode_prob = complex_mode_prob
            logger.info(
                "Complex mode prob is set to [1.0, 0.0, 0.0, 0.0] in protea mode"
            )
        else:
            self.complex_mode_prob = complex_mode_prob
            logger.info(f"complex mode prob: {complex_mode_prob}")

    def reload_checkpoint(self):
        if self.psm_config.psm_finetune_mode or self.psm_config.psm_validation_mode:
            if os.path.exists(self.args.loadcheck_path):
                self.load_pretrained_weights(
                    self.args, checkpoint_path=self.args.loadcheck_path
                )
                loaded = True
                logger.info(f"checkpoint: {self.args.loadcheck_path} is loaded")
            else:
                logger.warning(
                    "Finetune or validation mode, but no checkpoint is loaded"
                )
                loaded = False
        else:
            logger.info("No checkpoint is loaded")
            loaded = False
        if self.psm_config.psm_finetune_mode:
            settings = dict(
                psm_finetune_reset_head=self.psm_config.psm_finetune_reset_head,
                psm_finetune_head=(
                    self.psm_finetune_head.__class__ if self.psm_finetune_head else None
                ),
                psm_finetune_noise_mode=self.psm_config.psm_finetune_noise_mode,
            )
            logger.info(f"Finetune settings: {settings}")
            if self.psm_config.psm_finetune_reset_head:
                self.net.reset_head_for_finetune()
        else:
            assert not self.psm_finetune_head
            self.psm_finetune_head = None

        return loaded

    def half(self):
        to_return = super().half()
        if self.args.backbone == "graphormer" and self.psm_config.use_fp32_in_decoder:
            self.net.decoder = self.net.decoder.float()
            for key in self.net.forces_head:
                self.net.forces_head[key] = self.net.forces_head[key].float()
            for key in self.net.energy_head:
                self.net.energy_head[key] = self.net.energy_head[key].float()
            self.net.noise_head = self.net.noise_head.float()
        return to_return

    def _create_initial_pos_for_diffusion(self, batched_data):
        is_stable_periodic = batched_data["is_stable_periodic"]
        ori_pos = batched_data["pos"][is_stable_periodic]
        n_periodic_graphs = ori_pos.size()[0]
        init_cell_pos = torch.zeros_like(ori_pos)
        num_atoms_cube_root = batched_data["num_atoms"][is_stable_periodic] ** (
            1.0 / 3.0
        )
        lattice_size_factor = (
            self.psm_config.diff_init_lattice_size
            if self.psm_config.use_fixed_init_lattice_size
            else self.psm_config.diff_init_lattice_size_factor
            * num_atoms_cube_root[:, None, None]
        )
        init_cell_pos_input = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ]
            ],
            dtype=ori_pos.dtype,
            device=ori_pos.device,
        ).repeat([n_periodic_graphs, 1, 1]) * lattice_size_factor - (
            lattice_size_factor / 2.0
        )  # centering
        scatter_index = torch.arange(8, device=ori_pos.device).unsqueeze(0).unsqueeze(
            -1
        ).repeat([n_periodic_graphs, 1, 3]) + batched_data["num_atoms"][
            is_stable_periodic
        ].unsqueeze(
            -1
        ).unsqueeze(
            -1
        )
        init_cell_pos = init_cell_pos.scatter(1, scatter_index, init_cell_pos_input)
        batched_data["init_pos"] = torch.zeros_like(batched_data["pos"])
        batched_data["init_pos"][is_stable_periodic] = init_cell_pos

    def _create_protein_mask(self, batched_data):
        token_id = batched_data["token_id"]  # B x T
        # create protein aa mask with mask ratio
        batched_data["protein_masked_pos"] = (
            torch.rand_like(token_id.unsqueeze(-1), dtype=torch.float)
            < self.psm_config.mask_ratio
        ).expand_as(batched_data["pos"])

        # generate a random number [0.15, 0.6] as mask ratio
        if self.psm_config.mask_ratio < 0.15:
            mask_ratio = self.psm_config.mask_ratio
        else:
            mask_ratio = np.random.uniform(0.15, self.psm_config.mask_ratio)

        batched_data["protein_masked_aa"] = (
            torch.rand_like(token_id, dtype=torch.float) < mask_ratio
        )

        masked_pos = batched_data["protein_masked_pos"]

        # for both protein and complex, mask out protein aa and ligands nan/inf coords
        masked_protein = (
            ((token_id > 1) & (token_id < 158))
            .any(dim=-1, keepdim=True)
            .unsqueeze(-1)
            .expand_as(masked_pos)
        )  # mask_protein: B x T x 3
        masked_nan = (
            torch.isnan(batched_data["pos"])
            .any(dim=-1, keepdim=True)
            .expand_as(masked_pos)
        )  # mask_nan: B x T x 3
        masked_inf = (
            torch.isinf(batched_data["pos"])
            .any(dim=-1, keepdim=True)
            .expand_as(masked_pos)
        )  # mask_nan: B x T x 3

        # protein mask is used to mask out protein inf or nan during training
        mask = masked_protein & (masked_nan | masked_inf)

        # fileter low plddt residues
        if "confidence" in batched_data:
            confidence_mask = (
                batched_data["confidence"] < self.psm_config.plddt_threshold
            ) & (batched_data["confidence"] >= 0.0)

            mask = mask | confidence_mask.unsqueeze(-1)

        batched_data["protein_mask"] = mask

    # @torch.compiler.disable(recursive=False)
    def _protein_pretrain_mode(
        self,
        clean_mask,
        aa_mask,
        padding_mask,
        is_protein,
        is_seq_only,
        is_complex,
        time_step,
        noise_step,
        batched_data,
    ):
        """
        For protein pretrain mode, we have 3 modes:
        0: 50% masked seq and 50% noised structure to structure and seq
        1: clean seq to structure
        2: 50% masked seq to structure and masked seq
        For Complex pretrain mode, we have 3 modes:
        0: clean protein seq to protein structure and molecule structure, time_protein == time_ligand
        1: clean protein seq to protein structure and molecule structure, time_protein < time_ligand
        2: 50% masked protein seq and 50% noised structure to protein structure and seq, molecule all clean
        3: clean protein seq and structure to ligand structure

        """
        n_graph, nnodes = aa_mask.size()[:2]
        if batched_data["is_complex"].any():
            mask_choice = np.random.choice(
                np.arange(4), n_graph, p=self.complex_mode_prob
            )
        else:
            mask_choice = np.random.choice(np.arange(3), n_graph, p=self.mode_prob)
        mask_choice = torch.tensor([i for i in mask_choice]).to(clean_mask.device)
        clean_mask = clean_mask.unsqueeze(-1).repeat(1, nnodes)
        mask_choice = mask_choice.unsqueeze(-1).repeat(1, nnodes)

        if time_step is not None:
            time_protein = (
                (torch.rand(n_graph, device=clean_mask.device) * time_step)
                .unsqueeze(-1)
                .repeat(1, nnodes)
            )
            time_step = time_step.unsqueeze(-1).repeat(1, nnodes)
        elif noise_step is not None:
            noise_step_protein = (
                (torch.rand(n_graph, device=clean_mask.device) * noise_step)
                .unsqueeze(-1)
                .repeat(1, nnodes)
            )
            noise_step = noise_step.unsqueeze(-1).repeat(1, nnodes)

        # mode 0:
        aa_mask = torch.where(
            (mask_choice == 0) & is_complex.unsqueeze(-1), False, aa_mask
        )
        clean_mask = torch.where(
            (mask_choice == 0) & is_protein & (~is_complex.unsqueeze(-1)),
            ~aa_mask,
            clean_mask,
        )

        # mode 1:
        aa_mask = torch.where(
            (mask_choice == 1) & ~is_seq_only.unsqueeze(-1), False, aa_mask
        )
        # set ligand time t2 > t1 for mode 1
        if time_step is not None:
            time_step = torch.where(
                (mask_choice == 1) & is_complex.unsqueeze(-1) & is_protein,
                time_protein,
                time_step,
            )
        elif noise_step is not None:
            noise_step = torch.where(
                (mask_choice == 1) & is_complex.unsqueeze(-1) & is_protein,
                noise_step_protein,
                noise_step,
            )

        # mode 2:
        clean_mask = torch.where(
            (mask_choice == 2) & is_protein & is_complex.unsqueeze(-1),
            ~aa_mask,
            clean_mask,
        )

        if batched_data["is_complex"].any():
            # mode 3:
            clean_mask = torch.where(
                (mask_choice == 3) & is_protein & is_complex.unsqueeze(-1),
                True,
                clean_mask,
            )
            if time_step is not None:
                time_step = torch.where(
                    (mask_choice == 3) & is_protein & is_complex.unsqueeze(-1),
                    time_protein,
                    time_step,
                )
            elif noise_step is not None:
                noise_step = torch.where(
                    (mask_choice == 3) & is_protein & is_complex.unsqueeze(-1),
                    noise_step_protein,
                    noise_step,
                )

        # set padding mask to clean
        clean_mask = clean_mask.masked_fill(padding_mask, True)
        clean_mask = clean_mask.masked_fill(
            is_seq_only.unsqueeze(-1),
            False if self.psm_config.mlm_from_decoder_feature else True,
        )
        # set special token "<.>" to clean
        token_id = batched_data["token_id"]
        clean_mask = clean_mask.masked_fill(token_id == 156, True)

        if time_step is not None:
            time_step = time_step.masked_fill(token_id == 156, 0.0)
            # set T noise if protein is seq only
            time_step = time_step.masked_fill(is_seq_only.unsqueeze(-1), 1.0)
            # set 0 noise for padding
            time_step = time_step.masked_fill(padding_mask, 0.0)
            # # TODO: found this may cause instability issue, need to check
            # # # set T noise for batched_data["protein_mask"] nan/inf coords
            time_step = time_step.masked_fill(
                batched_data["protein_mask"].any(dim=-1), 1.0
            )

        if noise_step is not None:
            noise_step = noise_step.masked_fill(token_id == 156, -4.42)
            # set T noise if protein is seq only
            noise_step = noise_step.masked_fill(
                is_seq_only.unsqueeze(-1), 4.19
            )  # NOTE: 3σ is used as the maximum value
            # set 0 noise for padding
            noise_step = noise_step.masked_fill(padding_mask, -4.42)
            # # TODO: found this may cause instability issue, need to check
            # # # set T noise for batched_data["protein_mask"] nan/inf coords
            noise_step = noise_step.masked_fill(
                batched_data["protein_mask"].any(dim=-1), 4.19
            )

        # make sure noise really replaces nan/inf coords
        clean_mask = clean_mask.masked_fill(
            batched_data["protein_mask"].any(dim=-1), False
        )

        if time_step is not None:
            time_step = time_step.masked_fill(clean_mask, 0.0)

        if noise_step is not None:
            noise_step = noise_step.masked_fill(clean_mask, -4.42)

        return clean_mask, aa_mask, time_step, noise_step

    def _protea_pretrain_mode(
        self,
        clean_mask,
        aa_mask,
        padding_mask,
        is_protein,
        is_seq_only,
        is_complex,
        time_step,
        time_step_1d,
        noise_step,
        batched_data,
    ):
        """
        For protein pretrain mode, we have 3 modes:
        0: 50% masked seq and 50% noised structure to structure and seq
        1: clean seq to structure
        2: 50% masked seq to structure and masked seq
        For Complex pretrain mode, we have 3 modes:
        0: clean protein seq to protein structure and molecule structure, time_protein == time_ligand
        1: clean protein seq to protein structure and molecule structure, time_protein < time_ligand
        2: 50% masked protein seq and 50% noised structure to protein structure and seq, molecule all clean
        3: clean protein seq and structure to ligand structure

        """
        n_graph, nnodes = aa_mask.size()[:2]
        clean_mask = clean_mask.unsqueeze(-1).repeat(1, nnodes)

        time_step = time_step.unsqueeze(-1).repeat(1, nnodes)
        time_step_1d = time_step_1d.unsqueeze(-1).repeat(1, nnodes)

        # set mask_aa to all True tensor
        aa_mask = torch.ones_like(aa_mask, dtype=torch.bool)

        # set padding mask to clean
        clean_mask = clean_mask.masked_fill(padding_mask, True)
        aa_mask = aa_mask.masked_fill(padding_mask, False)
        clean_mask = clean_mask.masked_fill(
            is_seq_only.unsqueeze(-1),
            False if self.psm_config.mlm_from_decoder_feature else True,
        )

        # set special token "<.>" to clean
        token_id = batched_data["token_id"]
        clean_mask = clean_mask.masked_fill(token_id == 156, True)
        aa_mask = aa_mask.masked_fill(token_id == 156, False)

        time_step = time_step.masked_fill(token_id == 156, 0.0)
        time_step_1d = time_step_1d.masked_fill(token_id == 156, 0.0)
        # set T noise if protein is seq only
        time_step = time_step.masked_fill(is_seq_only.unsqueeze(-1), 1.0)
        time_step_1d = time_step_1d.masked_fill(is_seq_only.unsqueeze(-1), 1.0)
        # set 0 noise for padding
        time_step = time_step.masked_fill(padding_mask, 0.0)
        time_step_1d = time_step_1d.masked_fill(padding_mask, 0.0)
        # # TODO: found this may cause instability issue, need to check
        # # # set T noise for batched_data["protein_mask"] nan/inf coords
        time_step = time_step.masked_fill(batched_data["protein_mask"].any(dim=-1), 1.0)

        # make sure noise really replaces nan/inf coords
        clean_mask = clean_mask.masked_fill(
            batched_data["protein_mask"].any(dim=-1), False
        )
        time_step = time_step.masked_fill(clean_mask, 0.0)

        return clean_mask, aa_mask, time_step, time_step_1d, noise_step

    def _create_system_tags(self, batched_data):
        token_id = batched_data["token_id"]
        sample_type = batched_data["sample_type"]
        is_periodic = batched_data["pbc"].any(dim=-1)
        is_complex = sample_type == 6
        is_molecule = (~is_periodic) & (token_id <= 129).all(dim=-1) & (~is_complex)
        is_protein = (~is_periodic.unsqueeze(-1)) & (token_id > 129) & (token_id < 156)
        # is_heavy_atom = is_molecule & (token_id > 37).any(dim=-1)
        is_heavy_atom = is_molecule & (token_id > 130).any(dim=-1)

        is_seq_only = sample_type == 5
        is_seq_only = is_seq_only | batched_data["protein_mask"].all(dim=(-1, -2))

        is_energy_outlier = is_molecule & (
            torch.abs(batched_data["energy_per_atom"]) > 23
        )
        is_force_outlier = is_molecule & (
            batched_data["forces"].norm(dim=-1).mean(dim=-1) > 2.5
        )

        batched_data["is_periodic"] = is_periodic
        batched_data["is_molecule"] = is_molecule
        batched_data["is_protein"] = is_protein
        batched_data["is_heavy_atom"] = (
            is_energy_outlier | is_heavy_atom | is_force_outlier
        )
        batched_data["is_seq_only"] = is_seq_only
        batched_data["is_complex"] = is_complex

        # atom mask to leave out unit cell corners for periodic systems
        pos = batched_data["pos"]
        n_graphs, n_nodes = pos.size()[:2]

        # create non_atom_mask to mask out unit cell corners for pbc only
        non_atom_mask = torch.arange(
            n_nodes, dtype=torch.long, device=pos.device
        ).unsqueeze(0).repeat(n_graphs, 1) >= batched_data["num_atoms"].unsqueeze(-1)
        batched_data["non_atom_mask"] = non_atom_mask

        # create diff loss mask so that only diffusion loss of 4 out of 8 cell corners are calculated
        diff_loss_mask = torch.arange(
            n_nodes, dtype=torch.long, device=pos.device
        ).unsqueeze(0).repeat(n_graphs, 1) < batched_data["num_atoms"].unsqueeze(-1)
        is_stable_periodic = batched_data["is_stable_periodic"]
        stable_periodic_index = torch.nonzero(is_stable_periodic)[:, 0]
        diff_loss_mask[
            stable_periodic_index, batched_data["num_atoms"][is_stable_periodic]
        ] = True
        diff_loss_mask[
            stable_periodic_index, batched_data["num_atoms"][is_stable_periodic] + 1
        ] = True
        diff_loss_mask[
            stable_periodic_index, batched_data["num_atoms"][is_stable_periodic] + 2
        ] = True
        diff_loss_mask[
            stable_periodic_index, batched_data["num_atoms"][is_stable_periodic] + 4
        ] = True
        batched_data["diff_loss_mask"] = diff_loss_mask

    def _set_noise(
        self,
        padding_mask,
        batched_data,
        ori_angle=None,
        mask_pos=None,
        mask_angle=None,
        mode_mask=None,
        noise_step=None,
        time_step=None,
        time_step_1d=None,
        clean_mask=None,
        clean_mask_1d=None,
        infer=False,
        aa_mask=None,
    ):
        """
        set diffusion noise here
        """

        ori_pos = center_pos(batched_data, padding_mask).float()
        ori_pos = ori_pos.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        self._create_initial_pos_for_diffusion(batched_data)

        if (
            self.args.backbone
            in ["vanillatransformer", "dit", "e2dit", "ditp", "exp", "exp2", "exp3"]
            and not self.disable_data_aug
            and not batched_data["is_periodic"].any()  # do not rotate pbc material
        ):
            R = uniform_random_rotation(
                ori_pos.size(0), device=ori_pos.device, dtype=ori_pos.dtype
            )
            T = torch.randn(
                ori_pos.size(0), 3, device=ori_pos.device, dtype=ori_pos.dtype
            ).unsqueeze(1)
            ori_pos = torch.bmm(ori_pos, R) + T
            batched_data["forces"] = torch.bmm(batched_data["forces"].float(), R)
            # batched_data["init_pos"] = torch.bmm(batched_data["init_pos"], R)
            # batched_data["cell"] = torch.bmm(batched_data["cell"], R)

        ori_pos = ori_pos / self.psm_config.diffusion_rescale_coeff
        batched_data["ori_pos"] = ori_pos

        if self.psm_config.diffusion_mode == "edm":
            (
                noise_pos,
                noise,
                sigma_edm,
                weight_edm,
            ) = self.diffnoise.noise_sample(
                x_start=ori_pos,
                noise_step=noise_step,
                non_atom_mask=batched_data["non_atom_mask"],
                is_stable_periodic=batched_data["is_stable_periodic"],
                x_init=batched_data["init_pos"],
                clean_mask=clean_mask,
            )
            sigma = sigma_edm
            alpha = None
            weight = weight_edm
        elif self.psm_config.diffusion_mode == "protea":
            (
                noise_pos,
                noise,
                sqrt_one_minus_alphas_cumprod_t,
                sqrt_alphas_cumprod_t,
            ) = self.diffnoise.noise_sample(
                x_start=ori_pos,
                t=time_step,
                non_atom_mask=batched_data["non_atom_mask"],
                is_stable_periodic=batched_data["is_stable_periodic"],
                x_init=batched_data["init_pos"],
                clean_mask=clean_mask,
            )

            # set convert token_id to one_hot_token_id
            batched_data["one_hot_token_id"] = F.one_hot(
                batched_data["token_id"], num_classes=160
            ).float()

            (
                batched_data["one_hot_token_id"],
                noise_1d,
                sqrt_one_minus_alphas_cumprod_t_1d,
                sqrt_alphas_cumprod_t_1d,
            ) = self.diffnoise.noise_sample(
                x_start=batched_data["one_hot_token_id"],
                t=time_step_1d,
                non_atom_mask=batched_data["non_atom_mask"],
                is_stable_periodic=batched_data["is_stable_periodic"],
                clean_mask=~aa_mask,
            )

            sigma = sqrt_one_minus_alphas_cumprod_t
            alpha = sqrt_alphas_cumprod_t
            batched_data["sigma_1d"] = sqrt_one_minus_alphas_cumprod_t_1d
            batched_data["alpha_1d"] = sqrt_alphas_cumprod_t_1d
            batched_data["noise_1d"] = noise_1d
            weight = None
        else:
            (
                noise_pos,
                noise,
                sqrt_one_minus_alphas_cumprod_t,
                sqrt_alphas_cumprod_t,
            ) = self.diffnoise.noise_sample(
                x_start=ori_pos,
                t=time_step,
                non_atom_mask=batched_data["non_atom_mask"],
                is_stable_periodic=batched_data["is_stable_periodic"],
                x_init=batched_data["init_pos"],
                clean_mask=clean_mask,
            )
            sigma = sqrt_one_minus_alphas_cumprod_t
            alpha = sqrt_alphas_cumprod_t
            weight = None

        noise_pos = complete_cell(noise_pos, batched_data)

        # return noise_pos, noise, sqrt_one_minus_alphas_cumprod_t, sqrt_alphas_cumprod_t
        return noise_pos, noise, sigma, alpha, weight

    def load_pretrained_weights(self, args, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.

        Args:
            args: Command line arguments.
            checkpoint_path: Path to the pretrained weights.
        """
        checkpoints_state = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoints_state:
            checkpoints_state = checkpoints_state["model"]
        elif "module" in checkpoints_state:
            checkpoints_state = checkpoints_state["module"]

        for key in list(checkpoints_state.keys()):
            if key.startswith("base."):
                checkpoints_state[key[5:]] = checkpoints_state.pop(key)

        IncompatibleKeys = self.load_state_dict(checkpoints_state, strict=False)
        IncompatibleKeys = IncompatibleKeys._asdict()

        missing_keys = []
        for keys in IncompatibleKeys["missing_keys"]:
            if keys.find("dummy") == -1:
                missing_keys.append(keys)

        unexpected_keys = []
        for keys in IncompatibleKeys["unexpected_keys"]:
            if keys.find("dummy") == -1:
                unexpected_keys.append(keys)

        if len(missing_keys) > 0:
            logger.info(
                "Missing keys in {}: {}".format(
                    checkpoint_path,
                    missing_keys,
                )
            )

        if len(unexpected_keys) > 0:
            logger.info(
                "Unexpected keys {}: {}".format(
                    checkpoint_path,
                    unexpected_keys,
                )
            )

        logger.info(f"checkpoint: {checkpoint_path} is loaded")

    def max_positions(self):
        """
        Returns the maximum positions of the net.
        """
        return self.net.max_positions

    def sample_and_calc_match_metric(self, batched_data):
        match_results = {}
        self.net.eval()
        for sample_time_index in range(self.psm_config.num_sampling_time):
            original_pos = batched_data["pos"].clone()
            original_cell = batched_data["cell"].clone()
            if not self.psm_config.sample_ligand_only:
                batched_data["pos"] = torch.zeros_like(
                    batched_data["pos"]
                )  # zero position to avoid any potential leakage
            batched_data["cell"] = torch.zeros_like(batched_data["cell"])
            if (
                self.psm_config.diffusion_mode == "edm"
                and self.psm_config.diffusion_sampling == "edm"
            ):
                # if self.psm_config.edm_sampling_method == "af3":
                self.sample_AF3(batched_data=batched_data)
                # elif self.psm_config.edm_sampling_method == "2nd":
                #     pass
            else:
                self.sample(batched_data=batched_data)

            match_result_one_time = self.sampled_structure_converter.convert_and_match(
                batched_data, original_pos, sample_time_index
            )
            for match_result in match_result_one_time:
                for key in match_result:
                    if key not in match_results:
                        match_results[key] = []
                    match_results[key].append(match_result[key])
            batched_data[
                "pos"
            ] = original_pos  # recover original position, in case that we want to calculate diffusion loss and sampling RMSD at the same time in validation, and for subsequent sampling
            batched_data["cell"] = original_cell
        for key in match_results:
            match_results[key] = torch.tensor(
                match_results[key], device=batched_data["pos"].device
            )
            match_results[key] = (
                match_results[key].view(self.psm_config.num_sampling_time, -1).T
            )
        self.net.train()
        return match_results

    def _pre_forward_operation(self, batched_data):
        """
        Pre-forward operation for the model.

        Args:
            batched_data: Input data for the forward pass.
        """

        self._create_protein_mask(batched_data)
        self._create_system_tags(batched_data)
        pos = batched_data["pos"]

        n_graphs = pos.size(0)

        if self.psm_config.diffusion_mode == "edm":
            time_step = None
            time_step_1d = None
            noise_step, clean_mask = self.time_step_sampler.sample(
                n_graphs, pos.device, pos.dtype, self.psm_config.clean_sample_ratio
            )
        elif self.psm_config.diffusion_mode == "protea":
            noise_step = None
            time_step, clean_mask = self.time_step_sampler.sample(
                n_graphs, pos.device, pos.dtype, self.psm_config.clean_sample_ratio
            )
            time_step_1d, _ = self.time_step_sampler.sample(
                n_graphs, pos.device, pos.dtype, self.psm_config.clean_sample_ratio
            )
        else:
            noise_step = None
            time_step_1d = None
            time_step, clean_mask = self.time_step_sampler.sample(
                n_graphs, pos.device, pos.dtype, self.psm_config.clean_sample_ratio
            )

        clean_mask = clean_mask & ~(
            batched_data["is_protein"].any(dim=-1) | batched_data["is_complex"]
        )  # Proteins are always corrupted. For proteins, we only consider diffusion training on structure for now.

        # Do not predict energy of molecule with heavy atom avoid loss instability.
        if self.psm_config.clean_sample_ratio < 1.0:
            clean_mask = clean_mask & ~batched_data["is_heavy_atom"]

        clean_mask = clean_mask | (
            (batched_data["is_periodic"]) & (~batched_data["is_stable_periodic"])
        )  # A periodic sample which is not stable is always clean

        clean_mask = clean_mask & (
            (~batched_data["is_periodic"]) | (~batched_data["is_stable_periodic"])
        )  # A periodic sample which is stable is always corrupted

        token_id = batched_data["token_id"]
        padding_mask = token_id.eq(0)  # B x T x 1
        aa_mask = batched_data["protein_masked_aa"] & batched_data["is_protein"]
        aa_mask = aa_mask & ~padding_mask

        if self.psm_config.diffusion_mode == "protea":
            (
                clean_mask,
                aa_mask,
                time_step,
                time_step_1d,
                noise_step,
            ) = self._protea_pretrain_mode(
                clean_mask,
                aa_mask,
                padding_mask,
                batched_data["is_protein"],
                batched_data["is_seq_only"],
                batched_data["is_complex"],
                time_step,
                time_step_1d,
                noise_step,
                batched_data,
            )
        else:
            clean_mask, aa_mask, time_step, noise_step = self._protein_pretrain_mode(
                clean_mask,
                aa_mask,
                padding_mask,
                batched_data["is_protein"],
                batched_data["is_seq_only"],
                batched_data["is_complex"],
                time_step,
                noise_step,
                batched_data,
            )

        if self.psm_config.psm_finetune_mode:
            if self.training:
                noise_mode = self.psm_config.psm_finetune_noise_mode
            else:
                noise_mode = self.psm_config.psm_finetune_valid_noise_mode

            if noise_mode == "T":
                time_step = torch.ones_like(time_step)
                clean_mask = torch.zeros_like(clean_mask)
                # batched_data["pos"] = torch.zeros_like(batched_data["pos"])
            elif noise_mode == "zero":
                time_step = torch.zeros_like(time_step)
                clean_mask = torch.ones_like(clean_mask)
            elif noise_mode == "T_zero":
                # 50% zero, 50% T, set clean_mask=True to 0
                time_step = torch.ones_like(time_step)
                time_step = time_step.masked_fill(clean_mask, 0.0)
                batched_data["pos"] = batched_data["pos"].masked_fill(
                    ~clean_mask.unsqueeze(-1), 0.0
                )
            elif noise_mode == "T_Diff":
                # 50% diffusion, 50% T, set clean_mask=True to T
                time_step = time_step.masked_fill(clean_mask, 1.0)
                batched_data["pos"] = batched_data["pos"].masked_fill(
                    clean_mask.unsqueeze(-1), 0.0
                )
                clean_mask = torch.zeros_like(clean_mask)
            else:
                assert noise_mode == "diffusion"

        (
            pos,
            noise,
            sigma,
            alpha,
            weight,
        ) = self._set_noise(
            padding_mask=padding_mask,
            batched_data=batched_data,
            time_step=time_step,
            time_step_1d=time_step_1d,
            noise_step=noise_step,
            clean_mask=clean_mask,
            aa_mask=aa_mask,
        )
        batched_data["pos"] = pos

        if self.psm_config.diffusion_mode == "edm":
            batched_data["sigma_edm"] = sigma
            batched_data["sqrt_one_minus_alphas_cumprod_t"] = None
            batched_data["sqrt_alphas_cumprod_t"] = None
        elif self.psm_config.diffusion_mode == "protea":
            batched_data["sigma_edm"] = None
            batched_data["sqrt_one_minus_alphas_cumprod_t"] = sigma
            batched_data["sqrt_alphas_cumprod_t"] = alpha
        else:
            batched_data["sigma_edm"] = None
            batched_data["sqrt_one_minus_alphas_cumprod_t"] = sigma
            batched_data["sqrt_alphas_cumprod_t"] = alpha

        if self.psm_config.diffusion_mode == "edm":
            c_skip, c_out, c_in, c_noise = self.diffnoise.precondition(sigma)
        else:
            c_skip, c_out, c_in, c_noise = None, None, None, None

        batched_data["c_skip"] = c_skip
        batched_data["c_out"] = c_out
        batched_data["c_in"] = c_in
        batched_data["c_noise"] = c_noise
        batched_data["weight"] = weight

        return (
            clean_mask,
            aa_mask,
            time_step,
            time_step_1d,
            noise_step,
            noise,
            padding_mask,
        )

    def forward(self, batched_data, skip_sample=False, **kwargs):
        """
        Forward pass of the model.

        Args:
            batched_data: Input data for the forward pass.
            **kwargs: Additional keyword arguments.
        """

        if (
            self.psm_config.sample_in_validation
            and not self.training
            and not skip_sample
        ):
            match_results = self.sample_and_calc_match_metric(batched_data)

        (
            clean_mask,
            aa_mask,
            time_step,
            time_step_1d,
            noise_step,
            noise,
            padding_mask,
        ) = self._pre_forward_operation(batched_data)

        if self.psm_config.psm_sample_structure_in_finetune:
            self.net.eval()

        context = torch.no_grad() if self.psm_config.freeze_backbone else nullcontext()
        with context:
            result_dict = self.net(
                batched_data,
                time_step=time_step,
                time_step_1d=time_step_1d,
                clean_mask=clean_mask,
                aa_mask=aa_mask,
                **kwargs,
            )

        result_dict["data_name"] = (
            batched_data["data_name"] if "data_name" in batched_data else None
        )
        result_dict["noise"] = noise
        result_dict["clean_mask"] = clean_mask
        result_dict["aa_mask"] = aa_mask
        result_dict["diff_loss_mask"] = batched_data["diff_loss_mask"]
        result_dict["ori_pos"] = batched_data["ori_pos"]
        result_dict["force_label"] = batched_data["forces"]
        result_dict["padding_mask"] = padding_mask

        if self.psm_config.diffusion_mode == "edm":
            result_dict["noise_step"] = noise_step
            result_dict["weight_edm"] = batched_data["weight"]
            result_dict["time_step"] = None
            result_dict["sqrt_alphas_cumprod_t"] = None
            result_dict["sqrt_one_minus_alphas_cumprod_t"] = None
        else:
            result_dict["noise_step"] = None
            result_dict["weight_edm"] = None
            result_dict["time_step"] = time_step
            result_dict["sqrt_alphas_cumprod_t"] = batched_data["sqrt_alphas_cumprod_t"]
            result_dict["sqrt_one_minus_alphas_cumprod_t"] = batched_data[
                "sqrt_one_minus_alphas_cumprod_t"
            ]

        if (
            self.psm_config.sample_in_validation
            and not self.training
            and not skip_sample
        ):
            result_dict.update(match_results)

        if self.psm_finetune_head:
            result_dict = self.psm_finetune_head(result_dict)
            if self.psm_config.psm_sample_structure_in_finetune:
                self.eval()
                sampled_output = self.sample(batched_data)
                for k, v in sampled_output.items():
                    result_dict[k + "_sample"] = v
                self.train()

        return result_dict

    def compute_loss(self, model_output, batched_data) -> ModelOutput:
        """
        Compute loss for the model.

        Args:
            model_output: The output from the model.
            batched_data: The batch data.

        Returns:
            ModelOutput: The model output which includes loss, log_output, num_examples.
        """
        bs = batched_data["pos"].size(0)
        loss, logging_output = self.loss_fn(model_output, batched_data)
        if self.psm_finetune_head and hasattr(self.psm_finetune_head, "update_loss"):
            loss, logging_output = self.psm_finetune_head.update_loss(
                loss, logging_output, model_output, batched_data
            )
        return ModelOutput(loss=loss, num_examples=bs, log_output=logging_output)

    def config_optimizer(self, model: nn.Module = None):
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            tuple[Optimizer, LRScheduler]:
        """
        return (None, None)

    @torch.no_grad()
    def sample(
        self,
        batched_data,
        perturb=None,
        time_step=None,
        mask_aa=None,
        mask_pos=None,
        mask_angle=None,
        padding_mask=None,
        mode_mask=None,
        time_pos=None,
        time_aa=None,
        segment_labels=None,
        masked_tokens=None,
        **unused,
    ):
        """
        Sample method for diffussion model
        """
        if "ori_pos" in batched_data:
            batched_data["pos"] = batched_data["ori_pos"]

        self._create_protein_mask(batched_data)
        self._create_system_tags(batched_data)

        device = batched_data["pos"].device

        n_graphs = batched_data["pos"].shape[0]

        token_id = batched_data["token_id"]
        padding_mask = token_id.eq(0)  # B x T x 1
        orig_pos = center_pos(batched_data, padding_mask)

        self._create_initial_pos_for_diffusion(batched_data)

        clean_mask = torch.zeros_like(token_id, dtype=torch.bool, device=device)
        if self.psm_config.sample_ligand_only:
            clean_mask = batched_data["is_protein"]

        clean_mask = clean_mask.masked_fill(token_id == 156, True)
        clean_mask = clean_mask.masked_fill(padding_mask, True)
        clean_mask = clean_mask.masked_fill(
            batched_data["protein_mask"].any(dim=-1), False
        )

        batched_data["pos"] = self.diffnoise.get_sampling_start(
            batched_data["init_pos"],
            batched_data["non_atom_mask"],
            batched_data["is_stable_periodic"],
        )

        if clean_mask is not None:
            batched_data["pos"] = torch.where(
                clean_mask.unsqueeze(-1), orig_pos, batched_data["pos"]
            )

        batched_data["pos"] = complete_cell(batched_data["pos"], batched_data)

        if self.args.backbone in ["dit", "ditp", "exp", "exp2", "exp3"]:
            if_recenter = False
        else:
            if_recenter = True

        if if_recenter:
            batched_data["pos"] = center_pos(
                batched_data, padding_mask=padding_mask, clean_mask=clean_mask
            )  # centering to remove noise translation

        decoder_x_output = None
        for t in range(
            self.psm_config.num_timesteps - 1,
            -1,
            self.psm_config.num_timesteps_stepsize,
        ):
            # forward
            if self.psm_config.diffusion_mode == "edm":
                time_step = None
            else:
                time_step = self.time_step_sampler.get_continuous_time_step(
                    t, n_graphs, device=device, dtype=batched_data["pos"].dtype
                )
                time_step = time_step.unsqueeze(-1).repeat(
                    1, batched_data["pos"].shape[1]
                )
                if clean_mask is not None:
                    time_step = time_step.masked_fill(clean_mask, 0.0)

            x_t = batched_data["pos"].clone()  # to avoid in-place operation in edm
            if self.psm_config.diffusion_mode == "edm":
                t_hat = self.diffusion_process.t_to_sigma(t)
                # Reshape sigma to (B, L, 1)
                t_hat = t_hat.unsqueeze(-1).repeat(1, x_t.shape[1]).unsqueeze(-1)
                t_hat = t_hat.double()
                c_skip, c_out, c_in, c_noise = self.diffnoise.precondition(t_hat)
                batched_data["edm_sigma"] = t_hat
                batched_data["c_skip"] = c_skip
                batched_data["c_out"] = c_out
                batched_data["c_in"] = c_in
                batched_data["c_noise"] = c_noise
            else:
                batched_data[
                    "sqrt_one_minus_alphas_cumprod_t"
                ] = self.diffnoise._extract(
                    self.diffnoise.sqrt_one_minus_alphas_cumprod,
                    (time_step * self.psm_config.num_timesteps).long(),
                    batched_data["pos"].shape,
                )

            net_result = self.net(
                batched_data,
                time_step=time_step,
                clean_mask=clean_mask,
                padding_mask=padding_mask,
            )

            predicted_noise = net_result["noise_pred"]
            if self.psm_config.psm_finetune_mode:
                decoder_x_output = net_result["decoder_x_output"]
            epsilon = self.diffnoise.get_noise(
                batched_data["pos"],
                batched_data["non_atom_mask"],
                batched_data["is_stable_periodic"],
            )

            batched_data["pos"] = self.diffusion_process.sample_step(
                x_t,
                batched_data["init_pos"],
                predicted_noise,
                epsilon,
                t,
                stepsize=-self.psm_config.num_timesteps_stepsize,
            )

            if clean_mask is not None:
                batched_data["pos"] = torch.where(
                    clean_mask.unsqueeze(-1), orig_pos, batched_data["pos"]
                )

            batched_data["pos"] = complete_cell(batched_data["pos"], batched_data)
            if if_recenter:
                batched_data["pos"] = center_pos(
                    batched_data, padding_mask=padding_mask, clean_mask=clean_mask
                )  # centering to remove noise translation

            batched_data["pos"] = batched_data["pos"].detach()

        batched_data["pos"] = (
            batched_data["pos"] * self.psm_config.diffusion_rescale_coeff
        )
        pred_pos = batched_data["pos"].clone()

        if (
            self.psm_config.psm_finetune_mode
            and self.psm_finetune_head.__class__.__name__ == "PerResidueLDDTCaPredictor"
        ):
            logger.info("Running PerResidueLDDTCaPredictor")
            plddt = self.psm_finetune_head(
                {
                    "decoder_x_output": decoder_x_output,
                    "is_protein": batched_data["is_protein"],
                }
            )
            plddt_residue = plddt["plddt"]
            mean_plddt = plddt["mean_plddt"]
            plddt_per_prot = (plddt_residue * batched_data["is_protein"]).sum(
                dim=-1
            ) / (1e-10 + batched_data["is_protein"].sum(dim=-1))
            batched_data["plddt_residue"] = plddt_residue
            batched_data["mean_plddt"] = mean_plddt
            batched_data["plddt_per_prot"] = plddt_per_prot

        loss = torch.sum((pred_pos - orig_pos) ** 2, dim=-1, keepdim=True)

        return {
            "loss": loss,
            "pred_pos": pred_pos,
            "orig_pos": orig_pos,
        }

    @torch.no_grad()
    def sample_AF3(
        self,
        batched_data,
        perturb=None,
        time_step=None,
        mask_aa=None,
        mask_pos=None,
        mask_angle=None,
        padding_mask=None,
        mode_mask=None,
        time_pos=None,
        time_aa=None,
        segment_labels=None,
        masked_tokens=None,
        **unused,
    ):
        """
        Sample method in AF3 for EDM Model
        """
        if "ori_pos" in batched_data:
            batched_data["pos"] = batched_data["ori_pos"]

        self._create_protein_mask(batched_data)
        self._create_system_tags(batched_data)

        device = batched_data["pos"].device

        n_graphs = batched_data["pos"].shape[0]

        token_id = batched_data["token_id"]
        padding_mask = token_id.eq(0)  # B x T x 1

        orig_pos = center_pos(batched_data, padding_mask)

        self._create_initial_pos_for_diffusion(batched_data)

        clean_mask = torch.zeros_like(token_id, dtype=torch.bool, device=device)
        if self.psm_config.sample_ligand_only:
            clean_mask = batched_data["is_protein"]

        clean_mask = clean_mask.masked_fill(token_id == 156, True)
        clean_mask = clean_mask.masked_fill(padding_mask, True)
        clean_mask = clean_mask.masked_fill(
            batched_data["protein_mask"].any(dim=-1), False
        )

        batched_data["pos"] = self.diffnoise.get_sampling_start(
            batched_data["init_pos"],
            batched_data["non_atom_mask"],
            batched_data["is_stable_periodic"],
        )

        if clean_mask is not None:
            batched_data["pos"] = torch.where(
                clean_mask.unsqueeze(-1), orig_pos, batched_data["pos"]
            )

        batched_data["pos"] = complete_cell(batched_data["pos"], batched_data)

        if self.args.backbone in ["dit", "ditp", "exp", "exp2", "exp3"]:
            if_recenter = False
        else:
            if_recenter = True

        if if_recenter:
            batched_data["pos"] = center_pos(
                batched_data, padding_mask=padding_mask, clean_mask=clean_mask
            )  # centering to remove noise translation

        # AF3 Sampling
        num_steps = self.psm_config.edm_sample_num_steps
        rho = self.psm_config.edm_sample_rho
        inv_rho = 1.0 / rho
        sigma_min = self.psm_config.edm_sample_sigma_min
        sigma_max = self.psm_config.edm_sample_sigma_max
        gamma_0 = self.psm_config.af3_sample_gamma_0
        gamma_min = self.psm_config.af3_sample_gamma_min
        noise_scale = self.psm_config.diffusion_noise_std
        step_scale = self.psm_config.af3_sample_step_scale
        sigma_data = self.psm_config.edm_sigma_data

        # Time step discretization.
        step_indices = torch.arange(
            num_steps, dtype=batched_data["pos"].dtype, device=device
        )

        t_steps = (
            sigma_data
            * (
                sigma_max**inv_rho
                + step_indices
                / (num_steps - 1)
                * (sigma_min**inv_rho - sigma_max**inv_rho)
            )
            ** rho
            * torch.ones((n_graphs,), device=device)
        )  # shape is (B, )

        decoder_x_output = None

        for i, (t_prev, t_cur) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 1, ..., N
            # batched_data["pos"] = torch.where(
            #     clean_mask.unsqueeze(-1), orig_pos, batched_data["pos"]
            # )
            x_cur = batched_data["pos"].clone()

            gamma = gamma_0 if t_cur > gamma_min else 0.0

            # clean mask
            t_prev = t_prev.unsqueeze(-1).repeat(1, x_cur.shape[1])
            t_cur = t_cur.unsqueeze(-1).repeat(1, x_cur.shape[1])
            if clean_mask is not None:
                t_prev = t_prev.masked_fill(clean_mask, 0.0064)
                t_cur = t_cur.masked_fill(clean_mask, 0.0064)

            # # # Data Augmentation
            R = uniform_random_rotation(
                x_cur.size(0), device=x_cur.device, dtype=x_cur.dtype
            )
            x_cur = torch.bmm(x_cur, R)

            # Reshape sigma to (B, L, 1)
            t_prev = t_prev.unsqueeze(-1)
            t_cur = t_cur.unsqueeze(-1)
            t_hat = (1.0 + gamma) * t_prev

            if clean_mask is not None:
                t_hat = t_hat.masked_fill(clean_mask.unsqueeze(-1), 0.0064)

            # Euler step.
            ksi = (
                noise_scale
                * (t_hat**2 - t_prev**2).sqrt()
                * torch.randn_like(x_cur)
            )
            x_noisy = x_cur + ksi
            c_skip, c_out, c_in, c_noise = self.diffnoise.precondition(t_hat)
            batched_data["edm_sigma"] = t_hat
            batched_data["pos"] = x_noisy
            batched_data["c_skip"] = c_skip
            batched_data["c_out"] = c_out
            batched_data["c_in"] = c_in
            batched_data["c_noise"] = c_noise

            net_result = self.net(
                batched_data,
                time_step=None,
                clean_mask=clean_mask,
                padding_mask=padding_mask,
            )
            x0_pred = net_result["noise_pred"]
            if self.psm_config.psm_finetune_mode:
                decoder_x_output = net_result["decoder_x_output"]
            delta = (x_noisy - x0_pred) / t_hat
            dt = t_cur - t_hat
            x_next = x_noisy + dt * delta * step_scale
            batched_data["pos"] = x_next

            if clean_mask is not None:
                batched_data["pos"] = torch.where(
                    clean_mask.unsqueeze(-1), x_cur, batched_data["pos"]
                )

            batched_data["pos"] = complete_cell(batched_data["pos"], batched_data)
            if if_recenter:
                batched_data["pos"] = center_pos(
                    batched_data, padding_mask=padding_mask, clean_mask=clean_mask
                )  # centering to remove noise translation

            batched_data["pos"] = batched_data["pos"].detach()

        pred_pos = batched_data["pos"].clone()
        if (
            self.psm_config.psm_finetune_mode
            and self.psm_finetune_head.__class__.__name__ == "PerResidueLDDTCaPredictor"
        ):
            logger.info("Running PerResidueLDDTCaPredictor")
            plddt = self.psm_finetune_head(
                {
                    "decoder_x_output": decoder_x_output,
                    "is_protein": batched_data["is_protein"],
                }
            )
            plddt_residue = plddt["plddt"]
            mean_plddt = plddt["mean_plddt"]
            plddt_per_prot = (plddt_residue * batched_data["is_protein"]).sum(
                dim=-1
            ) / (1e-10 + batched_data["is_protein"].sum(dim=-1))
            batched_data["plddt_residue"] = plddt_residue
            batched_data["mean_plddt"] = mean_plddt
            batched_data["plddt_per_prot"] = plddt_per_prot

        loss = torch.sum((pred_pos - orig_pos) ** 2, dim=-1, keepdim=True)

        return {
            "loss": loss,
            "pred_pos": pred_pos,
            "orig_pos": orig_pos,
        }


@torch.compiler.disable(recursive=True)
def center_pos(batched_data, padding_mask, clean_mask=None):
    # get center of system positions
    is_stable_periodic = batched_data["is_stable_periodic"]  # B x 3 -> B
    periodic_center = (
        torch.gather(
            batched_data["pos"][is_stable_periodic],
            index=batched_data["num_atoms"][is_stable_periodic]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, 3),
            dim=1,
        )
        + torch.gather(
            batched_data["pos"][is_stable_periodic],
            index=batched_data["num_atoms"][is_stable_periodic]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, 3)
            + 7,
            dim=1,
        )
    ) / 2.0
    protein_mask = batched_data["protein_mask"] | batched_data["token_id"].eq(
        156
    ).unsqueeze(-1)
    if clean_mask is None:
        num_non_atoms = torch.sum(protein_mask.any(dim=-1), dim=-1)
        non_periodic_center = torch.sum(
            batched_data["pos"].masked_fill(
                padding_mask.unsqueeze(-1) | protein_mask, 0.0
            ),
            dim=1,
        ) / (batched_data["num_atoms"] - num_non_atoms).unsqueeze(-1)
    else:
        # leave out padding tokens when calculating non-atom/non-residue tokens
        num_non_atoms = torch.sum(
            protein_mask.any(dim=-1) | (clean_mask & ~padding_mask), dim=-1
        )
        non_periodic_center = torch.sum(
            batched_data["pos"].masked_fill(
                padding_mask.unsqueeze(-1) | protein_mask | clean_mask.unsqueeze(-1),
                0.0,
            ),
            dim=1,
        ) / (batched_data["num_atoms"] - num_non_atoms).unsqueeze(-1)

    center = non_periodic_center.unsqueeze(1)
    center[is_stable_periodic] = periodic_center
    batched_data["pos"] -= center

    batched_data["pos"] = batched_data["pos"].masked_fill(
        padding_mask.unsqueeze(-1), 0.0
    )
    # TODO: filter nan/inf to zero in coords from pdb data, needs better solution
    batched_data["pos"] = batched_data["pos"].masked_fill(protein_mask, 0.0)
    batched_data["pos"] = batched_data["pos"].masked_fill(
        batched_data["token_id"].eq(156).unsqueeze(-1), 0.0
    )
    return batched_data["pos"]


def complete_cell(pos, batched_data):
    is_stable_periodic = batched_data["is_stable_periodic"]
    periodic_pos = pos[is_stable_periodic]
    device = periodic_pos.device
    dtype = periodic_pos.dtype
    cell_matrix = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=dtype,
        device=device,
    )
    n_graphs = periodic_pos.size()[0]
    gather_index = torch.tensor(
        [0, 4, 2, 1], device=device, dtype=torch.long
    ).unsqueeze(0).unsqueeze(-1).repeat([n_graphs, 1, 3]) + batched_data["num_atoms"][
        is_stable_periodic
    ].unsqueeze(
        -1
    ).unsqueeze(
        -1
    )
    lattice = torch.gather(periodic_pos, 1, index=gather_index)
    corner = lattice[:, 0, :]
    lattice = lattice[:, 1:, :] - corner.unsqueeze(1)
    batched_data["cell"][is_stable_periodic, :, :] = lattice
    cell = torch.matmul(cell_matrix, lattice) + corner.unsqueeze(1)
    scatter_index = torch.arange(8, device=device).unsqueeze(0).unsqueeze(-1).repeat(
        [n_graphs, 1, 3]
    ) + batched_data["num_atoms"][is_stable_periodic].unsqueeze(-1).unsqueeze(-1)
    cell -= ((cell[:, 0, :] + cell[:, 7, :]) / 2.0).unsqueeze(1)
    periodic_pos = periodic_pos.scatter(1, scatter_index, cell)
    pos[is_stable_periodic] = periodic_pos

    token_id = batched_data["token_id"]
    padding_mask = token_id.eq(0)  # B x T x 1
    pos = pos.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    return pos


class PSM(nn.Module):
    """
    Class for training Physics science module
    """

    def __init__(
        self,
        args,
        psm_config: PSMConfig,
        molecule_energy_per_atom_std=1.0,
        periodic_energy_per_atom_std=1.0,
        molecule_force_std=1.0,
        periodic_force_std=1.0,
    ):
        super().__init__()
        self.max_positions = args.max_positions
        self.args = args
        self.backbone = args.backbone

        self.psm_config = psm_config

        self.cell_expander = CellExpander(
            self.psm_config.pbc_cutoff,
            self.psm_config.pbc_expanded_token_cutoff,
            self.psm_config.pbc_expanded_num_cell_per_direction,
            self.psm_config.pbc_multigraph_cutoff,
        )

        # Implement the embedding
        if args.backbone in ["vanillatransformer", "vectorvanillatransformer"]:
            self.embedding = PSMMix3dEmbedding(
                psm_config, use_unified_batch_sampler=args.use_unified_batch_sampler
            )
        elif args.backbone in ["dit", "e2dit", "ditgeom"]:
            if self.psm_config.diffusion_mode == "protea":
                self.embedding = ProteaEmbedding(psm_config)
            else:
                # self.embedding = PSMMix3dDitEmbedding(psm_config)
                self.embedding = PSMLightEmbedding(psm_config)
        elif args.backbone in ["ditp"]:
            self.embedding = PSMLightPEmbedding(psm_config)
        elif args.backbone in ["vanillatransformer_equiv"]:
            self.embedding = PSMMix3DEquivEmbedding(psm_config)
        elif args.backbone in ["exp", "exp2", "exp3"]:
            # self.embedding = PSMSeqEmbedding(psm_config)
            self.embedding = PSMMixSeqEmbedding(psm_config)
        else:
            self.embedding = PSMMixEmbedding(psm_config)

        self.encoder = None
        if args.backbone == "graphormer":
            # Implement the encoder
            self.encoder = PSMEncoder(args, psm_config)
            # Implement the decoder
            self.decoder = EquivariantDecoder(psm_config)
        elif args.backbone == "graphormer-e2":
            # Implement the encoder
            self.encoder = PSMEncoder(args, psm_config)
            # Implement the decoder
            self.decoder = E2former(**args.backbone_config)
        elif args.backbone == "equiformerv2":
            self.decoder = Equiformerv2SO2(**args.backbone_config)
        elif args.backbone == "equiformer":
            self.decoder = Equiformer(**args.backbone_config)
        elif args.backbone == "e2former":
            self.decoder = E2former(**args.backbone_config)
        elif args.backbone == "geomformer":
            self.encoder = None
            # Implement the decoder
            self.decoder = EquivariantDecoder(psm_config)
        elif args.backbone in ["vanillatransformer", "vanillatransformer_equiv"]:
            # Implement the encoder
            self.encoder = PSMPlainEncoder(args, psm_config)
            # Implement the decoder
            # self.decoder = EquivariantDecoder(psm_config)
            self.decoder = NodeTaskHead(psm_config)
            # self.decoder = VectorVanillaTransformer(psm_config)
        elif args.backbone in ["exp"]:
            # Implement the encoder
            self.encoder = PSMPlainEncoder(args, psm_config)
            # Implement the decoder
            self.decoder = DiffusionModule(args, psm_config)
        elif args.backbone in ["exp2"]:
            # Implement the encoder
            self.encoder = PSMPairPlainEncoder(args, psm_config)
            # Implement the decoder
            self.decoder = DiffusionModule2(args, psm_config)
        elif args.backbone in ["exp3"]:
            # Implement the encoder
            self.encoder = PSMPairPlainEncoder(args, psm_config)
            # Implement the decoder
            self.decoder = DiffusionModule3(args, psm_config)
        elif args.backbone in ["vectorvanillatransformer"]:
            self.encoder = None
            self.decoder = VectorVanillaTransformer(psm_config)
        elif args.backbone in ["dit"]:
            # Implement the encoder
            self.encoder = PSMDiTEncoder(args, psm_config)
            # self.decoder = EquivariantDecoder(psm_config)
            self.decoder = None
        elif args.backbone in ["ditp"]:
            self.encoder = PSMPDiTPairEncoder(args, psm_config)
            self.decoder = None
        elif args.backbone in ["ditgeom"]:
            # Implement the encoder
            self.encoder = PSMDiTEncoder(args, psm_config)
            self.decoder = EquivariantDecoder(psm_config)
        elif args.backbone in ["e2dit"]:
            # Implement the encoder
            self.encoder = PSMDiTEncoder(args, psm_config)
            self.decoder = E2former(**args.backbone_config)
            # self.decoder = EquivariantDecoder(psm_config)
        else:
            raise NotImplementedError

        if not (
            self.psm_config.psm_finetune_mode
            and self.psm_config.psm_finetune_skip_ori_head
        ):
            # simple energy, force and noise prediction heads
            self.energy_head = nn.ModuleDict()
            self.forces_head = nn.ModuleDict()

        for key in {"molecule", "periodic", "protein"}:
            if args.backbone in [
                "vanillatransformer",
                "vanillatransformer_equiv",
                "vectorvanillatransformer",
                "exp",
                "exp2",
                "exp3",
            ]:
                self.energy_head.update(
                    {
                        key: nn.Sequential(
                            nn.Linear(
                                psm_config.embedding_dim,
                                psm_config.embedding_dim,
                                bias=True,
                            ),
                            nn.SiLU(),
                            nn.LayerNorm(psm_config.embedding_dim),
                            nn.Linear(psm_config.embedding_dim, 1, bias=True),
                        )
                    }
                )
            elif args.backbone in ["dit", "e2dit", "ditgeom", "ditp"]:
                if args.decoder_feat4energy:
                    self.energy_head.update(
                        {
                            key: nn.Sequential(
                                nn.Linear(
                                    psm_config.embedding_dim,
                                    psm_config.embedding_dim,
                                    bias=True,
                                ),
                                nn.SiLU(),
                                nn.Linear(psm_config.embedding_dim, 1, bias=True),
                            )
                        }
                    )
                else:
                    self.energy_head.update(
                        {key: ScalarGatedOutput(psm_config.embedding_dim)}
                    )
            else:
                self.energy_head.update(
                    {
                        key: nn.Sequential(
                            nn.Linear(
                                psm_config.embedding_dim,
                                psm_config.embedding_dim,
                                bias=True,
                            ),
                            nn.SiLU(),
                            nn.Linear(psm_config.embedding_dim, 1, bias=True),
                        )
                    }
                )

            if args.backbone in [
                "vanillatransformer",
                "vanillatransformer_equiv",
                "vectorvanillatransformer",
            ]:
                if self.psm_config.separate_noise_head:
                    self.molecule_noise_head = VectorOutput(psm_config.embedding_dim)
                    self.periodic_noise_head = VectorOutput(psm_config.embedding_dim)
                    self.protein_noise_head = VectorOutput(psm_config.embedding_dim)
                else:
                    self.noise_head = VectorOutput(psm_config.embedding_dim)
                    self.periodic_noise_head = VectorOutput(psm_config.embedding_dim)
                if self.psm_config.force_head_type == ForceHeadType.LINEAR:
                    self.forces_head.update(
                        {key: nn.Linear(psm_config.embedding_dim, 1, bias=False)}
                    )
                else:
                    self.forces_head.update(
                        {key: ForceVecOutput(psm_config.embedding_dim)}
                    )
            elif args.backbone in ["exp", "exp2", "exp3"]:
                if self.psm_config.separate_noise_head:
                    self.molecule_noise_head = VectorProjOutput(
                        psm_config.embedding_dim
                    )
                    self.periodic_noise_head = VectorProjOutput(
                        psm_config.embedding_dim
                    )
                    self.protein_noise_head = VectorProjOutput(psm_config.embedding_dim)
                else:
                    self.noise_head = VectorProjOutput(psm_config.embedding_dim)
                    self.periodic_noise_head = VectorProjOutput(
                        psm_config.embedding_dim
                    )
                if self.psm_config.force_head_type == ForceHeadType.LINEAR:
                    self.forces_head.update(
                        {key: nn.Linear(psm_config.embedding_dim, 1, bias=False)}
                    )
                else:
                    self.forces_head.update(
                        {key: VectorGatedOutput(psm_config.embedding_dim)}
                    )
            elif args.backbone in ["dit", "e2dit", "ditgeom", "ditp"]:
                if self.psm_config.encoderfeat4noise:
                    if self.psm_config.separate_noise_head:
                        self.molecule_noise_head = VectorGatedOutput(
                            psm_config.embedding_dim
                        )
                        self.periodic_noise_head = VectorGatedOutput(
                            psm_config.embedding_dim
                        )
                        self.protein_noise_head = VectorGatedOutput(
                            psm_config.embedding_dim
                        )
                    else:
                        self.noise_head = VectorGatedOutput(psm_config.embedding_dim)
                        self.periodic_noise_head = VectorGatedOutput(
                            psm_config.embedding_dim
                        )
                else:
                    if self.psm_config.separate_noise_head:
                        self.molecule_noise_head = EquivariantVectorOutput(
                            psm_config.embedding_dim
                        )
                        self.periodic_noise_head = EquivariantVectorOutput(
                            psm_config.embedding_dim
                        )
                        self.protein_noise_head = EquivariantVectorOutput(
                            psm_config.embedding_dim
                        )
                    else:
                        self.noise_head = EquivariantVectorOutput(
                            psm_config.embedding_dim
                        )
                        self.periodic_noise_head = EquivariantVectorOutput(
                            psm_config.embedding_dim
                        )

                if self.psm_config.force_head_type == ForceHeadType.LINEAR:
                    self.forces_head.update(
                        {key: nn.Linear(psm_config.embedding_dim, 1, bias=False)}
                    )
                else:
                    self.forces_head.update(
                        {key: VectorGatedOutput(psm_config.embedding_dim)}
                    )
            else:
                if self.psm_config.separate_noise_head:
                    self.molecule_noise_head = EquivariantVectorOutput(
                        psm_config.embedding_dim
                    )
                    self.periodic_noise_head = EquivariantVectorOutput(
                        psm_config.embedding_dim
                    )
                    self.protein_noise_head = EquivariantVectorOutput(
                        psm_config.embedding_dim
                    )
                else:
                    self.noise_head = EquivariantVectorOutput(psm_config.embedding_dim)
                if self.psm_config.force_head_type == ForceHeadType.LINEAR:
                    self.forces_head.update(
                        {key: nn.Linear(psm_config.embedding_dim, 1, bias=False)}
                    )
                else:
                    self.forces_head.update(
                        {key: EquivariantVectorOutput(psm_config.embedding_dim)}
                    )

            # aa mask predict head
            self.aa_mask_head = nn.Sequential(
                nn.Linear(
                    psm_config.embedding_dim, psm_config.embedding_dim, bias=False
                ),
                nn.SiLU(),
                nn.Linear(psm_config.embedding_dim, 160, bias=False),
            )

            if self.args.AutoGradForce:
                self.autograd_force_head = GradientHead(
                    molecule_energy_per_atom_std=molecule_energy_per_atom_std,
                    periodic_energy_per_atom_std=periodic_energy_per_atom_std,
                    molecule_force_std=molecule_force_std,
                    periodic_force_std=periodic_force_std,
                )

        self.mlp_w = nn.Sequential(
            nn.Linear(psm_config.embedding_dim, psm_config.embedding_dim, bias=False),
            nn.SiLU(),
            nn.Linear(psm_config.embedding_dim, 1, bias=False),
        )

        if self.args.backbone in [
            "vanillatransformer",
            "vanillatransformer_equiv",
            "dit",
            "e2dit",
            "ditgeom",
            "ditp",
            "exp",
            "exp2",
            "exp3",
        ]:
            self.layer_norm = nn.LayerNorm(psm_config.embedding_dim)
            self.layer_norm_vec = nn.LayerNorm(psm_config.embedding_dim)

        # if self.args.AutoGradForce:
        self.autograd_force_head = GradientHead()

        self.num_vocab = max([VOCAB[key] for key in VOCAB]) + 1

        self.dist_head = nn.Sequential(
            nn.SiLU(), nn.Linear(psm_config.encoder_pair_embed_dim, 1, bias=False)
        )

        self.pair_proj = nn.Linear(
            psm_config.embedding_dim, 2 * psm_config.encoder_pair_embed_dim, bias=False
        )

        # self.inv_KbT = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _set_aa_mask(self, batched_data, aa_mask):
        token_id = batched_data["token_id"]
        if aa_mask is not None:
            mask_token_type = token_id.masked_fill(
                aa_mask, 157
            )  # 157 is the mask token
        else:
            mask_token_type = token_id

        batched_data["masked_token_type"] = mask_token_type

    def _create_node_type_edge(self, batched_data):
        masked_token_type = batched_data["masked_token_type"]
        n_node = masked_token_type.size()[-1]
        masked_token_type_i = (
            masked_token_type.unsqueeze(-1).repeat(1, 1, n_node).unsqueeze(-1)
        )
        masked_token_type_j = (
            masked_token_type.unsqueeze(1).repeat(1, n_node, 1).unsqueeze(-1)
        )
        if (
            self.psm_config.node_type_edge_method
            == GaussianFeatureNodeType.NON_EXCHANGABLE
        ):
            node_type_edge = masked_token_type_i * self.num_vocab + masked_token_type_j
        else:
            node_type_edge = torch.cat(
                [masked_token_type_i, masked_token_type_j], dim=-1
            )
        batched_data["node_type_edge"] = node_type_edge

    def forward(
        self,
        batched_data,
        time_step=None,
        time_step_1d=None,
        clean_mask=None,
        aa_mask=None,
        padding_mask=None,
        perturb=None,
        q=None,  # for computing the score model on the q
        q_0=None,
        delta_tq=None,  # for computing the score model on the q at time_pos + delta_tq
        mask_aa=None,
        mask_pos=None,
        mask_angle=None,
        mode_mask=None,
        time_pos=None,
        time_aa=None,
        segment_labels=None,
        masked_tokens=None,
        **unused,
    ):
        """
        Forward pass for PSM. This first computes the token

        Args:
            - batched_data: keys need to be defined in the data module
        Returns:
            - need to be defined
        """

        pos = batched_data["pos"]

        if self.psm_config.diffusion_mode == "edm":
            pos_noised_no_c_in = batched_data["pos"].clone()
            batched_data["pos"] = pos * batched_data["c_in"]

        if self.args.AutoGradForce:
            pos.requires_grad_(True)

        n_graphs, n_nodes = pos.size()[:2]
        is_periodic = batched_data["is_periodic"]
        is_molecule = batched_data["is_molecule"]
        is_protein = batched_data["is_protein"]

        self._set_aa_mask(batched_data, aa_mask)
        self._create_node_type_edge(batched_data)

        skip_decoder = (
            batched_data["is_seq_only"].all()
            and not self.psm_config.mlm_from_decoder_feature
            and self.args.backbone == "graphormer"
        )

        dist_map = None

        # B, L, H is Batch, Length, Hidden
        # token_embedding: B x L x H
        # padding_mask: B x L
        # token_type: B x L  (0 is used for PADDING)
        with (
            torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
            if False  # self.args.fp16
            else nullcontext()
        ):
            if (
                "pbc" in batched_data
                and batched_data["pbc"] is not None
                and torch.any(batched_data["pbc"])
            ):
                assert batched_data["is_stable_periodic"].all() or (
                    not batched_data["is_stable_periodic"].any()
                ), "Stable and unstable material structures appear in one micro-batch, which is not supported for now."
                if not batched_data["is_stable_periodic"].all():
                    use_local_attention = self.psm_config.pbc_use_local_attention
                else:
                    use_local_attention = False
                pbc_expand_batched = self.cell_expander.expand(
                    batched_data["pos"],
                    batched_data["init_pos"],
                    batched_data["pbc"],
                    batched_data["num_atoms"],
                    batched_data["masked_token_type"],
                    batched_data["cell"],
                    batched_data["node_type_edge"],
                    use_local_attention=use_local_attention,
                    use_grad=self.psm_config.AutoGradForce,
                )
            else:
                pbc_expand_batched = None

            if self.args.backbone in [
                "vanillatransformer",
                "vanillatransformer_equiv",
                "dit",
                "e2dit",
                "ditp",
                "exp",
                "exp2",
                "exp3",
            ]:
                (
                    token_embedding,
                    padding_mask,
                    time_embed,
                    mixed_attn_bias,
                    pos_embedding,
                ) = self.embedding(
                    batched_data,
                    time_step,
                    time_step_1d,
                    clean_mask,
                    aa_mask,
                    pbc_expand_batched=pbc_expand_batched,
                )
            else:
                (
                    token_embedding,
                    padding_mask,
                    time_embed,
                    mixed_attn_bias,
                ) = self.embedding(
                    batched_data,
                    time_step,
                    clean_mask,
                    aa_mask,
                    pbc_expand_batched=pbc_expand_batched,
                )

        # for invariant model struct, we first used encoder to get invariant feature
        # then used equivariant decoder to get equivariant output: like force, noise.
        if self.args.backbone in ["vanillatransformer", "vanillatransformer_equiv"]:
            encoder_output = self.encoder(
                token_embedding.transpose(0, 1),
                padding_mask,
                batched_data,
                pbc_expand_batched,
                mixed_attn_bias=mixed_attn_bias,
                ifbackprop=self.args.AutoGradForce,
            )

            with (
                torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
                if self.args.fp16
                else nullcontext()
            ):
                encoder_output = self.layer_norm(encoder_output)

                if not self.args.seq_only:
                    decoder_x_output, decoder_vec_output = self.decoder(
                        batched_data,
                        encoder_output,
                        mixed_attn_bias,
                        padding_mask,
                        pbc_expand_batched,
                    )
                encoder_output = encoder_output.transpose(0, 1)

        elif self.args.backbone in ["exp"]:
            encoder_output = self.encoder(
                token_embedding.transpose(0, 1),
                padding_mask,
                batched_data,
                pbc_expand_batched,
                mixed_attn_bias=mixed_attn_bias,
                ifbackprop=self.args.AutoGradForce,
            )

            with (
                torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
                if self.args.fp16
                else nullcontext()
            ):
                encoder_output = self.layer_norm(encoder_output)
                q, k = self.pair_proj(encoder_output.transpose(0, 1)).chunk(2, dim=-1)
                pair_feat = torch.einsum("bld,bkd->blkd", q, k)
                dist_map = self.dist_head(pair_feat)

                if not self.args.seq_only:
                    decoder_x_output = self.decoder(
                        batched_data,
                        encoder_output,
                        time_embed,
                        mixed_attn_bias,
                        padding_mask,
                        pbc_expand_batched,
                        pair_feat=pair_feat,
                    )

                decoder_vec_output = None
                encoder_output = encoder_output.transpose(0, 1)
        elif self.args.backbone in ["exp2", "exp3"]:
            encoder_output, x_pair = self.encoder(
                token_embedding.transpose(0, 1),
                padding_mask,
                batched_data,
                pbc_expand_batched,
                mixed_attn_bias=mixed_attn_bias,
                ifbackprop=self.args.AutoGradForce,
            )

            with (
                torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
                if False  # self.args.fp16
                else nullcontext()
            ):
                encoder_output = self.layer_norm(encoder_output)
                dist_map = self.dist_head(x_pair)

                if not self.args.seq_only:
                    decoder_x_output = self.decoder(
                        batched_data,
                        encoder_output,
                        time_embed,
                        mixed_attn_bias,
                        padding_mask,
                        pbc_expand_batched,
                        pair_feat=x_pair,
                    )

                decoder_vec_output = None
                encoder_output = encoder_output.transpose(0, 1)

        elif self.args.backbone in ["dit", "ditp"]:
            encoder_output = self.encoder(
                token_embedding,
                pos_embedding,
                padding_mask,
                batched_data,
                pbc_expand_batched,
                mixed_attn_bias=mixed_attn_bias,
                ifbackprop=self.args.AutoGradForce,
            )

            with (
                torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
                if self.args.fp16
                else nullcontext()
            ):
                encoder_output = self.layer_norm(encoder_output)
                decoder_x_output, decoder_vec_output = encoder_output, None

        elif self.args.backbone in ["ditgeom"]:
            encoder_output = self.encoder(
                token_embedding,
                pos_embedding,
                padding_mask,
                batched_data,
                pbc_expand_batched,
                mixed_attn_bias=mixed_attn_bias,
                ifbackprop=self.args.AutoGradForce,
            )

            with (
                torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
                if self.args.fp16
                else nullcontext()
            ):
                encoder_output = self.layer_norm(encoder_output)
                if not self.args.seq_only:
                    decoder_x_output, decoder_vec_output = self.decoder(
                        batched_data,
                        encoder_output.transpose(0, 1),
                        mixed_attn_bias,
                        # None,
                        padding_mask,
                        pbc_expand_batched,
                        time_embed=time_embed,
                    )
        elif self.args.backbone in ["e2dit"]:
            encoder_output = self.encoder(
                token_embedding,
                pos_embedding,
                padding_mask,
                batched_data,
                pbc_expand_batched,
                mixed_attn_bias=mixed_attn_bias,
                ifbackprop=self.args.AutoGradForce,
            )

            with (
                torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
                if self.args.fp16
                else nullcontext()
            ):
                encoder_output = self.layer_norm(encoder_output)
                if not self.args.seq_only:
                    (
                        decoder_x_output_noise,
                        decoder_vec_output_noise,
                        decoder_x_output,
                        decoder_vec_output,
                    ) = self.decoder(
                        batched_data,
                        encoder_output.transpose(0, 1),
                        # mixed_attn_bias,
                        None,
                        padding_mask,
                        pbc_expand_batched,
                        time_embed=time_embed,
                        sepFN=True,  # with this, noise output, force/e output are separated
                    )
        elif self.args.backbone in ["graphormer-e2"]:
            encoder_output = self.encoder(
                token_embedding.transpose(0, 1),
                padding_mask,
                batched_data,
                mixed_attn_bias,
                pbc_expand_batched,
            )

            if not skip_decoder:
                if self.psm_config.share_attention_bias:
                    decoder_attn_bias = mixed_attn_bias
                else:
                    decoder_attn_bias = mixed_attn_bias[-1]
                decoder_x_output, decoder_vec_output = self.decoder(
                    batched_data,
                    encoder_output,
                    decoder_attn_bias,
                    padding_mask,
                    pbc_expand_batched,
                    time_embed=time_embed,
                )
        elif self.args.backbone in ["graphormer"]:
            encoder_output = self.encoder(
                token_embedding.transpose(0, 1),
                padding_mask,
                batched_data,
                mixed_attn_bias,
                pbc_expand_batched,
            )

            if not skip_decoder:
                if self.psm_config.share_attention_bias:
                    decoder_attn_bias = mixed_attn_bias
                else:
                    decoder_attn_bias = mixed_attn_bias[-1]
                decoder_x_output, decoder_vec_output = self.decoder(
                    batched_data,
                    encoder_output,
                    decoder_attn_bias,
                    padding_mask,
                    pbc_expand_batched,
                    time_embed=time_embed,
                )
        elif self.args.backbone in ["vectorvanillatransformer"]:
            decoder_x_output, decoder_vec_output = self.decoder(
                batched_data,
                token_embedding.transpose(0, 1),
                mixed_attn_bias,
                padding_mask,
                pbc_expand_batched,
            )
        else:
            decoder_x_output, decoder_vec_output = self.decoder(
                batched_data,
                token_embedding.transpose(0, 1),
                mixed_attn_bias,
                padding_mask=padding_mask,
                pbc_expand_batched=pbc_expand_batched,
            )

        # atom mask to leave out unit cell corners for periodic systems
        non_atom_mask = torch.arange(
            n_nodes, dtype=torch.long, device=pos.device
        ).unsqueeze(0).repeat(n_graphs, 1) >= batched_data["num_atoms"].unsqueeze(-1)

        with (
            torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
            if False  # self.args.fp16
            else nullcontext()
        ):
            if not self.args.seq_only:
                if self.args.encoderfeat4noise:
                    assert self.args.backbone not in [
                        "graphormer",
                        "graphormer-e2",
                    ], "encoderfeat4noise=True is not compatible with graphormer and graphormer-e2"
                    invariant_output = encoder_output
                else:
                    invariant_output = decoder_x_output
                if self.psm_config.separate_noise_head:
                    molecule_noise_pred = self.molecule_noise_head(
                        invariant_output, decoder_vec_output
                    )
                    periodic_noise_pred = self.periodic_noise_head(
                        invariant_output, decoder_vec_output
                    )
                    protein_noise_pred = self.protein_noise_head(
                        invariant_output, decoder_vec_output
                    )
                    noise_pred = torch.where(
                        is_periodic[:, None, None],
                        periodic_noise_pred,
                        molecule_noise_pred,
                    )
                    noise_pred = torch.where(
                        is_protein[:, :, None], protein_noise_pred, noise_pred
                    )
                elif self.args.backbone not in ["graphormer", "graphormer-e2"]:
                    noise_pred = self.noise_head(invariant_output, decoder_vec_output)
                    periodic_noise_pred = self.periodic_noise_head(
                        invariant_output, decoder_vec_output
                    )
                    noise_pred = torch.where(
                        is_periodic[:, None, None], periodic_noise_pred, noise_pred
                    )
                else:
                    noise_pred = self.noise_head(invariant_output, decoder_vec_output)

                if self.args.decoder_feat4energy:
                    energy_per_atom = torch.where(
                        is_periodic.unsqueeze(-1),
                        self.energy_head["periodic"](decoder_x_output).squeeze(-1),
                        self.energy_head["molecule"](decoder_x_output).squeeze(-1),
                    )
                else:
                    energy_per_atom = torch.where(
                        is_periodic.unsqueeze(-1),
                        self.energy_head["periodic"](
                            encoder_output
                            if self.args.backbone not in ["graphormer", "graphormer-e2"]
                            else encoder_output.transpose(0, 1)
                        ).squeeze(-1),
                        self.energy_head["molecule"](
                            encoder_output
                            if self.args.backbone not in ["graphormer", "graphormer-e2"]
                            else encoder_output.transpose(0, 1)
                        ).squeeze(-1),
                    )

                if self.args.diffusion_mode == "edm":
                    noise_pred = (
                        batched_data["c_skip"] * pos_noised_no_c_in
                        + batched_data["c_out"] * noise_pred
                    )
                else:
                    scale_shift = self.mlp_w(time_embed)
                    logit_bias = torch.logit(
                        batched_data["sqrt_one_minus_alphas_cumprod_t"]
                    )
                    scale = torch.sigmoid(scale_shift + logit_bias)
                    if self.psm_config.diffusion_mode == "epsilon":
                        noise_pred = (
                            scale * (batched_data["pos"] - batched_data["init_pos"])
                            + (1 - scale) * noise_pred
                        )
                    elif self.psm_config.diffusion_mode == "x0":
                        noise_pred = (
                            scale * noise_pred + (1 - scale) * batched_data["pos"]
                        )
                    else:
                        raise ValueError(
                            f"diffusion mode: {self.args.diffusion_mode} is not supported"
                        )

                if (
                    self.args.AutoGradForce
                    and (clean_mask.all(dim=-1)).any()
                    and batched_data["has_forces"].any()
                    and (
                        batched_data["is_molecule"].any()
                        or batched_data["is_periodic"].any()
                    )
                ):
                    autograd_forces = self.autograd_force_head(
                        energy_per_atom,
                        non_atom_mask,
                        pos,
                        batched_data["is_periodic"],
                        batched_data["is_molecule"],
                    )
                else:
                    autograd_forces = None

                if (
                    (not self.psm_config.supervise_force_from_head_when_autograd)
                    and self.args.AutoGradForce
                    and autograd_forces is not None
                ):
                    forces = autograd_forces
                    autograd_forces = None
                elif self.args.NoisePredForce:
                    forces = (
                        -noise_pred
                        / batched_data["sqrt_one_minus_alphas_cumprod_t"]
                        / 20
                    )  # * self.inv_KbT
                else:
                    if self.psm_config.force_head_type == ForceHeadType.LINEAR:
                        forces = torch.where(
                            is_periodic.unsqueeze(-1).unsqueeze(-1),
                            self.forces_head["periodic"](decoder_vec_output).squeeze(
                                -1
                            ),
                            self.forces_head["molecule"](decoder_vec_output).squeeze(
                                -1
                            ),
                        )
                    elif self.psm_config.AutoGradForce:
                        forces = torch.zeros_like(batched_data["pos"])
                    else:
                        forces = torch.where(
                            is_periodic.unsqueeze(-1).unsqueeze(-1),
                            self.forces_head["periodic"](
                                decoder_x_output, decoder_vec_output
                            ).squeeze(-1),
                            self.forces_head["molecule"](
                                decoder_x_output, decoder_vec_output
                            ).squeeze(-1),
                        )

                # per-atom energy prediction
                total_energy = energy_per_atom.masked_fill(non_atom_mask, 0.0).sum(
                    dim=-1
                )
                energy_per_atom = total_energy / batched_data["num_atoms"]
            else:
                energy_per_atom = torch.zeros_like(batched_data["num_atoms"])
                total_energy = torch.zeros_like(batched_data["num_atoms"])
                forces = torch.zeros_like(batched_data["pos"])
                noise_pred = torch.zeros_like(batched_data["pos"])

            if not (
                self.psm_config.psm_finetune_mode
                and self.psm_config.psm_finetune_skip_ori_head
            ):
                aa_logits = self.aa_mask_head(encoder_output)
                # q, k = self.pair_proj(encoder_output).chunk(2, dim=-1)
                # dist_map = torch.einsum("bld,bkd->blkd", q, k)
                # dist_map = self.dist_head(dist_map)
            else:
                aa_logits = self.aa_mask_head(decoder_x_output)

                # q, k = self.pair_proj(decoder_x_output).chunk(2, dim=-1)
                # pair_feat = torch.einsum("bld,bkd->blkd", q, k)
                # dist_map = self.dist_head(dist_map)

        result_dict = {
            "energy_per_atom": energy_per_atom,
            "total_energy": total_energy,
            "forces": forces,
            "aa_logits": aa_logits,
            "time_step": time_step,
            "noise_pred": noise_pred,
            "non_atom_mask": non_atom_mask,
            "protein_mask": batched_data["protein_mask"],
            "is_molecule": is_molecule,
            "is_periodic": is_periodic,
            "is_protein": is_protein,
            "is_complex": batched_data["is_complex"],
            "is_seq_only": batched_data["is_seq_only"],
            "num_atoms": batched_data["num_atoms"],
            "pos": batched_data["pos"],
            "dist_map": dist_map,
        }

        if "one_hot_token_id" in batched_data:
            result_dict["one_hot_token_id"] = batched_data["one_hot_token_id"]

        if self.psm_config.psm_finetune_mode:
            result_dict.update(
                {
                    "decoder_x_output": decoder_x_output,
                    "decoder_vec_output": decoder_vec_output,
                }
            )

        return result_dict

    def reset_head_for_finetune(self):
        def _reset_one_head(head_module: nn.Module, prefix: str):
            if hasattr(head_module, "reset_parameters"):
                head_module.reset_parameters()
                logger.info(f"Reset parameters successfully in {prefix}")
            else:
                for name, module in head_module.named_children():
                    logger.info(f"Reset parameters into {prefix}.{name}")
                    _reset_one_head(module, prefix + "." + name)

        _reset_one_head(self.energy_head, "energy_head")
        _reset_one_head(self.forces_head, "forces_head")
        if self.psm_config.separate_noise_head:
            _reset_one_head(self.periodic_noise_head, "periodic_noise_head")
            _reset_one_head(self.molecule_noise_head, "molecule_noise_head")
            _reset_one_head(self.protein_noise_head, "protein_noise_head")
        else:
            if self.args.backbone not in ["graphormer", "graphormer-e2"]:
                _reset_one_head(self.periodic_noise_head, "periodic_noise_head")
            _reset_one_head(self.noise_head, "noise_head")

    def init_state_dict_weight(self, weight, bias):
        """
        Initialize the state dict weight.
        """
        pass
