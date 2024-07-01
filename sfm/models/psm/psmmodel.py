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
from tqdm import tqdm

from sfm.logging import logger
from sfm.models.psm.equivariant.equiformer_series import Equiformerv2SO2
from sfm.models.psm.equivariant.equivariant import EquivariantDecoder
from sfm.models.psm.equivariant.geomformer import EquivariantVectorOutput
from sfm.models.psm.equivariant.nodetaskhead import (
    NodeTaskHead,
    VectorOutput,
    VectorProjOutput,
)
from sfm.models.psm.invariant.dit_encoder import PSMDiTEncoder
from sfm.models.psm.invariant.invariant_encoder import PSMEncoder
from sfm.models.psm.invariant.plain_encoder import PSMPlainEncoder
from sfm.models.psm.modules.embedding import PSMMixEmbedding
from sfm.models.psm.modules.mixembedding import PSMMix3dEmbedding
from sfm.models.psm.modules.mixembedding_equiv import PSMMix3DEquivEmbedding
from sfm.models.psm.modules.pbc import CellExpander
from sfm.models.psm.psm_config import PSMConfig
from sfm.modules.layer_norm import AdaNorm
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model

from .modules.autograd import GradientHead
from .modules.dataaug import uniform_random_rotation
from .modules.diffusion import DIFFUSION_PROCESS_REGISTER
from .modules.sampled_structure_converter import SampledStructureConverter
from .modules.timestep_encoder import DiffNoise, TimeStepSampler


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

        self.net = PSM(args, self.psm_config)

        if self.psm_config.psm_finetune_mode or self.psm_config.psm_validation_mode:
            if os.path.exists(args.loadcheck_path):
                self.load_pretrained_weights(args, checkpoint_path=args.loadcheck_path)
            else:
                logger.warning(
                    "Finetune or validation mode, but no checkpoint is loaded"
                )
        else:
            logger.info("No checkpoint is loaded")

        self.diffnoise = DiffNoise(self.psm_config)
        self.diffusion_process = DIFFUSION_PROCESS_REGISTER[
            self.psm_config.diffusion_sampling
        ](self.diffnoise.alphas_cumprod)

        self.time_step_sampler = TimeStepSampler(self.psm_config.num_timesteps)

        self.loss_fn = loss_fn(args)

        if self.args.backbone in ["vanillatransformer"]:
            self.disable_data_aug = getattr(self.args, "disable_data_aug", False)
            if self.disable_data_aug:
                logger.warning(
                    f"=== N O T E === Data augmentation is disabled for {self.args.backbone}"
                )

        if self.psm_config.sample_in_validation:
            self.sampled_structure_converter = SampledStructureConverter(
                self.psm_config.sampled_structure_output_path
            )

        if self.psm_config.psm_finetune_mode:
            settings = dict(
                psm_finetune_reset_head=self.psm_config.psm_finetune_reset_head,
                psm_finetune_head=(
                    psm_finetune_head.__class__ if psm_finetune_head else None
                ),
                psm_finetune_noise_mode=self.psm_config.psm_finetune_noise_mode,
            )
            logger.info(f"Finetune settings: {settings}")
            if self.psm_config.psm_finetune_reset_head:
                self.net.reset_head_for_finetune()
            self.psm_finetune_head = psm_finetune_head
        else:
            assert not psm_finetune_head
            self.psm_finetune_head = None

        try:
            mode_prob = [float(item) for item in self.psm_config.mode_prob.split(",")]
            assert len(mode_prob) == 3
            assert sum(mode_prob) == 1.0
        except:
            mode_prob = [0.0, 0.0, 1.0]
        self.mode_prob = mode_prob
        logger.info(f"protein mode prob: {mode_prob}")

    def _create_initial_pos_for_diffusion(self, batched_data):
        periodic_mask = torch.any(batched_data["pbc"], dim=-1)
        ori_pos = batched_data["pos"][periodic_mask]
        n_periodic_graphs = ori_pos.size()[0]
        init_cell_pos = torch.zeros_like(ori_pos)
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
        ).repeat([n_periodic_graphs, 1, 1]) * self.psm_config.diff_init_lattice_size - (
            self.psm_config.diff_init_lattice_size / 2.0
        )  # centering
        scatter_index = torch.arange(8, device=ori_pos.device).unsqueeze(0).unsqueeze(
            -1
        ).repeat([n_periodic_graphs, 1, 3]) + batched_data["num_atoms"][
            periodic_mask
        ].unsqueeze(
            -1
        ).unsqueeze(
            -1
        )
        init_cell_pos = init_cell_pos.scatter(1, scatter_index, init_cell_pos_input)
        batched_data["init_pos"] = torch.zeros_like(batched_data["pos"])
        batched_data["init_pos"][periodic_mask] = init_cell_pos

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

        masked_protein = (
            ((token_id > 129) & (token_id < 158))
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
        batched_data["protein_mask"] = mask

    def _protein_pretrain_mode(
        self, clean_mask, aa_mask, padding_mask, is_protein_mask, is_seq_only, time_step
    ):
        n_graph, nnodes = aa_mask.size()[:2]
        mask_choice = np.random.choice(np.arange(3), n_graph, p=self.mode_prob)
        mask_choice = torch.tensor([i for i in mask_choice]).to(clean_mask.device)
        clean_mask = clean_mask.unsqueeze(-1).repeat(1, nnodes)
        mask_choice = mask_choice.unsqueeze(-1).repeat(1, nnodes)
        time_step = time_step.unsqueeze(-1).repeat(1, nnodes)
        # set T noise if protein is seq only
        time_step = time_step.masked_fill(is_seq_only.unsqueeze(-1), 1.0)

        # mode 0: 50% masked seq and 50% noised structure to structure and seq
        clean_mask = torch.where(
            (mask_choice == 0) & is_protein_mask.unsqueeze(-1), ~aa_mask, clean_mask
        )
        clean_mask = torch.where(is_seq_only.unsqueeze(-1), False, clean_mask)

        # mode 1: clean seq to structure
        aa_mask = torch.where(mask_choice == 1, False, aa_mask)

        # mode 2: 50% masked seq to structure and masked seq
        # nothing to do

        # set padding mask to clean
        clean_mask = clean_mask.masked_fill(padding_mask, True)

        return clean_mask, aa_mask, time_step

    def _create_system_tags(self, batched_data):
        token_id = batched_data["token_id"]
        sample_type = batched_data["sample_type"]
        is_periodic = batched_data["pbc"].any(dim=-1)
        is_molecule = (~is_periodic) & (token_id <= 129).all(dim=-1)
        is_protein = (~is_periodic) & (token_id > 129).any(dim=-1)
        is_heavy_atom = is_molecule & (token_id > 37).any(dim=-1)
        is_seq_only = (sample_type == 5) & is_protein

        batched_data["is_periodic"] = is_periodic
        batched_data["is_molecule"] = is_molecule
        batched_data["is_protein"] = is_protein
        batched_data["is_heavy_atom"] = is_heavy_atom
        batched_data["is_seq_only"] = is_seq_only

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
        periodic_index = torch.nonzero(is_periodic)[:, 0]
        diff_loss_mask[periodic_index, batched_data["num_atoms"][is_periodic]] = True
        diff_loss_mask[
            periodic_index, batched_data["num_atoms"][is_periodic] + 1
        ] = True
        diff_loss_mask[
            periodic_index, batched_data["num_atoms"][is_periodic] + 2
        ] = True
        diff_loss_mask[
            periodic_index, batched_data["num_atoms"][is_periodic] + 4
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
        time_step=None,
        clean_mask=None,
        infer=False,
    ):
        """
        set diffusion noise here
        """

        ori_pos = center_pos(batched_data, padding_mask)
        ori_pos = ori_pos.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        self._create_initial_pos_for_diffusion(batched_data)

        if self.args.backbone == "vanillatransformer" and not self.disable_data_aug:
            R = uniform_random_rotation(
                ori_pos.size(0), device=ori_pos.device, dtype=ori_pos.dtype
            )
            ori_pos = torch.bmm(ori_pos, R)
            batched_data["forces"] = torch.bmm(batched_data["forces"], R)
            batched_data["init_pos"] = torch.bmm(batched_data["init_pos"], R)
            batched_data["cell"] = torch.bmm(batched_data["cell"], R)

        batched_data["ori_pos"] = ori_pos

        noise_pos, noise, sqrt_one_minus_alphas_cumprod_t = self.diffnoise.noise_sample(
            x_start=ori_pos,
            t=time_step,
            non_atom_mask=batched_data["non_atom_mask"],
            is_periodic=batched_data["is_periodic"],
            x_init=batched_data["init_pos"],
            clean_mask=clean_mask,
        )
        noise_pos = complete_cell(noise_pos, batched_data)

        return noise_pos, noise, sqrt_one_minus_alphas_cumprod_t

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
        for sample_time_index in range(self.psm_config.num_sampling_time):
            original_pos = batched_data["pos"].clone()
            batched_data["pos"] = torch.zeros_like(
                batched_data["pos"]
            )  # zero position to avoid any potential leakage
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
        for key in match_results:
            match_results[key] = torch.tensor(
                match_results[key], device=batched_data["pos"].device
            )
            match_results[key] = (
                match_results[key].view(self.psm_config.num_sampling_time, -1).T
            )
        return match_results

    def forward(self, batched_data, **kwargs):
        """
        Forward pass of the model.

        Args:
            batched_data: Input data for the forward pass.
            **kwargs: Additional keyword arguments.
        """

        if self.psm_config.sample_in_validation and not self.training:
            match_results = self.sample_and_calc_match_metric(batched_data)

        self._create_system_tags(batched_data)
        self._create_protein_mask(batched_data)
        pos = batched_data["pos"]

        n_graphs = pos.size(0)
        time_step, clean_mask = self.time_step_sampler.sample(
            n_graphs, pos.device, pos.dtype, self.psm_config.clean_sample_ratio
        )
        clean_mask = (
            clean_mask & ~batched_data["is_protein"]
        )  # Proteins are always corrupted. For proteins, we only consider diffusion training on structure for now.
        # Do not predict energy of molecule with heavy atom avoid loss instability.
        clean_mask = clean_mask & ~batched_data["is_heavy_atom"]

        clean_mask = clean_mask | (
            (batched_data["is_periodic"]) & (~batched_data["is_stable_periodic"])
        )  # A periodic sample which is not stable is always clean

        clean_mask = clean_mask & (
            (~batched_data["is_periodic"]) | (~batched_data["is_stable_periodic"])
        )  # A periodic sample which is stable is always corrupted

        token_id = batched_data["token_id"]
        padding_mask = token_id.eq(0)  # B x T x 1
        aa_mask = batched_data["protein_masked_aa"] & batched_data[
            "is_protein"
        ].unsqueeze(-1)
        aa_mask = aa_mask & ~padding_mask

        clean_mask, aa_mask, time_step = self._protein_pretrain_mode(
            clean_mask,
            aa_mask,
            padding_mask,
            batched_data["is_protein"],
            batched_data["is_seq_only"],
            time_step,
        )

        if self.psm_config.psm_finetune_mode:
            if self.training:
                noise_mode = self.psm_config.psm_finetune_noise_mode
            else:
                noise_mode = self.psm_config.psm_finetune_valid_noise_mode

            if noise_mode == "T":
                time_step = torch.ones_like(time_step)
                clean_mask = torch.zeros_like(clean_mask)
                batched_data["pos"] = torch.zeros_like(batched_data["pos"])
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

        pos, noise, sqrt_one_minus_alphas_cumprod_t = self._set_noise(
            padding_mask=padding_mask,
            batched_data=batched_data,
            time_step=time_step,
            clean_mask=clean_mask,
        )
        batched_data["pos"] = pos
        batched_data[
            "sqrt_one_minus_alphas_cumprod_t"
        ] = sqrt_one_minus_alphas_cumprod_t
        result_dict = self.net(
            batched_data,
            time_step=time_step,
            clean_mask=clean_mask,
            aa_mask=aa_mask,
            **kwargs,
        )
        result_dict["noise"] = noise
        result_dict["clean_mask"] = clean_mask
        result_dict["aa_mask"] = aa_mask
        result_dict["diff_loss_mask"] = batched_data["diff_loss_mask"]
        result_dict["ori_pos"] = batched_data["ori_pos"]
        result_dict["sqrt_one_minus_alphas_cumprod_t"] = batched_data[
            "sqrt_one_minus_alphas_cumprod_t"
        ]
        result_dict["force_label"] = batched_data["forces"]

        if self.psm_config.sample_in_validation and not self.training:
            result_dict.update(match_results)

        if self.psm_finetune_head:
            result_dict = self.psm_finetune_head(result_dict)

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

    def config_optimizer(self):
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

        self._create_system_tags(batched_data)
        self._create_protein_mask(batched_data)

        device = batched_data["pos"].device

        n_graphs = batched_data["pos"].shape[0]

        token_id = batched_data["token_id"]
        padding_mask = token_id.eq(0)  # B x T x 1

        orig_pos = center_pos(batched_data, padding_mask)

        self._create_initial_pos_for_diffusion(batched_data)

        batched_data["pos"] = self.diffnoise.get_sampling_start(
            batched_data["init_pos"],
            batched_data["non_atom_mask"],
            batched_data["is_periodic"],
        )
        batched_data["pos"] = complete_cell(batched_data["pos"], batched_data)
        batched_data["pos"] = center_pos(
            batched_data, padding_mask=padding_mask
        )  # centering to remove noise translation

        for t in tqdm(range(self.psm_config.num_timesteps - 1, -1, -1)):
            # forward
            time_step = self.time_step_sampler.get_continuous_time_step(
                t, n_graphs, device=device, dtype=batched_data["pos"].dtype
            )
            time_step = time_step.unsqueeze(-1)
            batched_data["sqrt_one_minus_alphas_cumprod_t"] = self.diffnoise._extract(
                self.diffnoise.sqrt_one_minus_alphas_cumprod,
                (time_step * self.psm_config.num_timesteps).long(),
                batched_data["pos"].shape,
            )
            predicted_noise = self.net(batched_data, time_step=time_step)["noise_pred"]
            epsilon = self.diffnoise.get_noise(
                batched_data["pos"],
                batched_data["non_atom_mask"],
                batched_data["is_periodic"],
            )

            batched_data["pos"] = self.diffusion_process.sample_step(
                batched_data["pos"],
                batched_data["init_pos"],
                predicted_noise,
                epsilon,
                t,
            )
            batched_data["pos"] = complete_cell(batched_data["pos"], batched_data)
            batched_data["pos"] = center_pos(
                batched_data, padding_mask=padding_mask
            )  # centering to remove noise translation
            batched_data["pos"] = batched_data["pos"].detach()

        pred_pos = batched_data["pos"].clone()

        loss = torch.sum((pred_pos - orig_pos) ** 2, dim=-1, keepdim=True)

        return {"loss": loss, "pred_pos": pred_pos, "orig_pos": orig_pos}


def center_pos(batched_data, padding_mask):
    # get center of system positions
    is_periodic = batched_data["is_periodic"]  # B x 3 -> B
    periodic_center = (
        torch.gather(
            batched_data["pos"][is_periodic],
            index=batched_data["num_atoms"][is_periodic]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, 3),
            dim=1,
        )
        + torch.gather(
            batched_data["pos"][is_periodic],
            index=batched_data["num_atoms"][is_periodic]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, 3)
            + 7,
            dim=1,
        )
    ) / 2.0
    protein_mask = batched_data["protein_mask"]
    non_periodic_center = torch.sum(
        batched_data["pos"].masked_fill(padding_mask.unsqueeze(-1) | protein_mask, 0.0),
        dim=1,
    ) / batched_data["num_atoms"].unsqueeze(-1)
    center = non_periodic_center.unsqueeze(1)
    center[is_periodic] = periodic_center
    batched_data["pos"] -= center
    batched_data["pos"] = batched_data["pos"].masked_fill(
        padding_mask.unsqueeze(-1), 0.0
    )
    return batched_data["pos"]


def complete_cell(pos, batched_data):
    periodic_mask = torch.any(batched_data["pbc"], dim=-1)
    periodic_pos = pos[periodic_mask]
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
        periodic_mask
    ].unsqueeze(
        -1
    ).unsqueeze(
        -1
    )
    lattice = torch.gather(periodic_pos, 1, index=gather_index)
    corner = lattice[:, 0, :]
    lattice = lattice[:, 1:, :] - corner.unsqueeze(1)
    batched_data["cell"][periodic_mask, :, :] = lattice
    cell = torch.matmul(cell_matrix, lattice) + corner.unsqueeze(1)
    scatter_index = torch.arange(8, device=device).unsqueeze(0).unsqueeze(-1).repeat(
        [n_graphs, 1, 3]
    ) + batched_data["num_atoms"][periodic_mask].unsqueeze(-1).unsqueeze(-1)
    cell -= ((cell[:, 0, :] + cell[:, 7, :]) / 2.0).unsqueeze(1)
    periodic_pos = periodic_pos.scatter(1, scatter_index, cell)
    pos[periodic_mask] = periodic_pos

    token_id = batched_data["token_id"]
    padding_mask = token_id.eq(0)  # B x T x 1
    pos = pos.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    return pos


class PSM(nn.Module):
    """
    Class for training Physics science module
    """

    def __init__(self, args, psm_config: PSMConfig):
        super().__init__()
        self.max_positions = args.max_positions
        self.args = args

        self.psm_config = psm_config

        self.cell_expander = CellExpander(
            self.psm_config.pbc_cutoff,
            self.psm_config.pbc_expanded_token_cutoff,
            self.psm_config.pbc_expanded_num_cell_per_direction,
            self.psm_config.pbc_multigraph_cutoff,
        )

        # Implement the embedding
        if args.backbone in ["vanillatransformer", "dit"]:
            self.embedding = PSMMix3dEmbedding(
                psm_config, use_unified_batch_sampler=args.use_unified_batch_sampler
            )
            # self.embedding = PSMMixEmbedding(psm_config)
        elif args.backbone == "vanillatransformer_equiv":
            self.embedding = PSMMix3DEquivEmbedding(psm_config)
        else:
            self.embedding = PSMMixEmbedding(psm_config)

        self.encoder = None
        if args.backbone == "graphormer":
            # Implement the encoder
            self.encoder = PSMEncoder(args, psm_config)
            # Implement the decoder
            self.decoder = EquivariantDecoder(psm_config)
        elif args.backbone == "equiformerv2":
            args.backbone_config[
                "embedding_dim"
            ] = psm_config.encoder_embed_dim  # parameter unified!

            self.decoder = Equiformerv2SO2(**args.backbone_config)
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
        elif args.backbone in ["dit"]:
            # Implement the encoder
            self.encoder = PSMDiTEncoder(args, psm_config)
            # Implement the decoder
            # self.decoder = EquivariantDecoder(psm_config)
            self.decoder = NodeTaskHead(psm_config)
        else:
            raise NotImplementedError

        # simple energy, force and noise prediction heads
        self.energy_head = nn.ModuleDict()
        self.forces_head = nn.ModuleDict()

        for key in {"molecule", "periodic", "protein"}:
            if args.backbone in ["vanillatransformer", "vanillatransformer_equiv"]:
                self.energy_head.update(
                    {
                        key: nn.Sequential(
                            AdaNorm(psm_config.embedding_dim),
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

            if args.backbone in ["vanillatransformer", "vanillatransformer_equiv"]:
                self.noise_head = VectorOutput(psm_config.embedding_dim)
                self.forces_head.update({key: VectorOutput(psm_config.embedding_dim)})
            elif args.backbone in ["dit"]:
                self.noise_head = VectorProjOutput(psm_config.embedding_dim)
                self.forces_head.update(
                    {key: VectorProjOutput(psm_config.embedding_dim)}
                )
            else:
                self.noise_head = EquivariantVectorOutput(psm_config.embedding_dim)
                self.forces_head.update(
                    {key: EquivariantVectorOutput(psm_config.embedding_dim)}
                )

        # aa mask predict head
        self.aa_mask_head = nn.Sequential(
            nn.Linear(psm_config.embedding_dim, psm_config.embedding_dim, bias=False),
            nn.SiLU(),
            nn.Linear(psm_config.embedding_dim, 160, bias=False),
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
        ]:
            self.layer_norm = nn.LayerNorm(psm_config.embedding_dim)
            self.layer_norm_vec = nn.LayerNorm(psm_config.embedding_dim)

        if self.args.AutoGradForce:
            self.autograd_force_head = GradientHead()

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
        node_type_edge = torch.cat([masked_token_type_i, masked_token_type_j], dim=-1)
        batched_data["node_type_edge"] = node_type_edge

    def forward(
        self,
        batched_data,
        time_step=None,
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

        if self.args.AutoGradForce:
            pos.requires_grad_(True)

        n_graphs, n_nodes = pos.size()[:2]
        is_periodic = batched_data["is_periodic"]
        is_molecule = batched_data["is_molecule"]
        is_protein = batched_data["is_protein"]

        self._set_aa_mask(batched_data, aa_mask)
        self._create_node_type_edge(batched_data)

        # B, L, H is Batch, Length, Hidden
        # token_embedding: B x L x H
        # padding_mask: B x L
        # token_type: B x L  (0 is used for PADDING)
        with (
            torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
            if self.args.fp16
            else nullcontext()
        ):
            if (
                "pbc" in batched_data
                and batched_data["pbc"] is not None
                and torch.any(batched_data["pbc"])
            ):
                pbc_expand_batched = self.cell_expander.expand(
                    batched_data["pos"],
                    batched_data["init_pos"],
                    batched_data["pbc"],
                    batched_data["num_atoms"],
                    batched_data["masked_token_type"],
                    batched_data["cell"],
                    batched_data["node_type_edge"],
                    self.psm_config.pbc_use_local_attention,
                )
            else:
                pbc_expand_batched = None

            token_embedding, padding_mask, time_embed, mixed_attn_bias = self.embedding(
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
                        padding_mask,
                        pbc_expand_batched,
                    )
        elif self.args.backbone in ["dit"]:
            encoder_output = self.encoder(
                token_embedding,
                mixed_attn_bias,
                padding_mask,
                batched_data,
                pbc_expand_batched,
                ifbackprop=self.args.AutoGradForce,
            )

            with (
                torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
                if self.args.fp16
                else nullcontext()
            ):
                encoder_output = self.layer_norm(encoder_output)
                decoder_x_output, decoder_vec_output = encoder_output, encoder_output
                encoder_output = encoder_output.transpose(0, 1)

        elif self.encoder is not None:
            encoder_output = self.encoder(
                token_embedding.transpose(0, 1),
                padding_mask,
                batched_data,
                mixed_attn_bias,
                pbc_expand_batched,
            )

            decoder_x_output, decoder_vec_output = self.decoder(
                batched_data,
                encoder_output,
                mixed_attn_bias[-1],
                padding_mask,
                pbc_expand_batched,
            )
        else:
            decoder_x_output, decoder_vec_output = self.decoder(
                batched_data,
                token_embedding.transpose(0, 1),
                padding_mask,
                pbc_expand_batched=pbc_expand_batched,
            )

        with (
            torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
            if self.args.fp16
            else nullcontext()
        ):
            if not self.args.seq_only:
                if self.args.backbone in ["dit"]:
                    noise_pred = self.noise_head(decoder_x_output, decoder_vec_output)
                else:
                    noise_pred = self.noise_head(decoder_x_output, decoder_vec_output)
                # noise_pred = self.noise_head(encoder_output.transpose(0, 1), decoder_vec_output)

                energy_per_atom = torch.where(
                    is_periodic.unsqueeze(-1),
                    self.energy_head["periodic"](decoder_x_output).squeeze(-1),
                    self.energy_head["molecule"](decoder_x_output).squeeze(-1),
                )

                # atom mask to leave out unit cell corners for periodic systems
                non_atom_mask = torch.arange(
                    n_nodes, dtype=torch.long, device=energy_per_atom.device
                ).unsqueeze(0).repeat(n_graphs, 1) >= batched_data[
                    "num_atoms"
                ].unsqueeze(
                    -1
                )

                if self.args.AutoGradForce and pbc_expand_batched is not None:
                    forces = self.autograd_force_head(
                        energy_per_atom.masked_fill(non_atom_mask, 0.0).sum(
                            dim=-1, keepdim=True
                        ),
                        pos,
                    )
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

                if self.args.diffusion_mode == "epsilon":
                    scale_shift = self.mlp_w(time_embed)  # .unsqueeze(-1)
                    logit_bias = torch.logit(
                        batched_data["sqrt_one_minus_alphas_cumprod_t"]
                    )
                    scale = torch.sigmoid(scale_shift + logit_bias)
                    noise_pred = (
                        scale * (batched_data["pos"] - batched_data["init_pos"])
                        + (1 - scale) * noise_pred
                    )
                elif self.args.diffusion_mode == "x0":
                    scale_shift = self.mlp_w(time_embed)  # .unsqueeze(-1)
                    logit_bias = torch.logit(
                        batched_data["sqrt_one_minus_alphas_cumprod_t"]
                    )
                    scale = torch.sigmoid(scale_shift + logit_bias)
                    noise_pred = scale * noise_pred + (1 - scale) * batched_data["pos"]
                else:
                    raise ValueError(
                        f"diffusion mode: {self.args.diffusion_mode} is not supported"
                    )

                # per-atom energy prediction
                energy_per_atom = (
                    energy_per_atom.masked_fill(non_atom_mask, 0.0).sum(dim=-1)
                    / batched_data["num_atoms"]
                )
            else:
                energy_per_atom = torch.zeros_like(batched_data["num_atoms"])
                forces = torch.zeros_like(batched_data["pos"])
                noise_pred = torch.zeros_like(batched_data["pos"])
                non_atom_mask = torch.zeros_like(batched_data["pos"])

            if self.encoder is not None:
                aa_logits = self.aa_mask_head(encoder_output.transpose(0, 1))
            else:
                aa_logits = self.aa_mask_head(decoder_x_output)

        result_dict = {
            "energy_per_atom": energy_per_atom,
            "forces": forces,
            "aa_logits": aa_logits,
            "time_step": time_step,
            "noise_pred": noise_pred,
            "non_atom_mask": non_atom_mask,
            "protein_mask": batched_data["protein_mask"],
            "is_molecule": is_molecule,
            "is_periodic": is_periodic,
            "is_protein": is_protein,
            "is_seq_only": batched_data["is_seq_only"],
            "num_atoms": batched_data["num_atoms"],
        }

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
        _reset_one_head(self.noise_head, "noise_head")

    def init_state_dict_weight(self, weight, bias):
        """
        Initialize the state dict weight.
        """
        pass
