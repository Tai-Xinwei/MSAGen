# -*- coding: utf-8 -*-
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from sfm.logging import logger
from sfm.models.psm.equivariant.equiformer_series import Equiformerv2SO2
from sfm.models.psm.equivariant.equivariant import EquivariantDecoder
from sfm.models.psm.equivariant.geomformer import EquivariantVectorOutput
from sfm.models.psm.equivariant.nodetaskhead import NodeTaskHead, VectorOutput
from sfm.models.psm.invariant.invariant_encoder import PSMEncoder
from sfm.models.psm.invariant.plain_encoder import PSMPlainEncoder
from sfm.models.psm.modules.embedding import PSMMixEmbedding
from sfm.models.psm.modules.mixembedding import PSMMix3dEmbedding
from sfm.models.psm.modules.mixembedding_equiv import PSMMix3DEquivEmbedding
from sfm.models.psm.modules.pbc import CellExpander
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psmmodel import PSMModel, complete_cell
from sfm.modules.layer_norm import AdaNorm
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model

from .modules.diffusion import DIFFUSION_PROCESS_REGISTER
from .modules.sampled_structure_converter import SampledStructureConverter
from .modules.timestep_encoder import DiffNoise, TimeStepSampler


class ComplexNoise(DiffNoise):
    def __init__(self, psm_config: PSMConfig):
        super().__init__(psm_config)

    def noise_sample(
        self,
        x_start,
        t,
        protein_len,
        num_atoms,
        x_init=None,
        clean_mask: Optional[torch.Tensor] = None,
    ):
        t = (t * self.psm_config.num_timesteps).long()
        noise = self.get_noise(x_start, protein_len, num_atoms)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        if x_init is None:
            x_t = (
                sqrt_alphas_cumprod_t * x_start
                + sqrt_one_minus_alphas_cumprod_t * noise
            )
        else:
            x_t = (
                sqrt_alphas_cumprod_t * (x_start - x_init)
                + sqrt_one_minus_alphas_cumprod_t * noise
                + x_init
            )

        if clean_mask is not None:
            if len(clean_mask.shape) == 1:
                x_t = torch.where(clean_mask.unsqueeze(-1).unsqueeze(-1), x_start, x_t)
            elif len(clean_mask.shape) == 2:
                x_t = torch.where(clean_mask.unsqueeze(-1), x_start, x_t)
            else:
                raise ValueError(
                    f"clean_mask should be [B] or [B, L] tensor, but it's shape is {clean_mask.shape}"
                )

        return x_t, noise, sqrt_one_minus_alphas_cumprod_t

    def get_sample_noise(self, pos, protein_len, num_atoms):
        # pos : B x L x 3
        # first protein_len atoms are protein atoms
        # protein atoms should not be noised
        noise = torch.zeros_like(pos)
        for i in range(pos.size(0)):
            ligand_len = num_atoms[i] - protein_len[i]
            noise[i, protein_len[i] : num_atoms[i]] = torch.randn(ligand_len, 3)

            # TODO: cannot recenter the noise, some ligand have only one atom
            # # recenter
            # noise_center = (
            #     noise[i, protein_len[i] : num_atoms[i]].sum(dim=0, keepdim=True)
            #     / ligand_len
            # )
            # noise[i, protein_len[i] : num_atoms[i]] -= noise_center

        return noise * self.unit_noise_scale

    def get_noise(self, pos, protein_len, num_atoms):
        # pos : B x L x 3
        # first protein_len atoms are protein atoms
        # protein atoms should not be noised
        noise = torch.randn_like(pos) * self.unit_noise_scale
        for i in range(pos.size(0)):
            noise[i, num_atoms[i] :] = 0.0

        return noise

    def get_sampling_start(self, init_pos, protein_len, num_atoms):
        noise = self.get_sample_noise(init_pos, protein_len, num_atoms)
        return init_pos + noise


class ComplexModel(PSMModel):
    def __init__(
        self, args, loss_fn=None, not_init=False, psm_finetune_head: nn.Module = None
    ):
        super().__init__(args, loss_fn, not_init, psm_finetune_head)
        self.diffnoise = ComplexNoise(self.psm_config)

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
        ori_pos = center_pos(batched_data, padding_mask)
        ori_pos = ori_pos.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        self._create_initial_pos_for_diffusion(batched_data)
        batched_data["ori_pos"] = batched_data["pos"]

        noise_pos, noise, sqrt_one_minus_alphas_cumprod_t = self.diffnoise.noise_sample(
            x_start=ori_pos,
            t=time_step,
            protein_len=batched_data["protein_len"],
            num_atoms=batched_data["num_atoms"],
            x_init=batched_data["init_pos"],
            clean_mask=clean_mask,
        )
        noise_pos = complete_cell(noise_pos, batched_data)

        return noise_pos, noise, sqrt_one_minus_alphas_cumprod_t

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
            batched_data["protein_len"],
            batched_data["num_atoms"],
        )
        # Known protein atoms
        for i in range(batched_data["pos"].size(0)):
            batched_data["pos"][i, : batched_data["protein_len"][i]] = orig_pos[
                i, : batched_data["protein_len"][i]
            ]

        batched_data["pos"] = complete_cell(
            batched_data["pos"], batched_data, is_sampling=True
        )
        # batched_data["pos"] = center_pos(
        #    batched_data, padding_mask=padding_mask
        # )  # centering to remove noise translation

        for t in tqdm(range(self.psm_config.num_timesteps - 1, -1, -1)):
            # forward
            time_step = self.time_step_sampler.get_continuous_time_step(
                t, n_graphs, device=device, dtype=batched_data["pos"].dtype
            )
            time_step = time_step.unsqueeze(-1).repeat(
                1, batched_data["pos"].shape[1]
            )  # B x L
            for j in range(batched_data["pos"].shape[0]):
                time_step[
                    j, : batched_data["protein_len"][j]
                ] = 0.0  # Protein atoms are not noised
            batched_data["sqrt_one_minus_alphas_cumprod_t"] = self.diffnoise._extract(
                self.diffnoise.sqrt_one_minus_alphas_cumprod,
                (time_step * self.psm_config.num_timesteps).long(),
                batched_data["pos"].shape,
            )
            predicted_noise = self.net(batched_data, time_step=time_step)["noise_pred"]
            for i in range(predicted_noise.size(0)):
                predicted_noise[i, : batched_data["protein_len"][i]] = 0.0

            epsilon = self.diffnoise.get_noise(
                batched_data["pos"],
                batched_data["protein_len"],
                batched_data["num_atoms"],
            )
            tmp_pos = batched_data["pos"]
            batched_data["pos"] = self.diffusion_process.sample_step(
                batched_data["pos"],
                batched_data["init_pos"],
                predicted_noise,
                epsilon,
                t,
            )
            # Known protein atoms should not be de-noised
            for i in range(batched_data["pos"].size(0)):
                batched_data["pos"][i, : batched_data["protein_len"][i]] = tmp_pos[
                    i, : batched_data["protein_len"][i]
                ]

            batched_data["pos"] = complete_cell(
                batched_data["pos"], batched_data, is_sampling=True
            )
            # batched_data["pos"] = center_pos(
            #     batched_data, padding_mask=padding_mask
            # )  # centering to remove noise translation
            batched_data["pos"] = batched_data["pos"].detach()

        pred_pos = batched_data["pos"].clone()

        loss = torch.sum((pred_pos - orig_pos) ** 2, dim=-1, keepdim=True)

        return {"loss": loss, "pred_pos": pred_pos, "orig_pos": orig_pos}

    def _create_initial_pos_for_diffusion(self, batched_data):
        super()._create_initial_pos_for_diffusion(batched_data)
        batched_data["init_pos"] = torch.zeros_like(batched_data["init_pos"])

    def forward(self, batched_data, **kwargs):
        """
        Forward pass of the model.

        Args:
            batched_data: Input data for the forward pass.
            **kwargs: Additional keyword arguments.
        """

        if self.psm_config.sample_in_validation and not self.training:
            rmsds = []
            for sample_time_index in range(self.psm_config.num_sampling_time):
                original_pos = batched_data["pos"].clone()
                batched_data["pos"] = torch.zeros_like(
                    batched_data["pos"]
                )  # zero position to avoid any potential leakage
                self.sample(batched_data=batched_data)
                rmsds_one_time = self.sampled_structure_converter.convert_and_match(
                    batched_data, original_pos, sample_time_index
                )
                rmsds.append(rmsds_one_time)
                batched_data[
                    "pos"
                ] = original_pos  # recover original position, in case that we want to calculate diffusion loss and sampling RMSD at the same time in validation, and for subsequent sampling
            rmsds = torch.cat([rmsd.unsqueeze(-1) for rmsd in rmsds], dim=-1)

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
            clean_mask, aa_mask, padding_mask, batched_data["is_protein"], time_step
        )

        # Protein atoms are not noised
        for i in range(pos.size(0)):
            time_step[i, : batched_data["protein_len"][i]] = (
                torch.rand(1, device=time_step.device) * time_step[i, 0]
            )
            # clean_mask[i, : batched_data["protein_len"][i]] = 1

        if self.psm_config.psm_finetune_mode:
            if self.training:
                noise_mode = self.psm_config.psm_finetune_noise_mode
            else:
                noise_mode = self.psm_config.psm_finetune_valid_noise_mode
            noise_mode = "diffusion"

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

        if self.psm_config.sample_in_validation and not self.training:
            result_dict["rmsd"] = rmsds

        if self.psm_finetune_head:
            result_dict = self.psm_finetune_head(result_dict)
        return result_dict


def center_pos_ligand_only(batched_data, padding_mask):
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

    # center of Ligand atoms only

    protein_mask = torch.zeros_like(batched_data["pos"], dtype=torch.bool)
    for i in range(batched_data["pos"].size(0)):
        protein_mask[i, : batched_data["protein_len"][i]] = True
    non_periodic_center = torch.sum(
        batched_data["pos"].masked_fill(protein_mask | padding_mask.unsqueeze(-1), 0.0),
        dim=1,
    ) / (batched_data["num_atoms"] - batched_data["protein_len"]).unsqueeze(-1)
    center = non_periodic_center.unsqueeze(1)
    center[is_periodic] = periodic_center
    batched_data["pos"] -= center
    batched_data["pos"] = batched_data["pos"].masked_fill(
        padding_mask.unsqueeze(-1), 0.0
    )
    return batched_data["pos"]


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
