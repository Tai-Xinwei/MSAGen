# -*- coding: utf-8 -*-
# Copyright (c) Mircrosoft.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
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
from sfm.models.psm.psm_config import PSMConfig
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model

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
        load_ckpt=False,
    ):
        """
        Initialize the PSMModel class.

        Args:
            args: Command line arguments.
            loss_fn: The loss function to use.
            data_mean: The mean of the label. For label normalization.
            data_std: The standard deviation of the label. For label normalization.
            not_init: If True, the model will not be initialized. Default is False.
            load_ckpt: If True, the model will load a checkpoint. Default is False.
        """

        super().__init__()
        if not_init:
            return

        self.psm_config = PSMConfig(args)
        self.args = self.psm_config.args
        if args.rank == 0:
            logger.info(self.args)

        self.net = PSM(args, self.psm_config)

        if load_ckpt:
            self.load_pretrained_weights(args, checkpoint_path=args.loadcheck_path)
        else:
            logger.info("No checkpoint is loaded")

        self.diffnoise = DiffNoise(self.psm_config)
        self.diffusion_process = DIFFUSION_PROCESS_REGISTER[
            self.psm_config.diffusion_sampling
        ](self.diffnoise.alphas_cumprod)

        self.time_step_sampler = TimeStepSampler(self.psm_config.num_timesteps)

        self.loss_fn = loss_fn(args)

        if self.psm_config.sample_in_validation:
            self.sampled_structure_converter = SampledStructureConverter(
                self.psm_config.sampled_structure_output_path
            )

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
        batched_data["protein_masked_aa"] = (
            torch.rand_like(token_id, dtype=torch.float) < self.psm_config.mask_ratio
        )

        masked_pos = batched_data["protein_masked_pos"]
        # masked_aa = (
        #     batched_data["protein_masked_aa"].unsqueeze(-1).expand_as(masked_pos)
        # )
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

    def _create_system_tags(self, batched_data):
        token_id = batched_data["token_id"]
        is_periodic = batched_data["pbc"].any(dim=-1)
        is_molecule = (~is_periodic) & (token_id <= 129).all(dim=-1)
        is_protein = (~is_periodic) & (token_id > 129).any(dim=-1)
        batched_data["is_periodic"] = is_periodic
        batched_data["is_molecule"] = is_molecule
        batched_data["is_protein"] = is_protein

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

        noise_pos, noise, _ = self.diffnoise.noise_sample(
            x_start=ori_pos,
            t=time_step,
            non_atom_mask=batched_data["non_atom_mask"],
            is_periodic=batched_data["is_periodic"],
            x_init=batched_data["init_pos"],
            clean_mask=clean_mask,
        )
        noise_pos = complete_cell(noise_pos, batched_data)

        return noise_pos, noise

    def load_pretrained_weights(self, args, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.

        Args:
            args: Command line arguments.
            checkpoint_path: Path to the pretrained weights.
        """
        if args.ft or args.infer:
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
        else:
            logger.info("No checkpoint is loaded")

    def max_positions(self):
        """
        Returns the maximum positions of the net.
        """
        return self.net.max_positions

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

        token_id = batched_data["token_id"]
        padding_mask = token_id.eq(0)  # B x T x 1
        aa_mask = batched_data["protein_masked_aa"] & batched_data[
            "is_protein"
        ].unsqueeze(-1)
        aa_mask = aa_mask & ~padding_mask

        pos, noise = self._set_noise(
            padding_mask=padding_mask,
            batched_data=batched_data,
            time_step=time_step,
            clean_mask=clean_mask,
        )
        batched_data["pos"] = pos
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

        if self.psm_config.sample_in_validation and not self.training:
            result_dict["rmsd"] = rmsds

        return result_dict

    def ft_forward(self, batched_data, **kwargs):
        """
        Forward pass of the model during fine-tuning.

        Args:
            batched_data: Input data for the forward pass.
            **kwargs: Additional keyword arguments.
        """
        return self.net.ft_forward(batched_data, **kwargs)

    def compute_loss(self, model_output, batched_data) -> ModelOutput:
        """
        Compute loss for the model.

        Args:
            model_output: The output from the model.
            batched_data: The batch data.

        Returns:
            ModelOutput: The model output which includes loss, log_output, num_examples.
        """
        # bs = batched_data["token_id"].eq(158).sum().item()
        bs = batched_data["pos"].size(0)
        loss, logging_output = self.loss_fn(model_output, batched_data)
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
        batched_data["pos"] = complete_cell(
            batched_data["pos"], batched_data, is_sampling=True
        )
        batched_data["pos"] = center_pos(
            batched_data, padding_mask=padding_mask
        )  # centering to remove noise translation

        for t in tqdm(range(self.psm_config.num_timesteps - 1, -1, -1)):
            # forward
            time_step = self.time_step_sampler.get_continuous_time_step(
                t, n_graphs, device=device, dtype=batched_data["pos"].dtype
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
            batched_data["pos"] = complete_cell(
                batched_data["pos"], batched_data, is_sampling=True
            )
            batched_data["pos"] = center_pos(
                batched_data, padding_mask=padding_mask
            )  # centering to remove noise translation
            batched_data["pos"] = batched_data["pos"].detach()

        pred_pos = batched_data["pos"].clone()

        loss = torch.sum((pred_pos - orig_pos) ** 2, dim=-1, keepdim=True)

        return {"loss": loss, "pred_pos": pred_pos, "orig_pos": orig_pos}

    @torch.no_grad()
    def seq2structure(self, aa_seq, dtype: torch.dtype = torch.float16) -> dict:
        """
        Given an amino acid sequence, predict the 3D structure of the protein.
        """
        batched_data = {}
        N, L = aa_seq.shape

        batched_data["sample_type"] = 2
        batched_data["token_type"] = aa_seq
        batched_data["idx"] = 0

        batched_data["coords"] = torch.randn(N, L, 3, device=aa_seq.device, dtype=dtype)
        batched_data["num_atoms"] = aa_seq.size()[0]

        batched_data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        batched_data["pbc"] = torch.zeros(3, dtype=torch.float64).bool()
        batched_data["stress"] = torch.zeros(
            (3, 3), dtype=torch.float64, device=aa_seq.device
        )
        batched_data["forces"] = torch.zeros(
            (aa_seq.size()[0], 3), dtype=torch.float64, device=aa_seq.device
        )
        batched_data["energy"] = torch.tensor(
            [0.0], dtype=torch.float64, device=aa_seq.device
        )
        batched_data["energy_per_atom"] = torch.tensor(
            [0.0], dtype=torch.float64, device=aa_seq.device
        )
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.zeros([2, 0], dtype=torch.long)
        edge_attr = torch.zeros([0, 3], dtype=torch.long)
        indgree = adj.long().sum(dim=1).view(-1)

        batched_data["edge_index"] = edge_index
        batched_data["edge_attr"] = edge_attr
        batched_data["node_attr"] = torch.cat(
            [
                batched_data["token_type"].unsqueeze(-1),
                torch.zeros(
                    [batched_data["token_type"].size()[0], 8], dtype=torch.long
                ),
            ],
            dim=-1,
        )
        batched_data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        batched_data["in_degree"] = indgree

        shortest_path_result = (
            torch.full(adj.size(), 511, dtype=torch.long).cpu().numpy()
        )
        edge_input = torch.zeros([N, N, 0, 3], dtype=torch.long)
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        batched_data["edge_input"] = edge_input
        batched_data["spatial_pos"] = spatial_pos

        # {"loss": loss, "pred_pos": pred_pos, "orig_pos": orig_pos}
        # orig_pos would be N(0, 1) noise here, do not use.
        result_dict = self.sample(batched_data)

        return result_dict


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


def complete_cell(pos, batched_data, is_sampling=False):
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

    if is_sampling:
        corner = torch.gather(periodic_pos, 1, index=gather_index)[:, 0, :].unsqueeze(1)
        inverse_lattice = torch.inverse(batched_data["cell"][periodic_mask])
        frac_coords = torch.matmul(pos[periodic_mask] - corner, inverse_lattice) % 1.0
        pos[periodic_mask] = torch.where(
            batched_data["non_atom_mask"][periodic_mask].unsqueeze(-1),
            pos[periodic_mask],
            torch.matmul(frac_coords, batched_data["cell"][periodic_mask]) + corner,
        )

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

        # Implement the embedding
        if args.backbone == "vanillatransformer":
            self.embedding = PSMMix3dEmbedding(psm_config)
            # self.embedding = PSMMixEmbedding(psm_config)
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
            # Implement the decoder
            self.decoder = EquivariantDecoder(psm_config)
        elif args.backbone == "vanillatransformer":
            # Implement the encoder
            self.encoder = PSMPlainEncoder(args, psm_config)
            # Implement the decoder
            # self.decoder = EquivariantDecoder(psm_config)
            self.decoder = NodeTaskHead(psm_config)
        else:
            raise NotImplementedError

        # simple energy, force and noise prediction heads
        self.energy_head = nn.ModuleDict()
        self.forces_head = nn.ModuleDict()
        self.noise_head = nn.ModuleDict()

        for key in {"molecule", "periodic", "protein"}:
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

            if args.backbone == "vanillatransformer":
                self.noise_head.update({key: VectorOutput(psm_config.embedding_dim)})
                self.forces_head.update({key: VectorOutput(psm_config.embedding_dim)})
            else:
                self.noise_head.update(
                    {key: EquivariantVectorOutput(psm_config.embedding_dim)}
                )
                self.forces_head.update(
                    {key: EquivariantVectorOutput(psm_config.embedding_dim)}
                )
        # aa mask predict head
        self.aa_mask_head = nn.Sequential(
            nn.Linear(psm_config.embedding_dim, psm_config.embedding_dim, bias=False),
            nn.SiLU(),
            nn.Linear(psm_config.embedding_dim, 160, bias=False),
        )

    def _set_mask(self, mask_aa, mask_pos, residue_seq):
        """
        set mask here
        """
        pass

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
        n_graphs, n_nodes = pos.size()[:2]
        is_periodic = batched_data["is_periodic"]
        is_molecule = batched_data["is_molecule"]
        is_protein = batched_data["is_protein"]
        # B, L, H is Batch, Length, Hidden
        # token_embedding: B x L x H
        # padding_mask: B x L
        # token_type: B x L  (0 is used for PADDING)
        token_embedding, padding_mask, token_type = self.embedding(
            batched_data, time_step, clean_mask, aa_mask
        )
        # for invariant model struct, we first used encoder to get invariant feature
        # then used equivariant decoder to get equivariant output: like force, noise.
        if self.args.backbone == "vanillatransformer":
            (
                encoder_output,
                pbc_expand_batched,
            ) = self.encoder(  # CL: expand cell outside encoder?
                token_embedding.transpose(0, 1), padding_mask, batched_data, token_type
            )
            decoder_x_output, decoder_vec_output = self.decoder(
                batched_data,
                encoder_output,
                padding_mask,
                pbc_expand_batched,
            )
        elif self.encoder is not None:
            (
                encoder_output,
                pbc_expand_batched,
            ) = self.encoder(  # CL: expand cell outside encoder?
                token_embedding.transpose(0, 1), padding_mask, batched_data, token_type
            )

            decoder_x_output, decoder_vec_output = self.decoder(
                batched_data,
                encoder_output,
                padding_mask,
                pbc_expand_batched,
            )
        else:
            decoder_x_output, decoder_vec_output = self.decoder(
                batched_data,
                token_embedding.transpose(0, 1),
                padding_mask,
                pbc_expand_batched=None,
            )

        energy_per_atom = torch.where(
            is_periodic.unsqueeze(-1),
            self.energy_head["periodic"](decoder_x_output).squeeze(-1),
            self.energy_head["molecule"](decoder_x_output).squeeze(-1),
        )

        forces = torch.where(
            is_periodic.unsqueeze(-1).unsqueeze(-1),
            self.forces_head["periodic"](decoder_x_output, decoder_vec_output).squeeze(
                -1
            ),
            self.forces_head["molecule"](decoder_x_output, decoder_vec_output).squeeze(
                -1
            ),
        )

        noise_pred = torch.where(
            is_periodic.unsqueeze(-1).unsqueeze(-1),
            self.noise_head["periodic"](decoder_x_output, decoder_vec_output).squeeze(
                -1
            ),
            self.noise_head["molecule"](decoder_x_output, decoder_vec_output).squeeze(
                -1
            ),
        )
        noise_pred = torch.where(
            is_protein.unsqueeze(-1).unsqueeze(-1),
            self.noise_head["protein"](decoder_x_output, decoder_vec_output).squeeze(
                -1
            ),
            noise_pred,
        )

        # atom mask to leave out unit cell corners for periodic systems
        non_atom_mask = torch.arange(
            n_nodes, dtype=torch.long, device=energy_per_atom.device
        ).unsqueeze(0).repeat(n_graphs, 1) >= batched_data["num_atoms"].unsqueeze(-1)

        # per-atom energy prediction
        energy_per_atom = (
            energy_per_atom.masked_fill(non_atom_mask, 0.0).sum(dim=-1)
            / batched_data["num_atoms"]
        )

        if self.encoder is not None:
            aa_logits = self.aa_mask_head(encoder_output.transpose(0, 1))
        else:
            aa_logits = self.aa_mask_head(decoder_x_output)  # @Peiran

        return {
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
        }

    def ft_forward(
        self,
        batched_data,
        mode="T_noise",
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
        forward function used in finetuning
        """
        pass

    def init_state_dict_weight(self, weight, bias):
        """
        Initialize the state dict weight.
        """
        pass
