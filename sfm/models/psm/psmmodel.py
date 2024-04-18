# -*- coding: utf-8 -*-
# Copyright (c) Mircrosoft.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from tqdm import tqdm

from sfm.logging import logger
from sfm.models.psm.equivariant.equivariant import EquivariantDecoder
from sfm.models.psm.invariant.invariant_encoder import PSMEncoder
from sfm.models.psm.modules.embedding import PSMMixEmbedding
from sfm.models.psm.psm_config import DiffusionTrainingLoss, PSMConfig
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model

from .modules.timestep_encoder import DiffNoise, TimeStepSampler


class PSMModel(Model):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(
        self,
        args,
        data_mean=0.0,
        data_std=1.0,
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

        # Implement the Diffusion noise
        self.diffnoise = DiffNoise(self.psm_config)

        self.time_step_sampler = TimeStepSampler(self.psm_config.num_timesteps)

        self.loss_fn = loss_fn(args)

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
        masked_aa = (
            batched_data["protein_masked_aa"].unsqueeze(-1).expand_as(masked_pos)
        )
        masked_protein = (
            ((token_id > 129) & (token_id < 156))
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
        mask = masked_protein & (masked_pos | masked_aa | masked_nan | masked_inf)
        batched_data["protein_mask"] = mask

    def _create_system_tags(self, batched_data):
        token_id = batched_data["token_id"]
        is_periodic = batched_data["pbc"].any(dim=-1)
        is_molecule = (~is_periodic) & (token_id <= 129).all(dim=-1)
        is_protein = (~is_periodic) & (token_id > 129).any(dim=-1)
        batched_data["is_periodic"] = is_periodic
        batched_data["is_molecule"] = is_molecule
        batched_data["is_protein"] = is_protein

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
            ori_pos,
            time_step,
            x_init=batched_data["init_pos"],
            clean_mask=clean_mask,
            unit_noise_scale=self.psm_config.diffusion_noise_std,
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

        self._create_system_tags(batched_data)
        self._create_protein_mask(batched_data)
        pos = batched_data["pos"]
        n_graphs = pos.size(0)
        time_step, clean_mask = self.time_step_sampler.sample(
            n_graphs, pos.device, pos.dtype, self.psm_config.clean_sample_ratio
        )
        clean_mask &= ~batched_data[
            "is_protein"
        ]  # Proteins are always corrupted. For proteins, we only consider diffusion training on structure for now.
        aa_mask = batched_data["protein_masked_aa"] & batched_data[
            "is_protein"
        ].unsqueeze(-1)

        token_id = batched_data["token_id"]
        padding_mask = token_id.eq(0)  # B x T x 1

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

        pos_noise = torch.zeros(size=orig_pos.size(), device=device)
        pos_noise = pos_noise.normal_() * self.psm_config.diffusion_noise_std

        batched_data["pos"] = pos_noise + batched_data["init_pos"]
        batched_data["pos"] = batched_data["pos"].masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        batched_data["pos"] = complete_cell(batched_data["pos"], batched_data)

        if self.psm_config.diffusion_sampling == "ddpm":
            # Sampling from Step T-1 to Step 0
            for t in tqdm(range(self.psm_config.num_timesteps - 1, -1, -1)):
                hat_alpha_t = self.diffnoise.alphas_cumprod[t]
                hat_alpha_t_1 = 1.0 if t == 0 else self.diffnoise.alphas_cumprod[t - 1]
                alpha_t = (
                    hat_alpha_t / hat_alpha_t_1
                )  # CL: can we get the following three from `diffnoise`?
                beta_t = 1 - alpha_t
                sigma_t = (
                    0.0
                    if t == 0
                    else (
                        (1.0 - hat_alpha_t_1) / (1.0 - hat_alpha_t) * beta_t
                    ).sqrt()  # CL: I suggest calling it `beta_tilde_t`, and spare `sigma_t` for `(1 - hat_alpha_t).sqrt()`
                )

                # forward
                time_step = self.time_step_sampler.get_continuous_time_step(
                    t, n_graphs, device=device, dtype=batched_data["pos"].dtype
                )
                noise = self.net(batched_data, time_step=time_step)[
                    "noise_pred"
                ]  # CL: use `no_grad`?
                noise = noise.detach()

                epsilon = (
                    torch.zeros_like(batched_data["pos"]).normal_()
                    * self.psm_config.diffusion_noise_std
                )

                ext_pos = (
                    batched_data["pos"]
                    - batched_data["init_pos"]
                    - (1 - alpha_t) / (1 - hat_alpha_t).sqrt() * noise
                ) / alpha_t.sqrt() + sigma_t * epsilon

                batched_data["pos"] = ext_pos + batched_data["init_pos"]

                batched_data["pos"] = complete_cell(batched_data["pos"], batched_data)
                batched_data["pos"] = batched_data["pos"].detach()
                batched_data["pos"] = batched_data["pos"].masked_fill(
                    ~padding_mask.unsqueeze(-1), 0.0
                )
        elif self.psm_config.diffusion_sampling == "ddim":
            sampled_steps, _ = torch.sort(
                (
                    torch.randperm(
                        self.psm_config.num_timesteps - 2,
                        dtype=torch.long,
                        device=device,
                    )
                    + 1
                )[
                    : self.psm_config.ddim_steps - 1
                ]  # CL: be careful if using a different number of steps `psm_config.ddim_steps`!
            )
            sampled_steps = torch.cat(
                [
                    sampled_steps,
                    torch.tensor(
                        [self.psm_config.num_timesteps - 1], device=device
                    ).long(),
                ]
            )
            for i in tqdm(range(sampled_steps.shape[0] - 1, 0, -1)):
                t = sampled_steps[i]
                t_1 = sampled_steps[i - 1]
                hat_alpha_t = self.diffnoise.alphas_cumprod[
                    t
                ]  # CL: these quantities should be re-calculated.
                hat_alpha_t_1 = self.diffnoise.alphas_cumprod[t_1]
                alpha_t = hat_alpha_t / hat_alpha_t_1
                beta_t = 1.0 - alpha_t
                sigma_t = (  # CL: for DDIM, this sigma_t can be set to zero.
                    self.psm_config.ddim_eta
                    * ((1.0 - hat_alpha_t_1) / (1.0 - hat_alpha_t) * beta_t).sqrt()
                )

                # forward
                time_step = self.time_step_sampler.get_continuous_time_step(
                    t, n_graphs, device=device, dtype=batched_data["pos"].dtype
                )
                noise = self.net(batched_data, time_step=time_step)["noise_pred"]
                ext_pos = batched_data["pos"] - batched_data["init_cell"]
                x_0_pred = (
                    ext_pos - (1.0 - hat_alpha_t).sqrt() * noise
                ) / hat_alpha_t.sqrt()
                epsilon = (
                    torch.zeros_like(ext_pos).normal_()
                    * self.psm_config.diffusion_noise_std
                )
                ext_pos = (
                    hat_alpha_t_1.sqrt() * x_0_pred
                    + (1.0 - hat_alpha_t_1 - sigma_t**2).sqrt() * noise
                    + sigma_t * epsilon
                )
                batched_data["pos"] = ext_pos + batched_data["init_pos"]
                batched_data["pos"] = complete_cell(batched_data["pos"], batched_data)
                batched_data["pos"] = batched_data["pos"].detach()
                batched_data["pos"] = batched_data["pos"].masked_fill(
                    ~padding_mask.unsqueeze(-1), 0.0
                )

            # forward for last step
            t = sampled_steps[0]
            hat_alpha_t = self.diffnoise.alphas_cumprod[t]

            # forward
            time_step = self.time_step_sampler.get_continuous_time_step(
                t, n_graphs, device=device, dtype=batched_data["pos"].dtype
            )
            noise = self.net(batched_data, time_step=time_step)[
                "noise_pred"
            ]  # CL: why additional forward?
            ext_pos = batched_data["pos"] - batched_data["init_pos"]
            x_0_pred = (
                ext_pos - (1.0 - hat_alpha_t).sqrt() * noise
            ) / hat_alpha_t.sqrt()
            batched_data["pos"] = x_0_pred + batched_data["init_pos"]
            batched_data["pos"] = complete_cell(batched_data["pos"], batched_data)
            batched_data["pos"] = batched_data["pos"].detach()
            batched_data["pos"] = batched_data["pos"].masked_fill(
                ~padding_mask.unsqueeze(-1), 0.0
            )
        elif self.psm_config.diffusion_sampling == "ode":
            lattice_scatter_index = torch.tensor(
                [[[4], [2], [1]]], device=device, dtype=torch.long
            ).repeat([n_graphs, 1, 3])
            lattice_scatter_index += (
                batched_data["num_atoms"].unsqueeze(-1).unsqueeze(-1)
            )  # CL: remove these?
            for t in tqdm(range(self.psm_config.num_timesteps - 1, -1, -1)):
                time_step = self.time_step_sampler.get_continuous_time_step(
                    t, n_graphs, device=device, dtype=batched_data["pos"].dtype
                )
                beta_t = self.diffnoise.beta_list[t]
                noise = self.net(batched_data, time_step=time_step)["noise_pred"]
                score = -noise / (1.0 - self.diffnoise.alphas_cumprod[t]).sqrt()
                score = score.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
                ext_pos = batched_data["pos"].clone() - batched_data["init_pos"]
                epsilon = torch.zeros_like(ext_pos).normal_()

                batched_data["pos"] = (
                    (2 - (1.0 - beta_t).sqrt()) * ext_pos
                    + 0.5 * beta_t * (score)
                    + batched_data["init_pos"]
                )
                batched_data["pos"] = complete_cell(batched_data["pos"], batched_data)
                batched_data["pos"] = batched_data["pos"].detach()
                batched_data["pos"] = batched_data["pos"].masked_fill(
                    ~padding_mask.unsqueeze(-1), 0.0
                )
        else:
            raise ValueError(
                f"Unknown diffusion sampling strategy {self.psm_config.diffusion_sampling}. Support only ddim and ddpm."
            )

        pred_pos = batched_data["pos"].clone()

        loss = torch.sum((pred_pos - orig_pos) ** 2, dim=-1, keepdim=True)

        return loss


def center_pos(batched_data, padding_mask):
    # get center of system positions
    periodic_mask = torch.any(batched_data["pbc"], dim=-1)  # B x 3 -> B
    periodic_center = torch.sum(batched_data["cell"], dim=1) / 2.0
    protein_mask = batched_data["protein_mask"]
    non_periodic_center = torch.sum(
        batched_data["pos"].masked_fill(padding_mask.unsqueeze(-1) | protein_mask, 0.0),
        dim=1,
    ) / batched_data["num_atoms"].unsqueeze(-1)
    center = torch.where(
        periodic_mask.unsqueeze(-1), periodic_center, non_periodic_center
    )
    batched_data["pos"] -= center.unsqueeze(1)
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
    periodic_pos = periodic_pos.scatter(1, scatter_index, cell)
    pos[periodic_mask] = periodic_pos
    return pos


class PSM(nn.Module):
    """
    Class for training Physics science module
    """

    def __init__(self, args, psm_config: PSMConfig):
        super().__init__()
        self.max_positions = args.max_positions
        self.args = args

        # Implement the embedding
        self.embedding = PSMMixEmbedding(psm_config)

        # Implement the encoder
        self.encoder = PSMEncoder(args, psm_config)

        # Implement the decoder
        self.decoder = EquivariantDecoder(psm_config)

        # simple energy, force and noise prediction heads
        self.molecule_energy_head = nn.Sequential(
            nn.Linear(psm_config.embedding_dim, psm_config.embedding_dim, bias=True),
            nn.SiLU(),
            nn.Linear(psm_config.embedding_dim, 1, bias=True),
        )

        self.periodic_energy_head = nn.Sequential(
            nn.Linear(psm_config.embedding_dim, psm_config.embedding_dim, bias=True),
            nn.SiLU(),
            nn.Linear(psm_config.embedding_dim, 1, bias=True),
        )

        # TODO: more careful prediction head design
        self.molecule_force_head = nn.Linear(psm_config.embedding_dim, 1, bias=False)
        self.periodic_force_head = nn.Linear(psm_config.embedding_dim, 1, bias=False)

        self.molecule_noise_head = nn.Linear(psm_config.embedding_dim, 1, bias=False)
        self.periodic_noise_head = nn.Linear(psm_config.embedding_dim, 1, bias=False)
        self.protein_noise_head = nn.Linear(psm_config.embedding_dim, 1, bias=False)

        # aa mask predict head
        self.aa_mask_head = nn.Linear(psm_config.embedding_dim, 160, bias=False)

        self.psm_config = psm_config

    def _set_mask(self, mask_aa, mask_pos, residue_seq):
        """
        set mask here
        """
        pass

    def forward(
        self,
        batched_data,
        perturb=None,
        time_step=None,
        clean_mask=None,
        aa_mask=None,
        q=None,  # for computing the score model on the q
        q_0=None,
        delta_tq=None,  # for computing the score model on the q at time_pos + delta_tq
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

        token_embedding, padding_mask, token_type = self.embedding(
            batched_data, time_step, clean_mask, aa_mask
        )

        (
            encoder_output,
            pbc_expand_batched,
        ) = self.encoder(  # CL: expand cell outside encoder?
            token_embedding.transpose(0, 1), padding_mask, batched_data, token_type
        )

        decoder_x_output, decoder_vec_output = self.decoder(
            batched_data,
            encoder_output,
            batched_data["pos"],
            padding_mask,
            pbc_expand_batched,
        )

        # atom-wise energy prediction
        molecule_energy = self.molecule_energy_head(decoder_x_output).squeeze(-1)
        periodic_energy = self.periodic_energy_head(decoder_x_output).squeeze(-1)
        energy = torch.where(
            is_periodic.unsqueeze(-1), periodic_energy, molecule_energy
        )

        molecule_force = self.molecule_force_head(decoder_vec_output).squeeze(-1)
        periodic_force = self.periodic_force_head(decoder_vec_output).squeeze(-1)
        forces = torch.where(
            is_periodic.unsqueeze(-1).unsqueeze(-1), periodic_force, molecule_force
        )

        molecule_noise_pred = self.molecule_noise_head(decoder_vec_output).squeeze(-1)
        periodic_noise_pred = self.periodic_noise_head(decoder_vec_output).squeeze(-1)
        protein_noise_pred = self.protein_noise_head(decoder_vec_output).squeeze(-1)
        noise_pred = torch.where(
            is_periodic.unsqueeze(-1).unsqueeze(-1),
            periodic_noise_pred,
            molecule_noise_pred,
        )
        noise_pred = torch.where(
            is_protein.unsqueeze(-1).unsqueeze(-1), protein_noise_pred, noise_pred
        )

        # atom mask to leave out unit cell corners for periodic systems
        non_atom_mask = torch.arange(
            n_nodes, dtype=torch.long, device=energy.device
        ).unsqueeze(0).repeat(n_graphs, 1) >= batched_data["num_atoms"].unsqueeze(-1)

        # per-atom energy prediction
        energy = (
            molecule_energy.masked_fill(non_atom_mask, 0.0).sum(dim=-1)
            / batched_data["num_atoms"]
        )

        aa_logits = self.aa_mask_head(encoder_output.transpose(0, 1))

        return {
            "energy": energy,
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
