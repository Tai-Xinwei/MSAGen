# -*- coding: utf-8 -*-
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from sfm.models.psm.psm_config import DiffusionTimeStepEncoderType, PSMConfig


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(
        self,
        dim,
        max_period=10000,
    ):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0, "SinusoidalPositionEmbeddings requires dim to be even."
        self.max_period = max_period
        self.dummy = nn.Parameter(
            torch.empty(0, dtype=torch.float), requires_grad=False
        )  # to detect fp16

    def forward(self, time):
        device = time.device
        time = time * self.max_period
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = embeddings.to(self.dummy.dtype)
        return embeddings


class TimeStepEncoder(nn.Module):
    def __init__(
        self,
        n_timesteps,
        timestep_emb_dim,
        timestep_emb_type: DiffusionTimeStepEncoderType,
        mlp=True,
    ):
        super(TimeStepEncoder, self).__init__()

        if timestep_emb_type == DiffusionTimeStepEncoderType.POSITIONAL:
            self.time_proj = SinusoidalPositionEmbeddings(timestep_emb_dim)
        elif timestep_emb_type == DiffusionTimeStepEncoderType.DISCRETE_LEARNABLE:
            self.time_proj = nn.Embedding(n_timesteps + 1, timestep_emb_dim)
        else:
            raise NotImplementedError

        if mlp:
            self.time_embedding = nn.Sequential(
                nn.Linear(timestep_emb_dim, timestep_emb_dim),
                nn.GELU(),
                nn.Linear(timestep_emb_dim, timestep_emb_dim),
            )
        else:
            self.time_embedding = None

        self.n_timesteps = n_timesteps
        self.timestep_emb_type = timestep_emb_type

    def forward(self, timesteps, clean_mask: Optional[Tensor]):
        if self.timestep_emb_type == DiffusionTimeStepEncoderType.DISCRETE_LEARNABLE:
            discretized_time_steps = (
                timesteps * self.n_timesteps
            ).long()  # in range [0, n_timesteps - 1]
            if clean_mask is not None:
                discretized_time_steps[
                    clean_mask
                ] = self.n_timesteps  # use last time step embedding for clean samples
            t_emb = self.time_proj(discretized_time_steps)
        elif self.timestep_emb_type == DiffusionTimeStepEncoderType.POSITIONAL:
            if clean_mask is not None:
                timesteps = timesteps.masked_fill(
                    clean_mask, 0.0
                )  # use t = 0 for clean samples with positional time embedding (which is continuous time embedding)
            t_emb = self.time_proj(timesteps)
        else:
            raise ValueError(f"Unkown timestep_emb_type {self.timestep_emb_type}")
        if self.time_embedding is not None:
            t_emb = self.time_embedding(t_emb)

        return t_emb


class TimeStepSampler:
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps

    def sample(self, n_graph, device, dtype, clean_sample_ratio: float = 0.0):
        time_step = torch.rand(size=(n_graph // 2 + 1,), device=device)
        time_step = torch.cat([time_step, 1.0 - time_step], dim=0)[:n_graph]
        time_step = time_step.to(dtype=dtype)
        clean_mask = torch.tensor(
            np.random.rand(n_graph) <= clean_sample_ratio,
            dtype=torch.bool,
            device=device,
        )
        return time_step, clean_mask

    def get_continuous_time_step(self, t, n_graph, device, dtype):
        time_step = torch.zeros(
            [
                n_graph,
            ],
            device=device,
            dtype=dtype,
        )
        time_step = time_step.fill_(t * 1.0 / self.num_timesteps)
        return time_step


class DiffNoise(nn.Module):
    def __init__(self, psm_config: PSMConfig):
        super(DiffNoise, self).__init__()
        self.psm_config = psm_config

        assert psm_config.ddpm_schedule in ["linear", "quadratic", "sigmoid", "cosine"]
        (
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
            self.alphas_cumprod,
            self.beta_list,
        ) = self._beta_schedule(
            psm_config.num_timesteps + 1,
            psm_config.ddpm_beta_start,
            psm_config.ddpm_beta_end,
            psm_config.ddpm_schedule,
        )
        self.unit_noise_scale = psm_config.diffusion_noise_std

    def _beta_schedule(
        self, num_timesteps, beta_start, beta_end, schedule_type="sigmoid"
    ):
        if schedule_type == "linear":
            beta_list = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "quadratic":
            beta_list = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
            )
        elif schedule_type == "sigmoid":
            betas = torch.linspace(-6, 6, num_timesteps)
            beta_list = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        elif schedule_type == "cosine":
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            beta_list = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise NotImplementedError("only support linear, quadratic, sigmoid, cosine")

        alphas = 1 - beta_list
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        return (
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            alphas_cumprod,
            beta_list,
        )

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu().long())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def _noise_lattice_vectors(self, pos, non_atom_mask, noise, is_periodic):
        n_graphs = pos[is_periodic].size()[0]
        device = pos.device
        lattice_corner_noise = (
            torch.randn([n_graphs, 8, 3], device=device, dtype=pos.dtype)
            * self.unit_noise_scale
        )
        corner_noise = lattice_corner_noise[:, 0, :]
        gather_index = (
            torch.tensor([0, 4, 2, 1], device=device, dtype=torch.long)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat([n_graphs, 1, 3])
        )
        lattice_vector_noise = torch.gather(
            lattice_corner_noise, index=gather_index, dim=1
        )[:, 1:, :] - corner_noise.unsqueeze(1)
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
            dtype=pos.dtype,
            device=device,
        )
        corner_noise = torch.matmul(
            cell_matrix, lattice_vector_noise
        ) + corner_noise.unsqueeze(1)
        corner_noise_center = (corner_noise[:, 0, :] + corner_noise[:, -1, :]) / 2.0
        corner_noise -= corner_noise_center.unsqueeze(1)
        num_atoms = torch.sum(~non_atom_mask[is_periodic], dim=-1).long()
        scatter_index = torch.arange(8, device=device).unsqueeze(0).unsqueeze(
            -1
        ).repeat([n_graphs, 1, 3]) + num_atoms.unsqueeze(-1).unsqueeze(-1)
        noise[is_periodic] = noise[is_periodic].scatter(1, scatter_index, corner_noise)
        return noise

    def get_noise(self, pos, non_atom_mask, is_periodic):
        noise = torch.randn_like(pos) * self.unit_noise_scale
        # for non-cell corner nodes (atoms in molecules, amino acids in proteins and atoms in materials)
        # use zero-centered noise
        noise = noise.masked_fill(non_atom_mask.unsqueeze(-1), 0.0)
        noise_center = noise.sum(dim=1, keepdim=True) / torch.sum(
            (~non_atom_mask).long(), dim=-1
        ).unsqueeze(-1).unsqueeze(-1)
        noise[~is_periodic] -= noise_center[~is_periodic]
        noise = noise.masked_fill(non_atom_mask.unsqueeze(-1), 0.0)
        # for cell corner nodes (the noise is centered so that the noised cell is centered at the original point)
        noise = self._noise_lattice_vectors(pos, non_atom_mask, noise, is_periodic)
        return noise

    def get_sampling_start(self, init_pos, non_atom_mask, is_periodic):
        noise = self.get_noise(init_pos, non_atom_mask, is_periodic)
        return init_pos + noise

    def noise_sample(
        self,
        x_start,
        t,
        non_atom_mask: Tensor,
        is_periodic: Tensor,
        x_init=None,
        clean_mask: Optional[Tensor] = None,
    ):
        t = (t * self.psm_config.num_timesteps).long()
        noise = self.get_noise(x_start, non_atom_mask, is_periodic)

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
            x_t = torch.where(clean_mask.unsqueeze(-1).unsqueeze(-1), x_start, x_t)

        return x_t, noise, sqrt_one_minus_alphas_cumprod_t

    # Here, t is time point in (0, 1]
    def _angle_noise_sample(self, x_start, t):
        T = t
        # T = t / 1000.0
        T = T.unsqueeze(-1).unsqueeze(-1)

        beta_min = 0.01 * torch.pi
        beta_max = 1.0 * torch.pi

        sigma = beta_min ** (1 - T) * beta_max**T  # SMLD (31)

        noise = torch.randn_like(x_start) * sigma

        return x_start + noise, noise, sigma


if __name__ == "__main__":
    n_timesteps = 10
    timestep_emb_dim = 4
    # === Test SinusoidalPositionEmbeddings ===
    timestep_emb_type = "positional"
    model = TimeStepEncoder(n_timesteps, timestep_emb_dim, timestep_emb_type, mlp=True)
    print(model)
    timesteps = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    print(model(timesteps))

    # === Test learnable embeddings ===
    timestep_emb_type = "learnable"
    model = TimeStepEncoder(n_timesteps, timestep_emb_dim, timestep_emb_type, mlp=False)
    print(model)
    timesteps = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    print(model(timesteps))
