# -*- coding: utf-8 -*-
import math

import numpy as np
import torch
import torch.nn as nn

from sfm.logging import logger


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
    def __init__(self, n_timesteps, timestep_emb_dim, timestep_emb_type, mlp=True):
        super(TimeStepEncoder, self).__init__()

        if timestep_emb_type == "positional":
            self.time_proj = SinusoidalPositionEmbeddings(timestep_emb_dim)
        elif timestep_emb_type == "learnable":
            self.time_proj = nn.Embedding(n_timesteps, timestep_emb_dim)
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

    def forward(self, timesteps):
        t_emb = self.time_proj(timesteps)
        if self.time_embedding is not None:
            t_emb = self.time_embedding(t_emb)
        return t_emb


class DiffNoise(nn.Module):
    def __init__(self, args):
        super(DiffNoise, self).__init__()
        self.args = args

        assert args.ddpm_schedule in ["linear", "quadratic", "sigmoid", "cosine"]
        (
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
            self.alphas_cumprod,
            self.beta_list,
        ) = self._beta_schedule(
            args.num_timesteps + 1,
            args.ddpm_beta_start,
            args.ddpm_beta_end,
            args.ddpm_schedule,
        )

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

    def _noise_sample(self, x_start, t, unit_noise_scale=1.0):
        t = (t * self.args.num_timesteps).long()
        noise = torch.randn_like(x_start) * unit_noise_scale

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        noise = x_t - x_start

        return x_t, noise, sqrt_one_minus_alphas_cumprod_t

    # Here, t is time point in (0, 1]
    def _angle_noise_sample(self, x_start, t):
        T = t
        # T = t / 1000.0
        T = T.unsqueeze(-1).unsqueeze(-1)

        beta_min = 0.01 * torch.pi
        beta_max = 1.0 * torch.pi

        sigma = beta_min ** (1 - T) * beta_max**T  # SMLD (31)

        epsilon = torch.randn_like(x_start)

        noise = epsilon * sigma

        return x_start + noise, noise, sigma, epsilon


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
