# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn


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
