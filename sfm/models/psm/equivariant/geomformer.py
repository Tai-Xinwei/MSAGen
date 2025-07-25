# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from sfm.models.psm.psm_config import PSMConfig, VecInitApproach
from sfm.modules.rotary_embedding import SFMRotaryEmbedding

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class NodeGaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class_mapping = {
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(-2) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


class EquivariantVectorOutput(nn.Module):
    def __init__(self, hidden_channels=768, activation="silu"):
        super(EquivariantVectorOutput, self).__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1, activation=activation),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze(-1)


class EquivariantLayerNorm(nn.Module):
    r"""Rotationally-equivariant Vector Layer Normalization
    Expects inputs with shape (N, n, d), where N is batch size, n is vector dimension, d is width/number of vectors.
    """
    __constants__ = ["normalized_shape", "elementwise_linear"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_linear: bool

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-3,
        elementwise_linear: bool = True,
        device=None,
        dtype=None,
        use_smooth_norm=False,
        add_vec_bias=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(EquivariantLayerNorm, self).__init__()

        self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.elementwise_linear = elementwise_linear
        if self.elementwise_linear:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter(
                "weight", None
            )  # Without bias term to preserve equivariance!

        self.reset_parameters()
        self.use_smooth_norm = use_smooth_norm
        if self.use_smooth_norm and add_vec_bias:
            self.eps = self.eps**2

    def reset_parameters(self) -> None:
        if self.elementwise_linear:
            nn.init.ones_(self.weight)

    def mean_center(self, input):
        return input - input.mean(-1, keepdim=True)

    def covariance(self, input):
        return 1 / self.normalized_shape[0] * input @ input.transpose(-1, -2)

    def symsqrtinv(self, matrix):
        """Compute the inverse square root of a positive definite matrix.
        Based on https://github.com/pytorch/pytorch/issues/25481
        """
        _, s, v = matrix.svd()
        good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]
            v = v[..., :common]
            if unbalanced:
                good = good[..., :common]
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
        return (v * 1 / torch.sqrt(s + self.eps).unsqueeze(-2)) @ v.transpose(-2, -1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.float64)  # Need double precision for accurate inversion.
        if self.use_smooth_norm:
            # input B x N x rank_num x d
            input_norm = input.pow(2).sum(dim=-2, keepdim=True)  # B x N x 1 x C

            input_norm = torch.mean(input_norm, dim=-1)  # B x N x 1
            input_norm = (input_norm + self.eps).pow(-0.5)  # B x N x 1

            if self.elementwise_linear:
                input = input * self.weight
            else:
                input = input

            output = input * input_norm.unsqueeze(-1)  # B x N x rank_num x d

            return output.to(self.weight.dtype)
        else:
            input = self.mean_center(input)
            # We use different diagonal elements in case input matrix is approximately zero,
            # in which case all singular values are equal which is problematic for backprop.
            # See e.g. https://pytorch.org/docs/stable/generated/torch.svd.html
            reg_matrix = (
                torch.diag(torch.tensor([1.0, 2.0, 3.0]))
                .unsqueeze(0)
                .to(input.device)
                .type(input.dtype)
            )
            covar = self.covariance(input) + self.eps * reg_matrix
            covar_sqrtinv = self.symsqrtinv(covar)
            return (covar_sqrtinv @ input).to(self.weight.dtype) * self.weight.reshape(
                1, 1, self.normalized_shape[0]
            )

    def extra_repr(self) -> str:
        return "{normalized_shape}, " "elementwise_linear={elementwise_linear}".format(
            **self.__dict__
        )


class MemEffInvariantAttention(nn.Module):
    def __init__(
        self,
        hidden_channels,
        head_dim,
        dropout,
        d_tilde=1,
        add_rope=True,
        use_smooth_softmax=False,
        smooth_factor=0.0,
        use_no_pre_cutoff_softmax: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.head_dim = head_dim
        self.num_heads = hidden_channels // head_dim

        self.out_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.dropout = dropout
        self.attn_ln = nn.LayerNorm(hidden_channels)
        self.scaling = ((self.head_dim / d_tilde) ** 0.5) / self.head_dim

        self.use_smooth_softmax = use_smooth_softmax
        self.smooth_factor = smooth_factor
        self.use_no_pre_cutoff_softmax = use_no_pre_cutoff_softmax

        self.reset_parameters(1)

        self.rot_emb = None
        if add_rope:
            self.rot_emb = SFMRotaryEmbedding(dim=self.head_dim)

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        # self.out_proj.bias.data.fill_(0)

    def forward(
        self,
        q,
        k,
        v,
        attn_bias,
        key_padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
        is_protein: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ):
        if pbc_expand_batched is not None:
            outcell_index = (
                pbc_expand_batched["outcell_index"]
                .unsqueeze(-1)
                .repeat(1, 1, k.size()[-1])
            )
            if pbc_expand_batched["local_attention_weight"] is not None:
                local_attention_weight = pbc_expand_batched[
                    "local_attention_weight"
                ].to(dtype=q.dtype)
            else:
                local_attention_weight = None
            expand_k = torch.gather(k, dim=1, index=outcell_index)
            expand_v = torch.gather(v, dim=1, index=outcell_index)
            k = torch.cat([k, expand_k], dim=1)
            v = torch.cat([v, expand_v], dim=1)
            expand_mask = pbc_expand_batched["expand_mask"]
        else:
            local_attention_weight = None
            expand_mask = None

        bsz, tgt_len, src_len = q.shape[0], q.shape[1], k.shape[1]

        q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        # add rope
        if self.rot_emb is not None and is_protein is not None and is_protein.any():
            q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
            k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
            v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)
            is_protein_expanded = (
                is_protein.unsqueeze(1)
                .repeat(1, self.num_heads, 1)
                .view([bsz * self.num_heads, tgt_len, 1])
            )
            rot_q, rot_k = self.rot_emb(
                q, k, v, position_ids=position_ids, nhead=self.num_heads
            )
            q = torch.where(is_protein_expanded, rot_q, q)
            k = torch.where(is_protein_expanded, rot_k, k)
            q = q.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
            k = k.reshape(bsz, self.num_heads, src_len, self.head_dim)
            v = v.reshape(bsz, self.num_heads, src_len, self.head_dim)

        if key_padding_mask is not None:
            if pbc_expand_batched is not None:
                assert expand_mask is not None
                key_padding_mask = torch.cat([key_padding_mask, expand_mask], dim=1)
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if key_padding_mask is not None:
            if key_padding_mask.bool().any():
                attn_mask = torch.zeros(
                    (bsz, self.num_heads, tgt_len, src_len),
                    device=q.device,
                    dtype=q.dtype,
                )
                if attn_bias is not None:
                    attn_mask += attn_bias.contiguous().view(
                        bsz, self.num_heads, tgt_len, src_len
                    )
                attn_mask = attn_mask.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_mask = None
                if attn_bias is not None:
                    attn_mask = attn_bias.contiguous().view(
                        bsz, self.num_heads, tgt_len, src_len
                    )

        if local_attention_weight is not None or attn_bias is not None:
            attn_weights = q.matmul(
                k.transpose(-1, -2)
            )  # (bsz, num_heads, tgt_len, src_len)

            if attn_bias is not None:
                attn_weights = attn_weights + attn_bias

            if local_attention_weight is not None:
                local_attention_weight = local_attention_weight.to(dtype=q.dtype)
                if self.use_smooth_softmax:
                    attn_weights = (
                        attn_weights + self.smooth_factor
                    ) * local_attention_weight.unsqueeze(1) - self.smooth_factor

                    if key_padding_mask is not None:
                        # don't attend to padding symbols
                        attn_weights = attn_weights.masked_fill(
                            key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                            float("-inf"),
                        )
                else:
                    if not self.use_no_pre_cutoff_softmax:
                        attn_weights = attn_weights.masked_fill(
                            (local_attention_weight <= 1e-5).unsqueeze(1), float("-inf")
                        )

            attn_probs_float = F.dropout(
                F.softmax(attn_weights, dim=-1, dtype=torch.float32),
                self.dropout,
                training=self.training,
            )

            if local_attention_weight is not None:
                attn_probs_float = attn_probs_float * local_attention_weight.unsqueeze(
                    1
                )

            attn_probs = attn_probs_float.type_as(
                v
            )  # (bsz, num_heads, tgt_len, src_len)

            attn = attn_probs.matmul(v)
        else:
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                if attn_mask is not None:
                    attn_mask = attn_mask.to(dtype=q.dtype)
                attn = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout,
                    attn_mask=attn_mask,
                    is_causal=False,
                )

        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.hidden_channels)
        attn = self.attn_ln(attn)
        attn = self.out_proj(attn)

        return attn


class MemEffEquivariantAttention(nn.Module):
    def __init__(
        self,
        hidden_channels,
        head_dim,
        dropout,
        d_tilde=1,
        use_smooth_softmax=False,
        use_smooth_equviariant_norm=False,
        smooth_factor=0.0,
        add_rope=True,
        use_no_pre_cutoff_softmax: bool = False,
        add_vec_bias=False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.head_dim = head_dim
        self.num_heads = hidden_channels // head_dim
        self.out_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.dropout = dropout
        self.attn_ln = EquivariantLayerNorm(
            hidden_channels,
            use_smooth_norm=use_smooth_equviariant_norm,
            add_vec_bias=add_vec_bias,
        )
        self.scaling = ((self.head_dim / (d_tilde * 3)) ** 0.5) / self.head_dim

        self.rot_emb = None
        if add_rope:
            self.rot_emb = SFMRotaryEmbedding(dim=self.head_dim)

        self.use_smooth_softmax = use_smooth_softmax
        self.smooth_factor = smooth_factor
        self.use_no_pre_cutoff_softmax = use_no_pre_cutoff_softmax

        self.reset_parameters(1)

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(d_tilde))

    def forward(
        self,
        q,
        k,
        v,
        attn_bias,
        key_padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
        is_protein: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ):
        if pbc_expand_batched is not None:
            outcell_index = (
                pbc_expand_batched["outcell_index"]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, 1, k.size()[-2], k.size()[-1])
            )
            if pbc_expand_batched["local_attention_weight"] is not None:
                local_attention_weight = pbc_expand_batched[
                    "local_attention_weight"
                ].to(dtype=q.dtype)
            else:
                local_attention_weight = None
            expand_k = torch.gather(k, dim=1, index=outcell_index)
            expand_v = torch.gather(v, dim=1, index=outcell_index)
            k = torch.cat([k, expand_k], dim=1)
            v = torch.cat([v, expand_v], dim=1)
            expand_mask = pbc_expand_batched["expand_mask"]
        else:
            local_attention_weight = None
            expand_mask = None

        bsz, tgt_len, src_len = q.shape[0], q.shape[1], k.shape[1]
        pos_feature_num = q.shape[2]

        if self.rot_emb is not None and is_protein is not None and is_protein.any():
            q = (
                q.reshape(bsz, tgt_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * pos_feature_num * self.num_heads, tgt_len, self.head_dim)
            )
            k = (
                k.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * pos_feature_num * self.num_heads, src_len, self.head_dim)
            )
            v = (
                v.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * pos_feature_num * self.num_heads, src_len, self.head_dim)
            )
            is_protein_expanded = (
                is_protein.unsqueeze(1)
                .repeat(1, pos_feature_num * self.num_heads, 1)
                .view(bsz * pos_feature_num * self.num_heads, tgt_len, 1)
            )
            rot_q, rot_k = self.rot_emb(
                q,
                k,
                v,
                position_ids=position_ids,
                nhead=pos_feature_num * self.num_heads,
            )
            q = torch.where(is_protein_expanded, rot_q, q)
            k = torch.where(is_protein_expanded, rot_k, k)
            q = (
                q.reshape(bsz, pos_feature_num, self.num_heads, tgt_len, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz, self.num_heads, tgt_len, pos_feature_num * self.head_dim)
            )
            k = (
                k.reshape(bsz, pos_feature_num, self.num_heads, src_len, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )
            v = (
                v.reshape(bsz, pos_feature_num, self.num_heads, src_len, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )
        else:
            q = (
                q.reshape(bsz, tgt_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(bsz, self.num_heads, tgt_len, pos_feature_num * self.head_dim)
            )
            k = (
                k.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )
            v = (
                v.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )

        if key_padding_mask is not None:
            if pbc_expand_batched is not None:
                expand_mask = pbc_expand_batched["expand_mask"]
                key_padding_mask = torch.cat([key_padding_mask, expand_mask], dim=-1)
            if key_padding_mask.bool().any():
                attn_mask = torch.zeros(
                    (bsz, self.num_heads, tgt_len, src_len),
                    device=q.device,
                    dtype=q.dtype,
                )
                if attn_bias is not None:
                    attn_mask += attn_bias.contiguous().view(
                        bsz, self.num_heads, tgt_len, src_len
                    )
                attn_mask = attn_mask.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_mask = None
                if attn_bias is not None:
                    attn_mask = attn_bias.contiguous().view(
                        bsz, self.num_heads, tgt_len, src_len
                    )

        if local_attention_weight is not None or attn_bias is not None:
            attn_weights = q.matmul(
                k.transpose(-1, -2)
            )  # (bsz, num_heads, tgt_len, src_len)
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            if local_attention_weight is not None:
                local_attention_weight = local_attention_weight.to(dtype=q.dtype)
                if self.use_smooth_softmax:
                    attn_weights = (
                        attn_weights + self.smooth_factor
                    ) * local_attention_weight.unsqueeze(1) - self.smooth_factor

                    if key_padding_mask is not None:
                        # don't attend to padding symbols
                        attn_weights = attn_weights.masked_fill(
                            key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                            float("-inf"),
                        )
                else:
                    if not self.use_no_pre_cutoff_softmax:
                        attn_weights = attn_weights.masked_fill(
                            (local_attention_weight <= 1e-5).unsqueeze(1), float("-inf")
                        )
            attn_probs_float = F.dropout(
                F.softmax(attn_weights, dim=-1, dtype=torch.float32),
                self.dropout,
                self.training,
            )
            if local_attention_weight is not None:
                attn_probs_float = attn_probs_float * local_attention_weight.unsqueeze(
                    1
                )
            attn_probs = attn_probs_float.type_as(
                v
            )  # (bsz, num_heads, tgt_len, src_len)
            attn = attn_probs.matmul(v)
        else:
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                if attn_mask is not None:
                    attn_mask = attn_mask.to(dtype=q.dtype)
                attn = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout,
                    attn_mask=attn_mask,
                    is_causal=False,
                )

        attn = (
            attn.transpose(1, 2)
            .reshape(bsz, tgt_len, self.num_heads, pos_feature_num, self.head_dim)
            .transpose(2, 3)
            .reshape(bsz, tgt_len, pos_feature_num, self.hidden_channels)
        )

        attn = self.attn_ln(attn)

        attn = self.out_proj(attn)

        return attn


class InvariantAttention(nn.Module):
    def __init__(
        self,
        hidden_channels,
        head_dim,
        dropout,
        d_tilde=1,
        use_linear_bias=False,
        add_rope=True,
        use_smooth_softmax=False,
        smooth_factor=0.0,
        use_no_pre_cutoff_softmax: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.head_dim = head_dim
        self.num_heads = hidden_channels // head_dim
        self.use_linear_bias = use_linear_bias

        self.out_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=use_linear_bias
        )

        self.dropout = dropout
        self.attn_ln = nn.LayerNorm(hidden_channels)
        self.scaling = ((self.head_dim / d_tilde) ** 0.5) / self.head_dim

        self.reset_parameters(1)

        self.rot_emb = None
        if add_rope:
            self.rot_emb = SFMRotaryEmbedding(dim=self.head_dim)

        self.use_smooth_softmax = use_smooth_softmax
        self.smooth_factor = smooth_factor
        self.use_no_pre_cutoff_softmax = use_no_pre_cutoff_softmax

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.use_linear_bias:
            self.out_proj.bias.data.fill_(0)

    def forward(
        self,
        q,
        k,
        v,
        attn_bias,
        key_padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
        is_protein: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ):
        q = q * self.scaling

        if pbc_expand_batched is not None:
            outcell_index = (
                pbc_expand_batched["outcell_index"]
                .unsqueeze(-1)
                .repeat(1, 1, k.size()[-1])
            )
            local_attention_weight = pbc_expand_batched["local_attention_weight"]
            expand_k = torch.gather(k, dim=1, index=outcell_index)
            expand_v = torch.gather(v, dim=1, index=outcell_index)
            k = torch.cat([k, expand_k], dim=1)
            v = torch.cat([v, expand_v], dim=1)
        else:
            local_attention_weight = None

        bsz, tgt_len, src_len = q.shape[0], q.shape[1], k.shape[1]

        q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # add rope
        if self.rot_emb is not None and is_protein is not None and is_protein.any():
            q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
            k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
            v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)

            is_protein_expanded = (
                is_protein.unsqueeze(1)
                .repeat(1, self.num_heads, 1)
                .reshape([bsz * self.num_heads, tgt_len, 1])
            )
            rot_q, rot_k = self.rot_emb(
                q, k, v, position_ids=position_ids, nhead=self.num_heads
            )
            q = torch.where(is_protein_expanded, rot_q, q)
            k = torch.where(is_protein_expanded, rot_k, k)

            q = q.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
            k = k.reshape(bsz, self.num_heads, src_len, self.head_dim)
            v = v.reshape(bsz, self.num_heads, src_len, self.head_dim)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        attn_weights = q.matmul(
            k.transpose(-1, -2)
        )  # (bsz, num_heads, tgt_len, src_len)

        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias

        if local_attention_weight is not None:
            local_attention_weight = local_attention_weight.to(dtype=q.dtype)
            if self.use_smooth_softmax:
                attn_weights = (
                    attn_weights + self.smooth_factor
                ) * local_attention_weight.unsqueeze(1) - self.smooth_factor
            elif self.use_no_pre_cutoff_softmax:
                pass
            else:
                attn_weights = attn_weights.masked_fill(
                    (local_attention_weight <= 1e-5).unsqueeze(1), float("-inf")
                )

        if key_padding_mask is not None:
            if pbc_expand_batched is not None:
                expand_mask = pbc_expand_batched["expand_mask"]
                key_padding_mask = torch.cat([key_padding_mask, expand_mask], dim=-1)
            # don't attend to padding symbols
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

        attn_probs_float = F.dropout(
            F.softmax(attn_weights, dim=-1, dtype=torch.float32),
            self.dropout,
            training=self.training,
        )

        if local_attention_weight is not None:
            attn_probs_float = attn_probs_float * local_attention_weight.unsqueeze(1)

        attn_probs = attn_probs_float.type_as(
            attn_weights
        )  # (bsz, num_heads, tgt_len, src_len)

        attn = attn_probs.matmul(v)  # (bsz, num_heads, tgt_len, head_dim)

        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.hidden_channels)
        attn = self.attn_ln(attn)
        attn = self.out_proj(attn)

        return attn


class EquivariantAttention(nn.Module):
    def __init__(
        self,
        hidden_channels,
        head_dim,
        dropout,
        d_tilde=1,
        use_smooth_equviariant_norm=False,
        use_smooth_softmax=False,
        smooth_factor=0.0,
        add_rope=True,
        use_no_pre_cutoff_softmax: bool = False,
        add_vec_bias=False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.head_dim = head_dim
        self.num_heads = hidden_channels // head_dim
        self.out_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.dropout = dropout
        self.attn_ln = EquivariantLayerNorm(
            hidden_channels,
            use_smooth_norm=use_smooth_equviariant_norm,
            add_vec_bias=add_vec_bias,
        )
        self.scaling = ((self.head_dim / (d_tilde * 3)) ** 0.5) / self.head_dim

        self.reset_parameters(1)

        self.rot_emb = None
        if add_rope:
            self.rot_emb = SFMRotaryEmbedding(dim=self.head_dim)

        self.use_smooth_softmax = use_smooth_softmax
        self.smooth_factor = smooth_factor
        self.use_no_pre_cutoff_softmax = use_no_pre_cutoff_softmax

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(d_tilde))

    def forward(
        self,
        q,
        k,
        v,
        attn_bias,
        key_padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
        is_protein: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ):
        q = q * self.scaling

        if pbc_expand_batched is not None:
            outcell_index = (
                pbc_expand_batched["outcell_index"]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, 1, k.size()[-2], k.size()[-1])
            )
            local_attention_weight = pbc_expand_batched["local_attention_weight"]
            expand_k = torch.gather(k, dim=1, index=outcell_index)
            expand_v = torch.gather(v, dim=1, index=outcell_index)
            k = torch.cat([k, expand_k], dim=1)
            v = torch.cat([v, expand_v], dim=1)
        else:
            local_attention_weight = None

        bsz, tgt_len, src_len = q.shape[0], q.shape[1], k.shape[1]
        pos_feature_num = q.shape[2]

        if self.rot_emb is not None and is_protein is not None and is_protein.any():
            q = (
                q.reshape(bsz, tgt_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * pos_feature_num * self.num_heads, tgt_len, self.head_dim)
            )
            k = (
                k.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * pos_feature_num * self.num_heads, src_len, self.head_dim)
            )
            v = (
                v.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * pos_feature_num * self.num_heads, src_len, self.head_dim)
            )
            is_protein_expanded = (
                is_protein.unsqueeze(1)
                .repeat(1, pos_feature_num * self.num_heads, 1)
                .view(bsz * pos_feature_num * self.num_heads, tgt_len, 1)
            )
            rot_q, rot_k = self.rot_emb(
                q,
                k,
                v,
                position_ids=position_ids,
                nhead=pos_feature_num * self.num_heads,
            )
            q = torch.where(is_protein_expanded, rot_q, q)
            k = torch.where(is_protein_expanded, rot_k, k)
            q = (
                q.reshape(bsz, pos_feature_num, self.num_heads, tgt_len, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz, self.num_heads, tgt_len, pos_feature_num * self.head_dim)
            )
            k = (
                k.reshape(bsz, pos_feature_num, self.num_heads, src_len, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )
            v = (
                v.reshape(bsz, pos_feature_num, self.num_heads, src_len, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )
        else:
            q = (
                q.reshape(bsz, tgt_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(bsz, self.num_heads, tgt_len, pos_feature_num * self.head_dim)
            )
            k = (
                k.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )
            v = (
                v.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        attn_weights = q.matmul(
            k.transpose(-1, -2)
        )  # (bsz, num_heads, tgt_len, src_len)

        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias

        if local_attention_weight is not None:
            local_attention_weight = local_attention_weight.to(dtype=q.dtype)
            if self.use_smooth_softmax:
                attn_weights = (
                    attn_weights + self.smooth_factor
                ) * local_attention_weight.unsqueeze(1) - self.smooth_factor
            elif self.use_no_pre_cutoff_softmax:
                pass
            else:
                attn_weights = attn_weights.masked_fill(
                    (local_attention_weight <= 1e-5).unsqueeze(1), float("-inf")
                )

        if key_padding_mask is not None:
            if pbc_expand_batched is not None:
                expand_mask = pbc_expand_batched["expand_mask"]
                key_padding_mask = torch.cat([key_padding_mask, expand_mask], dim=-1)
            # don't attend to padding symbols
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

        attn_probs_float = F.dropout(
            F.softmax(attn_weights, dim=-1, dtype=torch.float32),
            self.dropout,
            self.training,
        )

        if local_attention_weight is not None:
            attn_probs_float = attn_probs_float * local_attention_weight.unsqueeze(1)

        attn_probs = attn_probs_float.type_as(
            attn_weights
        )  # (bsz, num_heads, tgt_len, src_len)

        attn = attn_probs.matmul(v)  # (bsz, num_heads, tgt_len, 3 * head_dim)

        attn = (
            attn.transpose(1, 2)
            .reshape(bsz, tgt_len, self.num_heads, pos_feature_num, self.head_dim)
            .transpose(2, 3)
            .reshape(bsz, tgt_len, pos_feature_num, self.hidden_channels)
        )

        attn = self.attn_ln(attn)

        attn = self.out_proj(attn)

        return attn


class InvariantSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_channels,
        head_dim,
        num_heads,
        dropout,
        use_linear_bias=False,
        use_memory_efficient_attention=True,
        use_smooth_softmax=False,
        smooth_factor=0.0,
        use_no_pre_cutoff_softmax: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.use_linear_bias = use_linear_bias
        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)
        if use_memory_efficient_attention:
            self.invariant_attention = MemEffInvariantAttention(
                hidden_channels,
                head_dim,
                dropout,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
            )
        else:
            self.invariant_attention = InvariantAttention(
                hidden_channels,
                head_dim,
                dropout,
                use_linear_bias=use_linear_bias,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
            )

        self.reset_parameters(1)

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.use_linear_bias:
            self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.use_linear_bias:
            self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.use_linear_bias:
            self.v_proj.bias.data.fill_(0)

    def forward(
        self,
        x,
        attn_bias,
        mask,
        pbc_expand_batched: Optional[Dict] = None,
        is_protein: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn = self.invariant_attention(
            q,
            k,
            v,
            attn_bias,
            mask,
            pbc_expand_batched=pbc_expand_batched,
            is_protein=is_protein,
            position_ids=position_ids,
        )

        return attn


class EquivariantSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_channels,
        head_dim,
        num_heads,
        dropout,
        use_memory_efficient_attention=True,
        use_smooth_equviariant_norm=False,
        use_smooth_softmax=False,
        smooth_factor=0.0,
        add_rope=True,
        use_no_pre_cutoff_softmax: bool = False,
        add_vec_bias=False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        if use_memory_efficient_attention:
            self.equiariant_attention = MemEffEquivariantAttention(
                hidden_channels,
                head_dim,
                dropout,
                use_smooth_equviariant_norm=use_smooth_equviariant_norm,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
            )
        else:
            self.equiariant_attention = EquivariantAttention(
                hidden_channels,
                head_dim,
                dropout,
                use_smooth_equviariant_norm=use_smooth_equviariant_norm,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                add_rope=add_rope,
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
                add_vec_bias=add_vec_bias,
            )

        self.reset_parameters(1)

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0 / math.sqrt(d_tilde))

    def forward(
        self,
        vec,
        attn_bias,
        mask,
        pbc_expand_batched: Optional[Dict] = None,
        is_protein: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ):
        q = self.q_proj(vec)
        k = self.k_proj(vec)
        v = self.v_proj(vec)

        attn = self.equiariant_attention(
            q,
            k,
            v,
            attn_bias,
            mask,
            pbc_expand_batched=pbc_expand_batched,
            is_protein=is_protein,
            position_ids=position_ids,
        )

        return attn


class Invariant2EquivariantAttention(nn.Module):
    def __init__(
        self,
        hidden_channels,
        head_dim,
        num_heads,
        dropout,
        use_linear_bias=False,
        use_memory_efficient_attention=True,
        use_smooth_softmax=False,
        smooth_factor=0.0,
        use_no_pre_cutoff_softmax: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.use_linear_bias = use_linear_bias
        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)
        self.k1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.k2_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.v1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.v2_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        if use_memory_efficient_attention:
            self.invariant_attention = MemEffInvariantAttention(
                hidden_channels,
                head_dim,
                dropout,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
            )
        else:
            self.invariant_attention = InvariantAttention(
                hidden_channels,
                head_dim,
                dropout,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
            )

        self.reset_parameters(1)

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.use_linear_bias:
            self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k1_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(self.k2_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(self.v1_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(self.v2_proj.weight, gain=1.0 / math.sqrt(d_tilde))

    def forward(
        self,
        x,
        vec,
        attn_bias,
        mask,
        pbc_expand_batched: Optional[Dict] = None,
        is_protein: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ):
        pos_feature_num = vec.shape[2]
        q = self.q_proj(x)
        k1 = self.k1_proj(vec)
        k2 = self.k2_proj(vec)
        k = (k1 * k2).sum(dim=-2) * (
            pos_feature_num**-0.5
        )  # (n_graph, n_node, feat_dim)
        v1 = self.v1_proj(vec)
        v2 = self.v2_proj(vec)
        v = (v1 * v2).sum(dim=-2) * (
            pos_feature_num**-0.5
        )  # (n_graph, n_node, feat_dim)

        attn = self.invariant_attention(
            q,
            k,
            v,
            attn_bias,
            mask,
            pbc_expand_batched=pbc_expand_batched,
            is_protein=is_protein,
            position_ids=position_ids,
        )

        return attn


class Equivariant2InvariantAttention(nn.Module):
    def __init__(
        self,
        hidden_channels,
        head_dim,
        num_heads,
        dropout,
        eQi_choice,
        gbf_args,
        use_linear_bias=False,
        use_memory_efficient_attention=True,
        use_smooth_equviariant_norm=False,
        use_smooth_softmax=False,
        smooth_factor=0.0,
        add_rope=True,
        use_no_pre_cutoff_softmax: bool = False,
        add_vec_bias=False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.eQi_choice = eQi_choice
        self.K = gbf_args[0]
        self.edge_types = gbf_args[1]
        self.use_linear_bias = use_linear_bias
        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.k1_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)
        self.v1_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)
        if eQi_choice == "original":
            self.k2_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.v2_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        elif "gbf" in eQi_choice:
            self.gbf = GaussianLayer(self.K, self.edge_types)
            self.gbf_proj = nn.Linear(self.K, hidden_channels, bias=use_linear_bias)

        if use_memory_efficient_attention:
            self.equiariant_attention = MemEffEquivariantAttention(
                hidden_channels,
                head_dim,
                dropout,
                use_smooth_equviariant_norm=use_smooth_equviariant_norm,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                add_rope=add_rope,
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
                add_vec_bias=add_vec_bias,
            )
        else:
            self.equiariant_attention = EquivariantAttention(
                hidden_channels,
                head_dim,
                dropout,
                use_smooth_equviariant_norm=use_smooth_equviariant_norm,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                add_rope=add_rope,
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
                add_vec_bias=add_vec_bias,
            )

        self.reset_parameters(1)

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(self.k1_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.use_linear_bias:
            self.k1_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k2_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(self.v1_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.use_linear_bias:
            self.v1_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v2_proj.weight, gain=1.0 / math.sqrt(d_tilde))

    def forward(
        self,
        x,
        vec,
        attn_bias,
        mask,
        pos_unit,
        gbf_args,
        pbc_expand_batched: Optional[Dict] = None,
        is_protein: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ):
        q = self.q_proj(vec)

        k1 = self.k1_proj(x)
        v1 = self.v1_proj(x)

        pos_mean_centered_unit, pos_relative_unit = pos_unit
        if "gbf" in self.eQi_choice:
            dist, edge_type = gbf_args
            gbf_feature = self.gbf_proj(self.gbf(dist, edge_type))
            edge_feature = gbf_feature.masked_fill(
                mask.unsqueeze(1).unsqueeze(-1), 0.0
            )  # (n_graph, n_node, n_node, feat_dim)

        if self.eQi_choice == "original":
            k2 = self.k2_proj(vec)
            k = k1.unsqueeze(2) * k2  # (n_graph, n_node, pos_feature_num, feat_dim)
            v2 = self.v2_proj(vec)
            v = v1.unsqueeze(2) * v2  # (n_graph, n_node, pos_feature_num, feat_dim)

        elif self.eQi_choice == "meanCentered_vanilla":
            k = pos_mean_centered_unit.unsqueeze(-1) * k1.unsqueeze(
                -2
            )  # (n_graph, n_node, pos_feature_num, feat_dim)
            v = pos_mean_centered_unit.unsqueeze(-1) * v1.unsqueeze(
                -2
            )  # (n_graph, n_node, pos_feature_num, feat_dim)

        elif self.eQi_choice == "sumRelative_vanilla":
            k_edge = pos_relative_unit.unsqueeze(-1) * k1.unsqueeze(2).unsqueeze(
                -2
            )  # (n_graph, n_node, n_node, pos_feature_num, feat_dim)
            k_edge = k_edge.masked_fill(
                mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0
            )
            k = k_edge.sum(dim=2)  # (n_graph, n_node, pos_feature_num, feat_dim)
            k = k.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0.0)
            v_edge = pos_relative_unit.unsqueeze(-1) * v1.unsqueeze(2).unsqueeze(
                -2
            )  # (n_graph, n_node, n_node, pos_feature_num, feat_dim)
            v_edge = v_edge.masked_fill(
                mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0
            )
            v = v_edge.sum(dim=2)  # (n_graph, n_node, pos_feature_num, feat_dim)
            v = v.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0.0)

        elif self.eQi_choice == "meanCentered_gbf":
            gbf_sum = pos_mean_centered_unit.unsqueeze(-1) * edge_feature.sum(
                dim=2
            ).unsqueeze(
                -2
            )  # (n_graph, n_node, pos_feature_num, feat_dim)
            gbf_sum = gbf_sum.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0.0)
            k = (
                k1.unsqueeze(-2) * gbf_sum
            )  # (n_graph, n_node, pos_feature_num, feat_dim)
            v = (
                v1.unsqueeze(-2) * gbf_sum
            )  # (n_graph, n_node, pos_feature_num, feat_dim)

        elif self.eQi_choice == "sumRelative_gbf":
            feat_edge = pos_relative_unit.unsqueeze(-1) * edge_feature.unsqueeze(
                -2
            )  # (n_graph, n_node, n_node, pos_feature_num, feat_dim)
            feat_edge = feat_edge.masked_fill(
                mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0
            )
            feat_sum = feat_edge.sum(
                dim=2
            )  # (n_graph, n_node, pos_feature_num, feat_dim)
            feat_sum = feat_sum.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0.0)
            k = (
                k1.unsqueeze(-2) * feat_sum
            )  # (n_graph, n_node, pos_feature_num, feat_dim)
            v = (
                v1.unsqueeze(-2) * feat_sum
            )  # (n_graph, n_node, pos_feature_num, feat_dim)

        attn = self.equiariant_attention(
            q,
            k,
            v,
            attn_bias,
            mask,
            pbc_expand_batched=pbc_expand_batched,
            is_protein=is_protein,
            position_ids=position_ids,
        )

        return attn


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_heads,
        ffn_embedding_dim,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        eQi_choice: str = "original",
        gbf_args=None,
        layer_index=0,
        use_memory_efficient_attention=False,
        use_linear_bias=False,
        use_smooth_equviariant_norm=False,
        use_smooth_softmax=False,
        no_rotary_embedding_for_vector=False,
        smooth_factor=0.0,
        use_no_pre_cutoff_softmax: bool = False,
        add_vec_bias=False,
    ):
        super().__init__()

        head_dim = hidden_channels // num_heads

        self.layer_index = layer_index

        self.use_linear_bias = use_linear_bias

        if self.layer_index % 2 == 0:
            self.invariant_self_attention = InvariantSelfAttention(
                hidden_channels,
                head_dim,
                num_heads,
                dropout=attention_dropout,
                use_memory_efficient_attention=use_memory_efficient_attention,
                use_linear_bias=use_linear_bias,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
            )
            self.equivariant_self_attention = EquivariantSelfAttention(
                hidden_channels,
                head_dim,
                num_heads,
                dropout=attention_dropout,
                use_memory_efficient_attention=use_memory_efficient_attention,
                use_smooth_equviariant_norm=use_smooth_equviariant_norm,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                add_rope=(not no_rotary_embedding_for_vector),
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
                add_vec_bias=add_vec_bias,
            )
        else:
            self.invariant2equivariant_attention = Invariant2EquivariantAttention(
                hidden_channels,
                head_dim,
                num_heads,
                dropout=attention_dropout,
                use_memory_efficient_attention=use_memory_efficient_attention,
                use_linear_bias=use_linear_bias,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
            )
            self.equivaiant2invariant_attention = Equivariant2InvariantAttention(
                hidden_channels,
                head_dim,
                num_heads,
                dropout=attention_dropout,
                eQi_choice=eQi_choice,
                gbf_args=gbf_args,
                use_memory_efficient_attention=use_memory_efficient_attention,
                use_linear_bias=use_linear_bias,
                use_smooth_equviariant_norm=use_smooth_equviariant_norm,
                use_smooth_softmax=use_smooth_softmax,
                smooth_factor=smooth_factor,
                add_rope=(not no_rotary_embedding_for_vector),
                use_no_pre_cutoff_softmax=use_no_pre_cutoff_softmax,
                add_vec_bias=add_vec_bias,
            )

        self.invariant_attn_layer_norm = nn.LayerNorm(hidden_channels)
        self.equivariant_attn_layer_norm = EquivariantLayerNorm(
            hidden_channels,
            use_smooth_norm=use_smooth_equviariant_norm,
            add_vec_bias=add_vec_bias,
        )

        self.activation_fn = gelu
        # for invariant branches, we can chose whether to use bias in linear mappings
        # note that equivariant_fc1 is for invariant branch
        self.invariant_fc1 = nn.Linear(
            hidden_channels, ffn_embedding_dim, bias=use_linear_bias
        )
        self.invariant_fc2 = nn.Linear(
            ffn_embedding_dim, hidden_channels, bias=use_linear_bias
        )
        self.equivariant_fc1 = nn.Linear(
            hidden_channels, ffn_embedding_dim, bias=use_linear_bias
        )

        # for equivariant branches, we always disable bias in linear mappings to keep equivariance
        self.equivariant_fc2 = nn.Linear(hidden_channels, ffn_embedding_dim, bias=False)
        self.equivariant_fc3 = nn.Linear(ffn_embedding_dim, hidden_channels, bias=False)

        self.invariant_ffn_layer_norm = nn.LayerNorm(hidden_channels)
        self.equivariant_ffn_layer_norm = EquivariantLayerNorm(
            hidden_channels,
            use_smooth_norm=use_smooth_equviariant_norm,
            add_vec_bias=add_vec_bias,
        )

        self.invariant_ffn_layer_norm_2 = nn.LayerNorm(ffn_embedding_dim)
        self.equivariant_ffn_layer_norm_2 = EquivariantLayerNorm(
            ffn_embedding_dim,
            use_smooth_norm=use_smooth_equviariant_norm,
            add_vec_bias=add_vec_bias,
        )

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.attention_dropout = attention_dropout

        self.reset_parameters()

    def reset_parameters(self, d_tilde=1.0):
        self.invariant_attn_layer_norm.reset_parameters()
        self.equivariant_attn_layer_norm.reset_parameters()

        nn.init.xavier_uniform_(
            self.invariant_fc1.weight, gain=1.0 / math.sqrt(d_tilde)
        )
        if self.use_linear_bias:
            self.invariant_fc1.bias.data.fill_(0)
        nn.init.xavier_uniform_(
            self.invariant_fc2.weight, gain=1.0 / math.sqrt(d_tilde)
        )
        if self.use_linear_bias:
            self.invariant_fc2.bias.data.fill_(0)
        nn.init.xavier_uniform_(
            self.equivariant_fc1.weight, gain=1.0 / math.sqrt(d_tilde)
        )
        if self.use_linear_bias:
            self.equivariant_fc1.bias.data.fill_(0)
        nn.init.xavier_uniform_(
            self.equivariant_fc2.weight, gain=1.0 / math.sqrt(d_tilde)
        )
        nn.init.xavier_uniform_(
            self.equivariant_fc3.weight, gain=1.0 / math.sqrt(d_tilde)
        )

        self.invariant_ffn_layer_norm.reset_parameters()
        self.equivariant_ffn_layer_norm.reset_parameters()
        self.invariant_ffn_layer_norm_2.reset_parameters()
        self.equivariant_ffn_layer_norm_2.reset_parameters()

    def forward(
        self,
        x,
        vec,
        attn_bias_iself,
        attn_bias_i2e,
        attn_bias_eself,
        attn_bias_e2i,
        mask,
        pos_unit,
        gbf_args,
        pbc_expand_batched: Optional[Dict] = None,
        is_protein: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ):
        # attetion
        dx = self.invariant_attn_layer_norm(x)
        dvec = self.equivariant_attn_layer_norm(vec)

        if self.layer_index % 2 == 0:
            dx_invariant = self.invariant_self_attention(
                dx,
                attn_bias_iself,
                mask,
                pbc_expand_batched=pbc_expand_batched,
                is_protein=is_protein,
                position_ids=position_ids,
            )
            dx_invariant = F.dropout(
                dx_invariant, p=self.dropout, training=self.training
            )
            dvec_equivariant = self.equivariant_self_attention(
                dvec,
                attn_bias_eself,
                mask,
                pbc_expand_batched=pbc_expand_batched,
                is_protein=is_protein,
                position_ids=position_ids,
            )
            x = x + dx_invariant
            vec = vec + dvec_equivariant
        else:
            dx_equivariant = self.invariant2equivariant_attention(
                dx,
                dvec,
                attn_bias_i2e,
                mask,
                pbc_expand_batched=pbc_expand_batched,
                is_protein=is_protein,
                position_ids=position_ids,
            )
            dx_equivariant = F.dropout(
                dx_equivariant, p=self.dropout, training=self.training
            )
            dvec_invariant = self.equivaiant2invariant_attention(
                dx,
                dvec,
                attn_bias_e2i,
                mask,
                pos_unit,
                gbf_args,
                pbc_expand_batched=pbc_expand_batched,
                is_protein=is_protein,
                position_ids=position_ids,
            )

            x = x + dx_equivariant
            vec = vec + dvec_invariant

        # FFN
        dx = self.invariant_ffn_layer_norm(x)
        dvec = self.equivariant_ffn_layer_norm(vec)

        dx_ffn = self.activation_fn(self.invariant_fc1(dx))
        dx_ffn = F.dropout(dx_ffn, p=self.activation_dropout, training=self.training)
        dx_ffn = self.invariant_ffn_layer_norm_2(dx_ffn)
        dx_ffn = self.invariant_fc2(dx_ffn)
        dx_ffn = F.dropout(dx_ffn, p=self.dropout, training=self.training)
        dvec_ffn = self.activation_fn(self.equivariant_fc1(dx)).unsqueeze(
            -2
        ) * self.equivariant_fc2(dvec)
        dvec_ffn = self.equivariant_ffn_layer_norm_2(dvec_ffn)
        dvec_ffn = self.equivariant_fc3(dvec_ffn)

        # add & norm
        x = x + dx_ffn
        vec = vec + dvec_ffn

        return x, vec


class GeomFormer(nn.Module):
    def __init__(
        self,
        psm_config: PSMConfig,
        num_pred_attn_layer: int,
        embedding_dim: int,
        num_attention_heads: int,
        ffn_embedding_dim: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        num_3d_bias_kernel: int,
        num_edges: int,
        num_atoms: int,
    ):
        super().__init__()
        self.psm_config = psm_config

        self.unified_encoder_layers = nn.ModuleList()
        for layer_index in range(num_pred_attn_layer):
            layer = EncoderLayer(
                embedding_dim,
                num_attention_heads,
                ffn_embedding_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                eQi_choice="original",
                gbf_args=[num_3d_bias_kernel, num_edges],
                layer_index=layer_index,
                use_memory_efficient_attention=psm_config.use_memory_efficient_attention,
                use_linear_bias=psm_config.equivar_use_linear_bias,
                use_smooth_equviariant_norm=psm_config.use_smooth_equviariant_norm,
                use_smooth_softmax=psm_config.use_smooth_softmax,
                smooth_factor=psm_config.smooth_factor,
                no_rotary_embedding_for_vector=psm_config.no_rotary_embedding_for_vector,
                use_no_pre_cutoff_softmax=psm_config.use_no_pre_cutoff_softmax,
            )
            self.unified_encoder_layers.append(layer)

        self.unified_gbf_pos = NodeGaussianLayer(num_3d_bias_kernel, num_atoms)

        self.unified_gbf_vec = GaussianLayer(num_3d_bias_kernel, num_edges)

        if self.psm_config.equivar_use_attention_bias:
            self.unified_gbf_attn_bias = GaussianLayer(num_3d_bias_kernel, num_edges)

            self.unified_bias_proj = nn.Linear(
                num_3d_bias_kernel,
                num_attention_heads,
                bias=self.psm_config.equivar_use_linear_bias,
            )

        if self.psm_config.equivar_vec_init == VecInitApproach.ZERO_CENTERED_POS:
            self.unified_vec_proj = nn.Linear(
                num_3d_bias_kernel,
                embedding_dim,
                bias=self.psm_config.equivar_use_linear_bias,
            )
        elif self.psm_config.equivar_vec_init in [
            VecInitApproach.RELATIVE_POS,
            VecInitApproach.AUGMENTED_RELATIVE_POS,
        ]:
            self.unified_vec_proj = nn.Linear(
                num_3d_bias_kernel,
                embedding_dim,
                bias=False,
            )
        elif self.psm_config.equivar_vec_init == VecInitApproach.RELATIVE_POS_VEC_BIAS:
            self.unified_vec_proj = nn.Linear(
                num_3d_bias_kernel,
                embedding_dim,
                bias=True,
            )

        self.unified_augmented_vec_proj = nn.Linear(
            num_3d_bias_kernel,
            embedding_dim,
            bias=self.psm_config.equivar_use_linear_bias,
        )

        self.unified_final_equivariant_ln = EquivariantLayerNorm(
            embedding_dim,
            use_smooth_norm=self.psm_config.use_smooth_equviariant_norm,
            add_vec_bias=False,
            # (
            #     self.psm_config.equivar_vec_init
            #     == VecInitApproach.RELATIVE_POS_VEC_BIAS
            # ),
        )

        self.unified_final_invariant_ln = nn.LayerNorm(embedding_dim)

        self.unified_final_feature_ln = nn.LayerNorm(embedding_dim)

        self.time_embed_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embedding_dim, psm_config.num_3d_bias_kernel, bias=True),
        )

    def forward(
        self,
        batched_data,
        x,
        pos,
        mixed_attn_bias,
        padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
        time_embed: Optional[Tensor] = None,
    ):
        if x.dtype != torch.float32 and self.psm_config.use_fp32_in_decoder:
            x = x.to(dtype=torch.float32)
        pos = pos.to(dtype=x.dtype)
        if time_embed is not None:
            time_embed = time_embed.to(dtype=x.dtype)
        n_node = pos.shape[1]
        if pbc_expand_batched is not None:
            # use pbc and multi-graph
            node_type_edge = pbc_expand_batched["expand_node_type_edge"]
            node_type = batched_data["masked_token_type"]
            expand_pos = torch.cat([pos, pbc_expand_batched["expand_pos"]], dim=1)
            uni_delta_pos = (pos.unsqueeze(2) - expand_pos.unsqueeze(1)).to(
                dtype=x.dtype
            )
            n_expand_node = expand_pos.size()[-2]
            expand_mask = torch.cat(
                [padding_mask, pbc_expand_batched["expand_mask"]], dim=-1
            )
        else:
            node_type = batched_data["masked_token_type"]
            node_type_edge = batched_data["node_type_edge"]
            uni_delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
            n_expand_node = n_node
            expand_mask = padding_mask

        dist = (
            uni_delta_pos.norm(dim=-1).view(-1, n_node, n_expand_node).to(dtype=x.dtype)
        )
        uni_delta_pos = uni_delta_pos / (
            dist.unsqueeze(-1) + 1e-5
        )  # avoid inplace operation(autograd)

        if self.psm_config.equivar_vec_init == VecInitApproach.ZERO_CENTERED_POS:
            # r_i/||r_i|| * gbf(||r_i||)
            pos_norm = pos.norm(dim=-1)
            uni_gbf_pos_feature = self.unified_gbf_pos(
                pos_norm, node_type.unsqueeze(-1)
            )
            uni_pos_feature = uni_gbf_pos_feature.masked_fill(
                padding_mask.unsqueeze(-1), 0.0
            )
            uni_vec_value = self.unified_vec_proj(uni_pos_feature).unsqueeze(-2)

            if time_embed is not None:
                uni_vec_value += time_embed

            vec = pos.unsqueeze(-1) * uni_vec_value
        elif self.psm_config.equivar_vec_init in [
            VecInitApproach.RELATIVE_POS,
            VecInitApproach.AUGMENTED_RELATIVE_POS,
            VecInitApproach.RELATIVE_POS_VEC_BIAS,
        ]:
            uni_gbf_pos_feature = self.unified_gbf_vec(
                dist, node_type_edge
            )  # n_graph x n_node x n_expand_node x num_kernel

            if time_embed is not None:
                uni_gbf_pos_feature += self.time_embed_proj(time_embed).unsqueeze(-2)

            uni_pos_feature = uni_gbf_pos_feature.masked_fill(
                padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0
            )
            uni_pos_feature = uni_pos_feature.masked_fill(
                expand_mask.unsqueeze(1).unsqueeze(-1), 0.0
            )

            if (
                self.psm_config.equivar_vec_init
                == VecInitApproach.AUGMENTED_RELATIVE_POS
            ):
                constant_one = torch.ones(
                    uni_delta_pos.size()[0:3] + (1,), device=uni_delta_pos.device
                )
                uni_delta_pos = torch.cat(
                    [constant_one, uni_delta_pos], dim=-1
                )  # uni_delta_pos_dim=3+1

            uni_pos_feature = uni_delta_pos.unsqueeze(-1) * uni_pos_feature.unsqueeze(
                -2
            )  # n_graph x n_node x n_expand_node x uni_pos_feature_dim x num_kernel
            if pbc_expand_batched is not None:
                local_attention_weight = pbc_expand_batched["local_attention_weight"]
                if local_attention_weight is not None:
                    local_attention_weight = local_attention_weight.to(dtype=x.dtype)
                    vec = (
                        uni_pos_feature
                        * local_attention_weight.unsqueeze(-1).unsqueeze(-1)
                    ).sum(
                        dim=-3
                    )  # n_graph x n_node x uni_pos_feature_dim x num_kernel
                else:
                    vec = uni_pos_feature.sum(
                        dim=-3
                    )  # n_graph x n_node x uni_pos_feature_dim x num_kernel
            else:
                vec = uni_pos_feature.sum(
                    dim=-3
                )  # n_graph x n_node x uni_pos_feature_dim x num_kernel
            if (
                self.psm_config.equivar_vec_init
                == VecInitApproach.AUGMENTED_RELATIVE_POS
            ):
                aug_vec = vec[:, :, 0, :].unsqueeze(-2)
                aug_vec = self.unified_augmented_vec_proj(
                    aug_vec
                )  # n_graph x n_node x 1 x embedding_dim
                vec = vec[:, :, 1:, :]
                vec = self.unified_vec_proj(vec)  # n_graph x n_node x 3 x embedding_dim
                vec = torch.cat(
                    [aug_vec, vec], dim=-2
                )  # n_graph x n_node x 4 x embedding_dim
            else:
                vec = self.unified_vec_proj(vec)  # n_graph x n_node x 3 x embedding_dim
        else:
            raise ValueError(
                f"Unkown equivariant vector initialization method {self.psm_config.equivar_vec_init}"
            )

        vec = vec.masked_fill(padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        pos_mean_centered_dist = pos.norm(dim=-1)
        pos_mean_centered_unit = pos / (pos_mean_centered_dist.unsqueeze(-1) + 1e-5)

        if self.psm_config.equivar_use_attention_bias:
            uni_graph_attn_bias = mixed_attn_bias
        else:
            uni_graph_attn_bias = None

        output = x.contiguous().transpose(0, 1)
        output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        for _, layer in enumerate(self.unified_encoder_layers):
            output, vec = layer(
                output,
                vec,
                uni_graph_attn_bias,
                uni_graph_attn_bias,
                uni_graph_attn_bias,
                uni_graph_attn_bias,
                padding_mask,
                [pos_mean_centered_unit, uni_delta_pos],
                [dist, node_type_edge],
                pbc_expand_batched,
                is_protein=batched_data["is_protein"],
                position_ids=batched_data["position_ids"],
            )

        node_output = self.unified_final_equivariant_ln(vec)
        output = self.unified_final_invariant_ln(output)

        if self.psm_config.equivar_vec_init == VecInitApproach.AUGMENTED_RELATIVE_POS:
            node_output = node_output[:, :, 1:, :]

        return output, node_output
