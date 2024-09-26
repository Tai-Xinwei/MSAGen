# -*- coding: utf-8 -*-
import math
import warnings
from typing import Dict, Optional

import e3nn
import torch
from e3nn import o3
from e3nn.math import normalize2mom
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists
from e3nn.util.jit import compile_mode

# for bessel radial basis
from fairchem.core.models.gemnet.layers.radial_basis import RadialBasis
from torch import logical_not, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch_cluster import radius_graph
from torch_scatter import scatter

from sfm.models.psm.equivariant.equiformer.gaussian_rbf import (
    GaussianRadialBasisLayer,
    GaussianSmearing,
)
from sfm.models.psm.equivariant.equiformer.graph_attention_transformer import (
    RadialProfile,
    irreps2gate,
    sort_irreps_even_first,
)
from sfm.models.psm.equivariant.equiformer_v2.drop import DropPath

# from .e2former_att import *
from sfm.models.psm.equivariant.layer_norm import (  # ,\; EquivariantInstanceNorm,EquivariantGraphNorm; EquivariantRMSNormArraySphericalHarmonicsV2,
    EquivariantLayerNormV2,
    RMS_Norm_SH,
    get_norm_layer,
)
from sfm.models.psm.equivariant.maceblocks import (
    EquivariantProductBasisBlock,
    reshape_irreps,
)
from sfm.models.psm.equivariant.so2 import (
    SO2_Convolution,
    SO2_Convolution_sameorder,
    SO3_Rotation,
)
from sfm.models.psm.equivariant.tensor_product import Simple_TensorProduct
from sfm.models.psm.invariant.mixture_bias import GaussianLayer
from sfm.modules.rotary_embedding import RotaryEmbedding

# # QM9
# _MAX_ATOM_TYPE = 20
# # Statistics of QM9 with cutoff radius = 5
# _AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666


_RESCALE = True
_USE_BIAS = True


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        ## x could be any dimension.
        return 0.8 * x * torch.sigmoid(x) + 0.2 * x

    def extra_repr(self):
        return "negative_slope={}".format(self.alpha)


class Irreps2Scalar(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        out_dim,
        hidden_dim=None,
        bias=True,
        act="smoothleakyrelu",
        rescale=_RESCALE,
    ):
        """
        1. from irreps to scalar output: [...,irreps] - > [...,out_dim]
        2. bias is used for l=0
        3. act is used for l=0
        4. rescale is default, e.g. irreps is c0*l0+c1*l1+c2*l2+c3*l3, rescale weight is 1/c0**0.5 1/c1**0.5 ...
        """
        super().__init__()
        self.irreps_in = (
            o3.Irreps(irreps_in) if isinstance(irreps_in, str) else irreps_in
        )
        if hidden_dim is not None:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.irreps_in[0][0]  # l=0 scalar_dim
        self.out_dim = out_dim
        self.act = act
        self.bias = bias
        self.rescale = rescale

        self.vec_proj_list = nn.ModuleList()
        # self.irreps_in_len = sum([mul*(ir.l*2+1) for mul, ir in self.irreps_in])
        # self.scalar_in_len = sum([mul for mul, ir in self.irreps_in])
        self.lirreps = len(self.irreps_in)
        self.output_mlp = nn.Sequential(
            SmoothLeakyReLU(0.2) if self.act == "smoothleakyrelu" else nn.Identity(),
            nn.Linear(self.hidden_dim, out_dim),  # NOTICE init
        )

        for idx in range(len(self.irreps_in)):
            l = self.irreps_in[idx][1].l
            in_feature = self.irreps_in[idx][0]
            if l == 0:
                vec_proj = nn.Linear(in_feature, self.hidden_dim)
                # bound = 1 / math.sqrt(in_feature)
                # torch.nn.init.uniform_(vec_proj.weight, -bound, bound)
                nn.init.xavier_uniform_(vec_proj.weight)
                vec_proj.bias.data.fill_(0)
            else:
                vec_proj = nn.Linear(in_feature, 2 * (self.hidden_dim), bias=False)
                # bound = 1 / math.sqrt(in_feature*(2*l+1))
                # torch.nn.init.uniform_(vec_proj.weight, -bound, bound)
                nn.init.xavier_uniform_(vec_proj.weight)
            self.vec_proj_list.append(vec_proj)

    def forward(self, input_embedding):
        """
        from e3nn import o3
        irreps_in = o3.Irreps("100x1e+40x2e+10x3e")
        irreps_out = o3.Irreps("20x1e+20x2e+20x3e")
        irrepslinear = IrrepsLinear(irreps_in, irreps_out)
        irreps2scalar = Irreps2Scalar(irreps_in, 128)
        node_embed = irreps_in.randn(200,30,5,-1)
        out_scalar = irreps2scalar(node_embed)
        out_irreps = irrepslinear(node_embed)
        """

        # if input_embedding.shape[-1]!=self.irreps_in_len:
        #     raise ValueError("input_embedding should have same length as irreps_in_len")

        shape = list(input_embedding.shape[:-1])
        num = input_embedding.shape[:-1].numel()
        input_embedding = input_embedding.reshape(num, -1)

        start_idx = 0
        scalars = 0
        for idx, (mul, ir) in enumerate(self.irreps_in):
            if idx == 0 and ir.l == 0:
                scalars = self.vec_proj_list[0](
                    input_embedding[..., : self.irreps_in[0][0]]
                )
                start_idx += mul * (2 * ir.l + 1)
                continue
            vec_proj = self.vec_proj_list[idx]
            vec = (
                input_embedding[:, start_idx : start_idx + mul * (2 * ir.l + 1)]
                .reshape(-1, mul, (2 * ir.l + 1))
                .permute(0, 2, 1)
            )  # [B, 2l+1, D]
            vec1, vec2 = torch.split(
                vec_proj(vec), self.hidden_dim, dim=-1
            )  # [B, 2l+1, D]
            vec_dot = (vec1 * vec2).sum(dim=1)  # [B, 2l+1, D]

            scalars = scalars + vec_dot  # TODO: concat
            start_idx += mul * (2 * ir.l + 1)

        output_embedding = self.output_mlp(scalars)
        output_embedding = output_embedding.reshape(shape + [self.out_dim])
        return output_embedding

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.irreps_in}, out_features={self.out_dim}"


# class IrrepsLinear(torch.nn.Module):
#     def __init__(
#         self,
#         irreps_in,
#         irreps_out,
#         hidden_dim=None,
#         bias=True,
#         act="smoothleakyrelu",
#         rescale=_RESCALE,
#     ):
#         """
#         1. from irreps to scalar output: [...,irreps] - > [...,out_dim]
#         2. bias is used for l=0
#         3. act is used for l=0
#         4. rescale is default, e.g. irreps is c0*l0+c1*l1+c2*l2+c3*l3, rescale weight is 1/c0**0.5 1/c1**0.5 ...
#         """
#         super().__init__()
#         self.irreps_in = o3.Irreps(irreps_in) if isinstance(irreps_in,str) else irreps_in
#         self.irreps_out = o3.Irreps(irreps_out) if isinstance(irreps_out,str) else irreps_out

#         self.irreps_in_len = sum([mul*(ir.l*2+1) for mul, ir in self.irreps_in])
#         self.irreps_out_len = sum([mul*(ir.l*2+1) for mul, ir in self.irreps_out])
#         if hidden_dim is not None:
#             self.hidden_dim = hidden_dim
#         else:
#             self.hidden_dim = self.irreps_in[0][0]  # l=0 scalar_dim
#         self.act = act
#         self.bias = bias
#         self.rescale = rescale

#         self.vec_proj_list = nn.ModuleList()
#         # self.irreps_in_len = sum([mul*(ir.l*2+1) for mul, ir in self.irreps_in])
#         # self.scalar_in_len = sum([mul for mul, ir in self.irreps_in])
#         self.output_mlp = nn.Sequential(
#             SmoothLeakyReLU(0.2) if self.act == "smoothleakyrelu" else nn.Identity(),
#             nn.Linear(self.hidden_dim, self.irreps_out[0][0]),
#         )
#         self.weight_list = nn.ParameterList()
#         for idx in range(len(self.irreps_in)):
#             l = self.irreps_in[idx][1].l
#             in_feature = self.irreps_in[idx][0]
#             if l == 0:
#                 vec_proj = nn.Linear(in_feature, self.hidden_dim)
#                 nn.init.xavier_uniform_(vec_proj.weight)
#                 vec_proj.bias.data.fill_(0)
#             else:
#                 vec_proj = nn.Linear(in_feature, 2 * self.hidden_dim, bias=False)
#                 nn.init.xavier_uniform_(vec_proj.weight)

#                 # weight for l>0
#                 out_feature = self.irreps_out[idx][0]
#                 weight = torch.nn.Parameter(
#                                 torch.randn( out_feature,in_feature)
#                             )
#                 bound = 1 / math.sqrt(in_feature) if self.rescale else 1
#                 torch.nn.init.uniform_(weight, -bound, bound)
#                 self.weight_list.append(weight)

#             self.vec_proj_list.append(vec_proj)


#     def forward(self, input_embedding):
#         """
#         from e3nn import o3
#         irreps_in = o3.Irreps("100x1e+40x2e+10x3e")
#         irreps_out = o3.Irreps("20x1e+20x2e+20x3e")
#         irrepslinear = IrrepsLinear(irreps_in, irreps_out)
#         irreps2scalar = Irreps2Scalar(irreps_in, 128)
#         node_embed = irreps_in.randn(200,30,5,-1)
#         out_scalar = irreps2scalar(node_embed)
#         out_irreps = irrepslinear(node_embed)
#         """

#         # if input_embedding.shape[-1]!=self.irreps_in_len:
#         #     raise ValueError("input_embedding should have same length as irreps_in_len")

#         shape = list(input_embedding.shape[:-1])
#         num = input_embedding.shape[:-1].numel()
#         input_embedding = input_embedding.reshape(num, -1)

#         start_idx = 0
#         scalars = self.vec_proj_list[0](input_embedding[..., : self.irreps_in[0][0]])
#         output_embedding = []
#         for idx, (mul, ir) in enumerate(self.irreps_in):
#             if idx == 0:
#                 start_idx += mul * (2 * ir.l + 1)
#                 continue
#             vec_proj = self.vec_proj_list[idx]
#             vec = (
#                 input_embedding[:, start_idx : start_idx + mul * (2 * ir.l + 1)]
#                 .reshape(-1, mul, (2 * ir.l + 1))
#             )  # [B, D, 2l+1]
#             vec1, vec2 = torch.split(
#                 vec_proj(vec.permute(0, 2, 1)), self.hidden_dim, dim=-1
#             )  # [B, 2l+1, D]
#             vec_dot = (vec1 * vec2).sum(dim=1)  # [B, 2l+1, D]

#             scalars = scalars + vec_dot # TODO: concat

#             # linear for l>0
#             weight = self.weight_list[idx-1]
#             out = torch.matmul(weight,vec).reshape(num,-1) # [B*L, -1]
#             output_embedding.append(out)

#             start_idx += mul * (2 * ir.l + 1)
#         try:
#             scalars = self.output_mlp(scalars)
#         except:
#             raise ValueError(f"scalars shape: {scalars.shape}")
#         output_embedding.insert(0, scalars)
#         output_embedding = torch.cat(output_embedding, dim=1)
#         output_embedding = output_embedding.reshape(shape + [self.irreps_out_len])
#         return output_embedding

#     def __repr__(self):
#         return f"{self.__class__.__name__}(in_features={self.irreps_in}, out_features={self.irreps_out}"


class IrrepsLinear(torch.nn.Module):
    def __init__(
        self, irreps_in, irreps_out, bias=True, act="smoothleakyrelu", rescale=_RESCALE
    ):
        """
        1. from irreps_in to irreps_out output: [...,irreps_in] - > [...,irreps_out]
        2. bias is used for l=0
        3. act is used for l=0
        4. rescale is default, e.g. irreps is c0*l0+c1*l1+c2*l2+c3*l3, rescale weight is 1/c0**0.5 1/c1**0.5 ...
        """
        super().__init__()
        self.irreps_in = (
            o3.Irreps(irreps_in) if isinstance(irreps_in, str) else irreps_in
        )
        self.irreps_out = (
            o3.Irreps(irreps_out) if isinstance(irreps_out, str) else irreps_out
        )

        self.act = act
        self.bias = bias
        self.rescale = rescale

        for idx2 in range(len(self.irreps_out)):
            if self.irreps_out[idx2][1] not in self.irreps_in:
                raise ValueError(
                    f"Error: each irrep of irreps_out {self.irreps_out} should be in irreps_in {self.irreps_in}. Please check your input and output "
                )

        self.weight_list = nn.ParameterList()
        self.bias_list = nn.ParameterList()
        self.act_list = nn.ModuleList()
        self.irreps_in_len = sum([mul * (ir.l * 2 + 1) for mul, ir in self.irreps_in])
        self.irreps_out_len = sum([mul * (ir.l * 2 + 1) for mul, ir in self.irreps_out])
        self.instructions = []
        start_idx = 0
        for idx1 in range(len(self.irreps_in)):
            l = self.irreps_in[idx1][1].l
            mul = self.irreps_in[idx1][0]
            for idx2 in range(len(self.irreps_out)):
                if self.irreps_in[idx1][1].l == self.irreps_out[idx2][1].l:
                    self.instructions.append(
                        [idx1, mul, l, start_idx, start_idx + (l * 2 + 1) * mul]
                    )
                    out_feature = self.irreps_out[idx2][0]

                    weight = torch.nn.Parameter(torch.randn(out_feature, mul))
                    bound = 1 / math.sqrt(mul) if self.rescale else 1
                    torch.nn.init.uniform_(weight, -bound, bound)
                    self.weight_list.append(weight)

                    bias = torch.nn.Parameter(
                        torch.randn(1, out_feature, 1)
                        if self.bias and l == 0
                        else torch.zeros(1, out_feature, 1)
                    )
                    self.bias_list.append(bias)

                    activation = (
                        nn.Sequential(SmoothLeakyReLU())
                        if self.act == "smoothleakyrelu" and l == 0
                        else nn.Sequential()
                    )
                    self.act_list.append(activation)

            start_idx += (l * 2 + 1) * mul

    def forward(self, input_embedding):
        """
        from e3nn import o3
        irreps_in = o3.Irreps("100x1e+40x2e+10x3e")
        irreps_out = o3.Irreps("20x1e+20x2e+20x3e")
        irrepslinear = IrrepsLinear(irreps_in, irreps_out)
        irreps2scalar = Irreps2Scalar(irreps_in, 128)
        node_embed = irreps_in.randn(200,30,5,-1)
        out_scalar = irreps2scalar(node_embed)
        out_irreps = irrepslinear(node_embed)
        """

        if input_embedding.shape[-1] != self.irreps_in_len:
            raise ValueError("input_embedding should have same length as irreps_in_len")

        shape = list(input_embedding.shape[:-1])
        num = input_embedding.shape[:-1].numel()
        input_embedding = input_embedding.reshape(num, -1)

        output_embedding = []
        for idx, (_, mul, l, start, end) in enumerate(self.instructions):
            weight = self.weight_list[idx]
            bias = self.bias_list[idx]
            activation = self.act_list[idx]

            out = (
                torch.matmul(
                    weight, input_embedding[:, start:end].reshape(-1, mul, (2 * l + 1))
                )
                + bias
            )
            out = activation(out).reshape(num, -1)
            output_embedding.append(out)

        output_embedding = torch.cat(output_embedding, dim=1)
        output_embedding = output_embedding.reshape(shape + [self.irreps_out_len])
        return output_embedding


class EquivariantAttention(torch.nn.Module):
    def __init__(self, irreps_in, attn_type="v0", num_attn_heads=8):
        """ """
        super().__init__()
        self.irreps_in = (
            o3.Irreps(irreps_in) if isinstance(irreps_in, str) else irreps_in
        )
        self.irreps_in_len = sum([mul * (ir.l * 2 + 1) for mul, ir in self.irreps_in])
        self.attn_type = attn_type
        self.instructions = []
        self.mul_sizes = []

        start_idx = 0
        self.alpha_dot_list = nn.ParameterList()

        for idx1 in range(len(self.irreps_in)):
            l = self.irreps_in[idx1][1].l
            mul = self.irreps_in[idx1][0]
            self.instructions.append(
                [idx1, mul, l, start_idx, start_idx + (l * 2 + 1) * mul]
            )

            start_idx += (l * 2 + 1) * mul
            self.mul_sizes.append(mul)
            if attn_type == "v1":
                alpha_dot = torch.nn.Parameter(torch.randn(num_attn_heads, mul))
                std = 1.0 / math.sqrt(mul)
                torch.nn.init.uniform_(alpha_dot, -std, std)
                self.alpha_dot_list.append(alpha_dot)

    def forward(self, Q, K):
        """
        Q, K: [B,L,heads,irreps_head]
        """
        B, L, H, _ = Q.shape
        # assert  input_embedding.shape[-1]==self.irreps_in_len, "input_embedding should have same length as irreps_in_len"
        # assert scalars.shape[-1]==sum(self.mul_sizes), f"scalars should have dim {sum(self.mul_sizes)} but got {scalars.shape[-1]}"

        # shape = list(input_embedding.shape[:-1])
        # num = input_embedding.shape[:-1].numel()
        # input_embedding = input_embedding.reshape(num,-1)
        alpha = 0.0
        if self.attn_type == "v0":
            for idx, (_, mul, l, start, end) in enumerate(self.instructions):
                if idx == 0:
                    vec_q = Q[..., start:end]  # (B, L, H, D)
                    vec_k = K[..., start:end]  # (B, L, H, D)
                    alpha_l = torch.einsum(
                        "bmhd,bnhd->bmnh", vec_q, vec_k
                    )  # (B, L, L, H)
                else:
                    vec_q = (
                        Q[..., start:end]
                        .reshape(B, L, H, mul, (2 * l + 1))
                        .permute(0, 1, 2, 4, 3)
                    )  # (B, L, H, 2*l+1, D)
                    vec_k = (
                        K[..., start:end]
                        .reshape(B, L, H, mul, (2 * l + 1))
                        .permute(0, 1, 2, 4, 3)
                    )  # (B, L, H, 2*l+1, D)
                    alpha_l = torch.einsum(
                        "bmhld,bnhld->bmnh", vec_q, vec_k
                    )  # (B, L, L, H, D)
                alpha += alpha_l
        elif self.attn_type == "v1":
            for idx, (_, mul, l, start, end) in enumerate(self.instructions):
                alpha_dot = self.alpha_dot_list[idx]
                if idx == 0:
                    vec_q = Q[..., start:end]  # (B, L, H, D)
                    vec_k = K[..., start:end]  # (B, L, H, D)
                    alpha_l = torch.einsum(
                        "bmhd,bnhd->bmnhd", vec_q, vec_k
                    )  # (B, L, L, H, D)
                    alpha_l = torch.einsum(
                        "bmnhd,hd ->bmnh", alpha_l, alpha_dot
                    )  # (B, L, L, H)
                else:
                    vec_q = (
                        Q[..., start:end]
                        .reshape(B, L, H, mul, (2 * l + 1))
                        .permute(0, 1, 2, 4, 3)
                    )  # (B, L, H, 2*l+1, D)
                    vec_k = (
                        K[..., start:end]
                        .reshape(B, L, H, mul, (2 * l + 1))
                        .permute(0, 1, 2, 4, 3)
                    )  # (B, L, H, 2*l+1, D)
                    alpha_l = torch.einsum(
                        "bmhld,bnhld->bmnhd", vec_q, vec_k
                    )  # (B, L, L, H, D)
                    alpha_l = torch.einsum(
                        "bmnhd,hd ->bmnh", alpha_l, alpha_dot
                    )  # (B, L, L, H) TODO: opt einsum
                alpha += alpha_l
        return alpha


@compile_mode("script")
class Vec2AttnHeads(torch.nn.Module):
    """
    Reshape vectors of shape [..., irreps_head] to vectors of shape
    [..., num_heads, irreps_head].
    """

    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        shape = list(x.shape[:-1])
        num = x.shape[:-1].numel()
        x = x.reshape(num, -1)

        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        out = out.reshape(shape + [self.num_heads, -1])
        return out

    def __repr__(self):
        return "{}(irreps_head={}, num_heads={})".format(
            self.__class__.__name__, self.irreps_head, self.num_heads
        )


@compile_mode("script")
class AttnHeads2Vec(torch.nn.Module):
    """
    Convert vectors of shape [..., num_heads, irreps_head] into
    vectors of shape [..., irreps_head * num_heads].
    """

    def __init__(self, irreps_head, num_heads=-1):
        super().__init__()
        self.irreps_head = irreps_head
        self.num_heads = num_heads
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        head_cnt = x.shape[-2]
        shape = list(x.shape[:-2])
        num = x.shape[:-2].numel()
        x = x.reshape(num, head_cnt, -1)
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        out = out.reshape(shape + [-1])
        return out

    def __repr__(self):
        return "{}(irreps_head={})".format(self.__class__.__name__, self.irreps_head)


class EquivariantDropout(nn.Module):
    def __init__(self, irreps, drop_prob):
        """
        equivariant for irreps: [..., irreps]
        """

        super(EquivariantDropout, self).__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)
        self.mul = o3.ElementwiseTensorProduct(
            irreps, o3.Irreps("{}x0e".format(self.num_irreps))
        )

    def forward(self, x):
        """
        x: [..., irreps]

        t1 = o3.Irreps("5x0e+4x1e+3x2e")
        func = EquivariantDropout(t1, 0.5)
        out = func(t1.randn(2,3,-1))
        """
        if not self.training or self.drop_prob == 0.0:
            return x

        shape = x.shape
        N = x.shape[:-1].numel()
        x = x.reshape(N, -1)
        mask = torch.ones((N, self.num_irreps), dtype=x.dtype, device=x.device)
        mask = self.drop(mask)

        out = self.mul(x, mask)

        return out.reshape(shape)


class TensorProductRescale(torch.nn.Module):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
        mode="default",
    ):
        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.rescale = rescale
        self.use_bias = bias

        # e3nn.__version__ == 0.4.4
        # Use `path_normalization` == 'none' to remove normalization factor
        if mode == "simple":
            self.tp = Simple_TensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                instructions=instructions,
                # normalization=normalization,
                # internal_weights=internal_weights,
                # shared_weights=shared_weights,
                # path_normalization="none",
            )
        else:
            self.tp = o3.TensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                instructions=instructions,
                normalization=normalization,
                internal_weights=internal_weights,
                shared_weights=shared_weights,
                path_normalization="none",
            )

        self.init_rescale_bias()

    def calculate_fan_in(self, ins):
        return {
            "uvw": (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
            "uvu": self.irreps_in2[ins.i_in2].mul,
            "uvv": self.irreps_in1[ins.i_in1].mul,
            "uuw": self.irreps_in1[ins.i_in1].mul,
            "uuu": 1,
            "uvuv": 1,
            "uvu<v": 1,
            "u<vw": self.irreps_in1[ins.i_in1].mul
            * (self.irreps_in2[ins.i_in2].mul - 1)
            // 2,
        }[ins.connection_mode]

    def init_rescale_bias(self) -> None:
        irreps_out = self.irreps_out
        # For each zeroth order output irrep we need a bias
        # Determine the order for each output tensor and their dims
        self.irreps_out_orders = [
            int(irrep_str[-2]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_dims = [
            int(irrep_str.split("x")[0]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_slices = irreps_out.slices()

        # Store tuples of slices and corresponding biases in a list
        self.bias = None
        self.bias_slices = []
        self.bias_slice_idx = []
        self.irreps_bias = self.irreps_out.simplify()
        self.irreps_bias_orders = [
            int(irrep_str[-2]) for irrep_str in str(self.irreps_bias).split("+")
        ]
        self.irreps_bias_parity = [
            irrep_str[-1] for irrep_str in str(self.irreps_bias).split("+")
        ]
        self.irreps_bias_dims = [
            int(irrep_str.split("x")[0])
            for irrep_str in str(self.irreps_bias).split("+")
        ]
        if self.use_bias:
            self.bias = []
            for slice_idx in range(len(self.irreps_bias_orders)):
                if (
                    self.irreps_bias_orders[slice_idx] == 0
                    and self.irreps_bias_parity[slice_idx] == "e"
                ):
                    out_slice = self.irreps_bias.slices()[slice_idx]
                    out_bias = torch.nn.Parameter(
                        torch.zeros(
                            self.irreps_bias_dims[slice_idx], dtype=self.tp.weight.dtype
                        )
                    )
                    self.bias += [out_bias]
                    self.bias_slices += [out_slice]
                    self.bias_slice_idx += [slice_idx]
        self.bias = torch.nn.ParameterList(self.bias)

        self.slices_sqrt_k = {}
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = (
                    slices_fan_in[slice_idx] + fan_in
                    if slice_idx in slices_fan_in.keys()
                    else fan_in
                )
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                if self.rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.0
                self.slices_sqrt_k[slice_idx] = (
                    self.irreps_out_slices[slice_idx],
                    sqrt_k,
                )

            # Re-initialize weights in each instruction
            if self.tp.internal_weights:
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                    # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                    # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)
                    # else:
                    #    sqrt_k = 1.
                    #
                    # if self.rescale:
                    # weight.data.uniform_(-sqrt_k, sqrt_k)
                    #    weight.data.mul_(sqrt_k)
                    # self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            # for (out_slice_idx, out_slice, out_bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
            #    sqrt_k = 1 / slices_fan_in[out_slice_idx] ** 0.5
            #    out_bias.uniform_(-sqrt_k, sqrt_k)

    def forward_tp_rescale_bias(self, x, y, weight=None):
        out = self.tp(x, y, weight)
        # if self.rescale and self.tp.internal_weights:
        #    for (slice, slice_sqrt_k) in self.slices_sqrt_k.values():
        #        out[:, slice] /= slice_sqrt_k
        if self.use_bias:
            for _, slice, bias in zip(self.bias_slice_idx, self.bias_slices, self.bias):
                # out[:, slice] += bias
                out.narrow(-1, slice.start, slice.stop - slice.start).add_(bias)
        return out

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        return out


class SeparableFCTP(torch.nn.Module):
    def __init__(
        self,
        irreps_x,
        irreps_y,
        irreps_out,
        fc_neurons,
        use_activation=False,
        norm_layer="graph",
        internal_weights=False,
        mode="default",
    ):
        """
        Use separable FCTP for spatial convolution.
        [...,irreps_x] tp [...,irreps_y] - > [..., irreps_out]

        fc_neurons is not needed in e2former
        """

        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_x)
        self.irreps_edge_attr = o3.Irreps(irreps_y)
        self.irreps_node_output = o3.Irreps(irreps_out)
        norm = get_norm_layer(norm_layer)

        self.dtp = DepthwiseTensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            self.irreps_node_output,
            bias=False,
            internal_weights=internal_weights,
            mode=mode,
        )

        self.dtp_rad = None
        self.fc_neurons = fc_neurons
        if fc_neurons is not None:
            warnings.warn("NOTICEL: fc_neurons is not needed in e2former")
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for slice, slice_sqrt_k in self.dtp.slices_sqrt_k.values():
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k

        irreps_lin_output = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(
            self.irreps_node_output
        )
        if use_activation:
            irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output = irreps_lin_output.simplify()
        self.lin = IrrepsLinear(
            self.dtp.irreps_out.simplify(), irreps_lin_output, bias=False, act=None
        )

        self.norm = None
        if norm_layer is not None:
            self.norm = norm(self.irreps_node_output)

        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0:
                gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
            else:
                gate = Gate(
                    irreps_scalars,
                    [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                    irreps_gates,
                    [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                    irreps_gated,  # gated tensors
                )
            self.gate = gate

    def forward(self, irreps_x, irreps_y, xy_scalar_fea, batch=None, **kwargs):
        """
        x: [..., irreps]

        irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
        sep_tp = SeparableFCTP(irreps_in,"1x1e",irreps_in,fc_neurons=None,
                            use_activation=False,norm_layer=None,
                            internal_weights=True)
        out = sep_tp(irreps_in.randn(100,10,-1),torch.randn(100,10,3),None)
        print(out.shape)
        """
        shape = irreps_x.shape[:-1]
        N = irreps_x.shape[:-1].numel()
        irreps_x = irreps_x.reshape(N, -1)
        irreps_y = irreps_y.reshape(N, -1)

        weight = None
        if self.dtp_rad is not None and xy_scalar_fea is not None:
            xy_scalar_fea = xy_scalar_fea.reshape(N, -1)
            weight = self.dtp_rad(xy_scalar_fea)
        out = self.dtp(irreps_x, irreps_y, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out.reshape(list(shape) + [-1])


def Body2TensorProduct(
    irreps_node_input,
    irreps_edge_attr,
    irreps_node_output,
    internal_weights=False,
    bias=True,
    mode="default",
):
    """
    The irreps of output is pre-determined.
    `irreps_node_output` is used to get certain types of vectors.
    """
    irreps_output = []
    instructions = []

    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            if i > j or i == 0:
                continue

            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))

    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output)  # irreps_output.sort()
    instructions = [
        (i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions
    ]
    if mode != "default":
        if bias is True or internal_weights is False:
            raise ValueError("tp not support some parameter, please check your code.")
    tp = TensorProductRescale(
        irreps_node_input,
        irreps_edge_attr,
        irreps_output,
        instructions,
        internal_weights=internal_weights,
        shared_weights=internal_weights,
        bias=bias,
        rescale=_RESCALE,
        mode=mode,
    )
    return tp


class Body2_interaction(torch.nn.Module):
    def __init__(
        self,
        irreps_x,
        fc_neurons=None,
        use_activation=False,
        norm_layer="graph",
        internal_weights=False,
    ):
        """
        Use separable FCTP for spatial convolution.
        [...,irreps_x] tp [...,irreps_y] - > [..., irreps_out]

        fc_neurons is not needed in e2former
        """

        super().__init__()
        self.irreps_node_input = (
            o3.Irreps(irreps_x) if isinstance(irreps_x, str) else irreps_x
        )
        self.irreps_small = o3.Irreps("128x0e+128x1e+64x2e")
        self.irreps_small_fc_left = IrrepsLinear(
            self.irreps_node_input, self.irreps_small, bias=True, act=None
        )
        self.irreps_small_fc_right = IrrepsLinear(
            self.irreps_node_input, self.irreps_small, bias=True, act=None
        )
        self.dtp = Body2TensorProduct(
            self.irreps_small,
            self.irreps_small,
            self.irreps_small,
            bias=False,
            internal_weights=internal_weights,  # instruction is uvu
            mode="default",
        )

        self.lin = IrrepsLinear(
            self.dtp.irreps_out.simplify(), self.irreps_node_input, bias=True, act=None
        )

    def forward(self, irreps_x, **kwargs):
        """
        x: [..., irreps]

        irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
        sep_tp = SeparableFCTP(irreps_in,"1x1e",irreps_in,fc_neurons=None,
                            use_activation=False,norm_layer=None,
                            internal_weights=True)
        out = sep_tp(irreps_in.randn(100,10,-1),torch.randn(100,10,3),None)
        print(out.shape)
        """
        shape = irreps_x.shape[:-1]
        N = irreps_x.shape[:-1].numel()
        irreps_x = irreps_x.reshape(N, -1)

        out = self.dtp(
            self.irreps_small_fc_left(irreps_x),
            self.irreps_small_fc_right(irreps_x),
            None,
        )
        # print(out.shape,self.dtp.irreps_out)
        out = self.lin(out)

        return out.reshape(list(shape) + [-1])


class Body3_interaction_MACE(torch.nn.Module):
    def __init__(
        self,
        irreps_x,
        fc_neurons=None,
        use_activation=False,
        norm_layer="graph",
        internal_weights=False,
    ):
        """
        Use separable FCTP for spatial convolution.
        [...,irreps_x] tp [...,irreps_y] - > [..., irreps_out]

        fc_neurons is not needed in e2former
        """

        super().__init__()
        self.irreps_node_input = (
            o3.Irreps(irreps_x) if isinstance(irreps_x, str) else irreps_x
        )
        self.irreps_small = o3.Irreps("256x0e+256x1e+256x2e")
        self.reshape_func = reshape_irreps(self.irreps_small)
        self.irreps_small_fc = IrrepsLinear(
            self.irreps_node_input, self.irreps_small, bias=True, act=None
        )

        self.num_elements = 80
        self.dtp = EquivariantProductBasisBlock(
            node_feats_irreps=self.irreps_small,
            target_irreps=self.irreps_small,
            correlation=3,
            num_elements=self.num_elements,
            use_sc=False,
        )

        self.lin = IrrepsLinear(
            self.irreps_small, self.irreps_node_input, bias=True, act=None
        )

    def forward(self, irreps_x, atomic_numbers, **kwargs):
        """
        x: [..., irreps]

        irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
        sep_tp = SeparableFCTP(irreps_in,"1x1e",irreps_in,fc_neurons=None,
                            use_activation=False,norm_layer=None,
                            internal_weights=True)
        out = sep_tp(irreps_in.randn(100,10,-1),torch.randn(100,10,3),None)
        print(out.shape)
        """
        shape = irreps_x.shape[:-1]
        N = irreps_x.shape[:-1].numel()
        irreps_x = irreps_x.reshape(N, -1)
        irreps_x_small = self.reshape_func(self.irreps_small_fc(irreps_x))
        # print(irreps_x_small.shape,torch.max(atomic_numbers),torch.min(atomic_numbers))
        irreps_x_small = self.dtp(
            irreps_x_small,
            sc=None,
            node_attrs=torch.nn.functional.one_hot(
                atomic_numbers.reshape(-1).long(), num_classes=self.num_elements
            ).float(),
        )

        # print(out.shape,self.dtp.irreps_out)
        irreps_x_small = self.lin(irreps_x_small)

        return irreps_x_small.reshape(list(shape) + [-1])


@compile_mode("script")
class E2Attention(torch.nn.Module):
    """
    1. Message = Alpha * Value
    2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
    3. 0e -> Activation -> Inner Product -> (Alpha)
    4. (0e+1e+...) -> (Value)


    tp_type:
    v0: use irreps_node_input 0e
    v1: use sum of edge features
    v2: use sum of edge features and irreps_node_input 0e

    attn_type
    v0: initial implementation with attn_weight multiplication
    v1: GATv2 implementation without attn bias


    """

    def __init__(
        self,
        irreps_node_input="256x0e+64x1e+32x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+8x1e+4x2e",
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="v2",
        add_rope=True,
        **kwargs,
    ):
        super().__init__()

        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e
        self.num_attention_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        self.irreps_head = (
            o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )
        # irreps_node_output,  attention will not change the input shape/embeding length
        self.irreps_node_output = self.irreps_node_input
        # new params
        self.tp_type = tp_type
        self.attn_type = attn_type

        self.gbf_attn = GaussianLayer(
            self.attn_weight_input_dim
        )  # default output_dim = 128
        self.node_embedding = nn.Embedding(256, 32)
        nn.init.uniform_(self.node_embedding.weight.data, -0.001, 0.001)

        if self.attn_type == "w2head3":
            self.attn_weight2heads = nn.Sequential(
                nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
                nn.LayerNorm(attn_weight_input_dim),
                torch.nn.SiLU(),
                nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
                nn.LayerNorm(attn_weight_input_dim),
                torch.nn.SiLU(),
                nn.Linear(attn_weight_input_dim, 2 * num_attn_heads),
            )
        else:
            self.attn_weight2heads = nn.Sequential(
                nn.Linear(attn_weight_input_dim, 2 * num_attn_heads),
            )
        # self.attn_weight2heads = nn.Sequential(
        #     nn.Linear(attn_weight_input_dim, 2*num_attn_heads),
        #     # torch.nn.SiLU(),
        #     # nn.Linear(attn_weight_input_dim, 2 * num_attn_heads),
        # )

        self.multiheadatten_irreps = (
            (self.irreps_head * num_attn_heads).sort().irreps.simplify()
        )

        self.q_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )
        self.k_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )

        self.v_irlinear = IrrepsLinear(
            self.irreps_node_input,
            self.multiheadatten_irreps,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irreps

        self.vec2heads = Vec2AttnHeads(
            self.irreps_head,
            num_attn_heads,
        )
        self.heads2vec = AttnHeads2Vec(irreps_head=self.irreps_head)

        # alpha_dot, used for scalar to num_heads dimension
        # self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        self.alpha_act = SmoothLeakyReLU(0.2)  # used for attention_ij
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        self.sep_tp = None
        if self.nonlinear_message or self.rescale_degree:
            raise ValueError(
                "sorry, rescale_degree or non linear message passing is not supported in tp transformer"
            )

        else:
            if self.tp_type is None or self.tp_type == "None":
                self.sep_tp = SeparableFCTP(
                    self.multiheadatten_irreps,
                    "1x1e",
                    self.multiheadatten_irreps,
                    fc_neurons=None,
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=True,
                    mode="default",
                )
            else:
                self.sep_tp = SeparableFCTP(
                    self.multiheadatten_irreps,
                    "1x1e",
                    self.multiheadatten_irreps,
                    fc_neurons=[self.scalar_dim, self.scalar_dim, self.scalar_dim],
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=False,
                    mode="default",
                )

                if self.tp_type == "v2":
                    self.gbf = GaussianLayer(
                        self.attn_weight_input_dim
                    )  # default output_dim = 128
                    self.pos_embedding_proj = nn.Linear(
                        self.attn_weight_input_dim, self.scalar_dim
                    )
                    self.node_scalar_proj = nn.Linear(self.scalar_dim, self.scalar_dim)
                else:
                    raise ValueError(
                        f"attn_type should be None or v0 or v1 or v2 but got {self.attn_type}"
                    )

        self.proj = IrrepsLinear(
            (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout()
        self.add_rope = add_rope
        if add_rope:
            self.rot_emb = RotaryEmbedding(dim=self.attn_scalar_head)

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """
        irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
        B,L = 4,20
        node_pos = torch.randn(B,L,3)
        node_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))

        attn_scalar_head = 32
        func = GraphAttention(
            irreps_node_input = irreps_in,
            attn_weight_input_dim = 32, # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads = 8,
            attn_scalar_head = attn_scalar_head,
            irreps_head = "32x0e+8x1e+4x2e",
            rescale_degree=False,
            nonlinear_message=False,
            alpha_drop=0.1,
            proj_drop=0.1,
        )
        out = func(node_pos,
            irreps_in.randn(B,L,-1),
            node_dis,
            torch.randn(B,L,L,attn_scalar_head))
        print(out.shape)
        """

        # attn_weight f(|r_ij|): B*L*L*heads
        # edge_dis: B*L*L
        # node_input: B*L*irreps (irreps e.g. 128x0e+64x1e+32x2e)
        # ir2scalar: B*L*irreps -> N*L*hidden (hidden e.g. head*32)
        # attn_weight: B*L*L*rbf_dim
        B, L, _ = node_irreps_input.shape
        query = self.q_ir2scalar(node_irreps_input)
        key = self.k_ir2scalar(node_irreps_input)
        value = self.v_irlinear(node_irreps_input)  # B*L*irreps
        # embed = self.node_embedding(atomic_numbers)
        # if self.attn_type.startswith('v3'):
        # attn_weight = self.gbf_attn(edge_dis, batched_data["node_type_edge"].long()) # B*L*L*-1
        # torch.cat(
        #     [
        #         attn_weight,
        #         embed.reshape(B, L, 1, -1)
        #         .repeat(1, 1, L, 1),
        #         embed.reshape(B, 1, L, -1)
        #         .repeat(1, L, 1, 1),
        #     ],
        #     dim=-1,
        # ))

        if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
            outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
            outcell_index_0 = batched_data["outcell_index_0"]
            key = torch.cat([key, key[outcell_index_0, outcell_index]], dim=1)
            value = torch.cat([value, value[outcell_index_0, outcell_index]], dim=1)
            node_pos = torch.cat([node_pos, batched_data["expand_pos"].float()], dim=1)

        elif self.add_rope and batched_data["mol_type"] == 2:
            query = query.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
            key = key.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)
            query, key = self.rot_emb(
                query.transpose(1, 2).reshape(
                    B * self.num_attention_heads, L, self.attn_scalar_head
                ),
                key.transpose(1, 2).reshape(
                    B * self.num_attention_heads, L, self.attn_scalar_head
                ),
            )
            query = query.reshape(
                B, self.num_attention_heads, L, self.attn_scalar_head
            ).transpose(1, 2)
            key = key.reshape(
                B, self.num_attention_heads, L, self.attn_scalar_head
            ).transpose(1, 2)

        query = query.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
        key = key.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)

        # attn_weight: RBF(|r_ij|)
        if "nb" in self.attn_type:
            alpha = torch.einsum("bmhs,bnhs->bmnh", query, key)
        else:
            attn_weight = self.attn_weight2heads(attn_weight)
            alpha = attn_weight[..., : self.num_attention_heads] * torch.einsum(
                "bmhs,bnhs->bmnh", query, key
            )

        alpha = self.alpha_act(alpha)

        # if attn_mask is not None:
        #     # alpha = alpha.masked_fill(attn_mask, float('-inf'))
        alpha = alpha.masked_fill(attn_mask, -1e7)
        alpha = torch.nn.functional.softmax(alpha, dim=2)
        alpha = self.alpha_dropout(alpha)
        # alpha = alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8)  # B*L*L*head
        if "nb" not in self.attn_type:
            alpha = (
                alpha * attn_weight[..., self.num_attention_heads :]
            )  # add attn weight in messages
            # alpha = alpha.masked_fill(attn_mask, 0)

        if self.tp_type is None or self.tp_type == "None":
            node_scalars = None
        elif self.tp_type == "v2":
            edge_feature = self.gbf(
                edge_dis, batched_data["node_type_edge"].long()
            )  # B*L*L*-1
            edge_feature = edge_feature.sum(dim=-2)  # B*L*-1
            node_scalars = self.pos_embedding_proj(
                edge_feature
            ) + self.node_scalar_proj(node_irreps_input[..., : self.scalar_dim])
            if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
                outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
                outcell_index_0 = batched_data["outcell_index_0"]
                node_scalars = torch.cat(
                    [node_scalars, node_scalars[outcell_index_0, outcell_index]], dim=1
                )
        else:
            raise ValueError(f"tp_type should be v0 or v1 or v2 but got {self.tp_type}")

        node_pos_abs = torch.sqrt(torch.sum(node_pos**2, dim=-1, keepdim=True) + 1e-8)
        node_pos = node_pos / node_pos_abs
        value_1 = self.sep_tp(
            value, node_pos, node_scalars
        )  # B*N*irreps, B*N*3 uvw - > B*N*irreps
        value_1 = self.vec2heads(value_1)  # B*L*heads*irreps_head
        value_1 = -torch.einsum(
            "bmlh,blhi->bmhi",
            alpha
            * (node_pos_abs.unsqueeze(dim=1) / (edge_dis.unsqueeze(dim=-1) + 1e-8)),
            value_1,
        )
        value_1 = self.heads2vec(value_1)
        # alpha = alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8)  # B*L*L*head

        value = self.vec2heads(value)  # B*L*heads*irreps_head
        value = torch.einsum(
            "bmlh,blhi->bmhi",
            alpha
            * (
                node_pos_abs[:, :L].unsqueeze(dim=2)
                / (edge_dis.unsqueeze(dim=-1) + 1e-8)
            ),
            value,
        )
        value = self.heads2vec(value)
        value_0 = self.sep_tp(
            value[:, :L],
            node_pos[:, :L],
            None if node_scalars is None else node_scalars[:, :L],
        )  # TODO: different scalars?

        node_output = value_0 + value_1 + value
        node_output = self.proj(node_output)

        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)

        return node_output

        # B*L*L*head

    def extra_repr(self):
        output_str = "rescale_degree={}, ".format(self.rescale_degree)
        return output_str


@compile_mode("script")
class E2Attention_headatt(torch.nn.Module):
    """
    1. Message = Alpha * Value
    2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
    3. 0e -> Activation -> Inner Product -> (Alpha)
    4. (0e+1e+...) -> (Value)


    tp_type:
    v0: use irreps_node_input 0e
    v1: use sum of edge features
    v2: use sum of edge features and irreps_node_input 0e

    attn_type
    v0: initial implementation with attn_weight multiplication
    v1: GATv2 implementation without attn bias


    """

    def __init__(
        self,
        irreps_node_input="256x0e+64x1e+32x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+8x1e+4x2e",
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="v2",
        add_rope=True,
        layer_id=None,
        **kwargs,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e
        self.num_attention_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        self.irreps_head = (
            o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )
        # irreps_node_output,  attention will not change the input shape/embeding length
        self.irreps_node_output = self.irreps_node_input
        # new params
        self.tp_type = tp_type
        self.attn_type = attn_type

        self.gbf_attn = GaussianLayer(
            self.attn_weight_input_dim
        )  # default output_dim = 128
        self.node_embedding = nn.Embedding(256, 32)
        nn.init.uniform_(self.node_embedding.weight.data, -0.001, 0.001)

        self.attn_weight2heads = nn.Sequential(
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, 2 * num_attn_heads),
        )

        # self.attn_weight2heads = nn.Sequential(
        #     nn.Linear(attn_weight_input_dim, 2*num_attn_heads),
        #     # torch.nn.SiLU(),
        #     # nn.Linear(attn_weight_input_dim, 2 * num_attn_heads),
        # )

        self.multiheadatten_irreps = (
            (self.irreps_head * num_attn_heads).sort().irreps.simplify()
        )

        self.q_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )
        self.k_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )

        self.v_irlinear = IrrepsLinear(
            self.irreps_node_input,
            self.multiheadatten_irreps,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irreps

        self.vec2heads = Vec2AttnHeads(
            self.irreps_head,
            num_attn_heads,
        )
        self.heads2vec = AttnHeads2Vec(irreps_head=self.irreps_head)

        # alpha_dot, used for scalar to num_heads dimension
        # self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        self.alpha_act = SmoothLeakyReLU(0.2)  # used for attention_ij
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        self.sep_tp = None
        if self.nonlinear_message or self.rescale_degree:
            raise ValueError(
                "sorry, rescale_degree or non linear message passing is not supported in tp transformer"
            )

        else:
            if self.tp_type is None or self.tp_type == "None":
                self.sep_tp = SeparableFCTP(
                    self.multiheadatten_irreps,
                    "1x1e",
                    self.multiheadatten_irreps,
                    fc_neurons=None,
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=True,
                    mode="default",
                )
            else:
                self.sep_tp = SeparableFCTP(
                    self.multiheadatten_irreps,
                    "1x1e",
                    self.multiheadatten_irreps,
                    fc_neurons=[self.scalar_dim, self.scalar_dim, self.scalar_dim],
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=False,
                    mode="default",
                )

                if self.tp_type == "v2":
                    self.gbf = GaussianLayer(
                        self.attn_weight_input_dim
                    )  # default output_dim = 128
                    self.pos_embedding_proj = nn.Linear(
                        self.attn_weight_input_dim, self.scalar_dim
                    )
                    self.node_scalar_proj = nn.Linear(self.scalar_dim, self.scalar_dim)
                else:
                    raise ValueError(
                        f"attn_type should be None or v0 or v1 or v2 but got {self.attn_type}"
                    )

        self.proj = IrrepsLinear(
            (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep

        self.q_ir2scalar2 = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )
        self.k_ir2scalar2 = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )

        self.v_irlinear2 = IrrepsLinear(
            self.irreps_node_input,
            self.multiheadatten_irreps,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irreps

        self.attn_weight2heads2 = nn.Sequential(
            nn.Linear(attn_weight_input_dim + 32 + 32, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, num_attn_heads + 1),
        )

        self.proj2 = IrrepsLinear(
            self.irreps_head,
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep

        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout()
        self.add_rope = add_rope
        if add_rope:
            self.rot_emb = RotaryEmbedding(dim=self.attn_scalar_head)

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """
        irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
        B,L = 4,20
        node_pos = torch.randn(B,L,3)
        node_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))

        attn_scalar_head = 32
        func = GraphAttention(
            irreps_node_input = irreps_in,
            attn_weight_input_dim = 32, # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads = 8,
            attn_scalar_head = attn_scalar_head,
            irreps_head = "32x0e+8x1e+4x2e",
            rescale_degree=False,
            nonlinear_message=False,
            alpha_drop=0.1,
            proj_drop=0.1,
        )
        out = func(node_pos,
            irreps_in.randn(B,L,-1),
            node_dis,
            torch.randn(B,L,L,attn_scalar_head))
        print(out.shape)
        """

        # attn_weight f(|r_ij|): B*L*L*heads
        # edge_dis: B*L*L
        # node_input: B*L*irreps (irreps e.g. 128x0e+64x1e+32x2e)
        # ir2scalar: B*L*irreps -> N*L*hidden (hidden e.g. head*32)
        # attn_weight: B*L*L*rbf_dim
        B, L, _ = node_irreps_input.shape
        query = self.q_ir2scalar(node_irreps_input)
        key = self.k_ir2scalar(node_irreps_input)
        value = self.v_irlinear(node_irreps_input)  # B*L*irreps
        # embed = self.node_embedding(atomic_numbers)
        # if self.attn_type.startswith('v3'):
        # attn_weight = self.gbf_attn(edge_dis, batched_data["node_type_edge"].long()) # B*L*L*-1
        # torch.cat(
        #     [
        #         attn_weight,
        #         embed.reshape(B, L, 1, -1)
        #         .repeat(1, 1, L, 1),
        #         embed.reshape(B, 1, L, -1)
        #         .repeat(1, L, 1, 1),
        #     ],
        #     dim=-1,
        # ))

        if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
            outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
            outcell_index_0 = batched_data["outcell_index_0"]
            key = torch.cat([key, key[outcell_index_0, outcell_index]], dim=1)
            value = torch.cat([value, value[outcell_index_0, outcell_index]], dim=1)
            node_pos = torch.cat([node_pos, batched_data["expand_pos"].float()], dim=1)

        elif self.add_rope and batched_data["mol_type"] == 2:
            query = query.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
            key = key.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)
            query, key = self.rot_emb(
                query.transpose(1, 2).reshape(
                    B * self.num_attention_heads, L, self.attn_scalar_head
                ),
                key.transpose(1, 2).reshape(
                    B * self.num_attention_heads, L, self.attn_scalar_head
                ),
            )
            query = query.reshape(
                B, self.num_attention_heads, L, self.attn_scalar_head
            ).transpose(1, 2)
            key = key.reshape(
                B, self.num_attention_heads, L, self.attn_scalar_head
            ).transpose(1, 2)

        query = query.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
        key = key.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)

        attn_weight_new = self.attn_weight2heads(attn_weight)
        alpha = attn_weight_new[..., : self.num_attention_heads] * torch.einsum(
            "bmhs,bnhs->bmnh", query, key
        )

        alpha = self.alpha_act(alpha)

        # if attn_mask is not None:
        #     # alpha = alpha.masked_fill(attn_mask, float('-inf'))
        alpha = alpha.masked_fill(attn_mask, -1e7)
        alpha = torch.nn.functional.softmax(alpha, dim=2)
        alpha = self.alpha_dropout(alpha)
        # alpha = alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8)  # B*L*L*head
        alpha = alpha * attn_weight_new[..., self.num_attention_heads :]
        # add attn weight in messages
        # alpha = alpha.masked_fill(attn_mask, 0)

        if self.tp_type is None or self.tp_type == "None":
            node_scalars = None
        elif self.tp_type == "v2":
            edge_feature = self.gbf(
                edge_dis, batched_data["node_type_edge"].long()
            )  # B*L*L*-1
            edge_feature = edge_feature.sum(dim=-2)  # B*L*-1
            node_scalars = self.pos_embedding_proj(
                edge_feature
            ) + self.node_scalar_proj(node_irreps_input[..., : self.scalar_dim])
            if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
                outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
                outcell_index_0 = batched_data["outcell_index_0"]
                node_scalars = torch.cat(
                    [node_scalars, node_scalars[outcell_index_0, outcell_index]], dim=1
                )
        else:
            raise ValueError(f"tp_type should be v0 or v1 or v2 but got {self.tp_type}")

        node_pos_abs = torch.sqrt(torch.sum(node_pos**2, dim=-1, keepdim=True) + 1e-8)
        node_pos = node_pos / node_pos_abs
        value_1 = self.sep_tp(
            value, node_pos, node_scalars
        )  # B*N*irreps, B*N*3 uvw - > B*N*irreps
        value_1 = self.vec2heads(value_1)  # B*L*heads*irreps_head
        value_1 = -torch.einsum(
            "bmlh,blhi->bmhi",
            alpha
            * (node_pos_abs.unsqueeze(dim=1) / (edge_dis.unsqueeze(dim=-1) + 1e-8)),
            value_1,
        )
        value_1 = self.heads2vec(value_1)
        # alpha = alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8)  # B*L*L*head

        value = self.vec2heads(value)  # B*L*heads*irreps_head
        value = torch.einsum(
            "bmlh,blhi->bmhi",
            alpha
            * (
                node_pos_abs[:, :L].unsqueeze(dim=2)
                / (edge_dis.unsqueeze(dim=-1) + 1e-8)
            ),
            value,
        )
        value = self.heads2vec(value)
        value_0 = self.sep_tp(
            value[:, :L],
            node_pos[:, :L],
            None if node_scalars is None else node_scalars[:, :L],
        )  # TODO: different scalars?

        node_output = value_0 + value_1  # + value
        node_output = self.proj(node_output)

        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)

        query2 = self.q_ir2scalar(node_irreps_input)
        key2 = self.k_ir2scalar(node_irreps_input)
        value2 = self.v_irlinear(node_irreps_input)  # B*L*irreps
        query2 = query2.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
        key2 = key2.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)

        embed = self.node_embedding(atomic_numbers)
        attn_weight2 = torch.cat(
            [
                attn_weight,
                embed.reshape(B, L, 1, -1).repeat(1, 1, L, 1),
                embed.reshape(B, 1, L, -1).repeat(1, L, 1, 1),
            ],
            dim=-1,
        )
        attn_weight2 = self.attn_weight2heads2(attn_weight2)
        alpha2 = attn_weight2[..., : self.num_attention_heads] * torch.einsum(
            "bmhs,bnhs->bmnh", query2, key2
        )

        alpha2 = torch.nn.functional.sigmoid(
            attn_weight2[..., self.num_attention_heads :]
        ) * torch.nn.functional.softmax(alpha2, dim=3)

        alpha2 = alpha2.masked_fill(attn_mask, 0)
        value2 = self.vec2heads(value2)  # B*L*heads*irreps_head
        value2 = torch.einsum(
            "bmlh,blhi->bmi",
            alpha2,
            value2,
        )
        if self.layer_id is not None:
            if self.layer_id % 3 == 0:
                return node_output
            elif self.layer_id % 3 == 1:
                return self.proj2(value2)
            elif self.layer_id % 3 == 2:
                return node_output + self.proj2(value2)
        return node_output + self.proj2(value2)

        # B*L*L*head

    def extra_repr(self):
        output_str = "rescale_degree={}, ".format(self.rescale_degree)
        return output_str


@compile_mode("script")
class E2Attention_headatt2(torch.nn.Module):
    """
    1. Message = Alpha * Value
    2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
    3. 0e -> Activation -> Inner Product -> (Alpha)
    4. (0e+1e+...) -> (Value)


    tp_type:
    v0: use irreps_node_input 0e
    v1: use sum of edge features
    v2: use sum of edge features and irreps_node_input 0e

    attn_type
    v0: initial implementation with attn_weight multiplication
    v1: GATv2 implementation without attn bias


    """

    def __init__(
        self,
        irreps_node_input="256x0e+64x1e+32x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+8x1e+4x2e",
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="v2",
        add_rope=True,
        **kwargs,
    ):
        super().__init__()

        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e
        self.num_attention_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        self.irreps_head = (
            o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )
        # irreps_node_output,  attention will not change the input shape/embeding length
        self.irreps_node_output = self.irreps_node_input
        # new params
        self.tp_type = tp_type
        self.attn_type = attn_type

        self.gbf_attn = GaussianLayer(
            self.attn_weight_input_dim
        )  # default output_dim = 128
        self.node_embedding = nn.Embedding(256, 32)
        nn.init.uniform_(self.node_embedding.weight.data, -0.001, 0.001)

        self.attn_weight2heads = nn.Sequential(
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, 2 * num_attn_heads),
        )

        # self.attn_weight2heads = nn.Sequential(
        #     nn.Linear(attn_weight_input_dim, 2*num_attn_heads),
        #     # torch.nn.SiLU(),
        #     # nn.Linear(attn_weight_input_dim, 2 * num_attn_heads),
        # )

        self.multiheadatten_irreps = (
            (self.irreps_head * num_attn_heads).sort().irreps.simplify()
        )

        self.q_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )
        self.k_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )

        self.v_irlinear = IrrepsLinear(
            self.irreps_node_input,
            self.multiheadatten_irreps,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irreps

        self.vec2heads = Vec2AttnHeads(
            self.irreps_head,
            num_attn_heads,
        )
        self.heads2vec = AttnHeads2Vec(irreps_head=self.irreps_head)

        # alpha_dot, used for scalar to num_heads dimension
        # self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        self.alpha_act = SmoothLeakyReLU(0.2)  # used for attention_ij
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        self.sep_tp = None
        if self.nonlinear_message or self.rescale_degree:
            raise ValueError(
                "sorry, rescale_degree or non linear message passing is not supported in tp transformer"
            )

        else:
            if self.tp_type is None or self.tp_type == "None":
                self.sep_tp = SeparableFCTP(
                    self.multiheadatten_irreps,
                    "1x1e",
                    self.multiheadatten_irreps,
                    fc_neurons=None,
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=True,
                    mode="default",
                )
            else:
                self.sep_tp = SeparableFCTP(
                    self.multiheadatten_irreps,
                    "1x1e",
                    self.multiheadatten_irreps,
                    fc_neurons=[self.scalar_dim, self.scalar_dim, self.scalar_dim],
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=False,
                    mode="default",
                )

                if self.tp_type == "v2":
                    self.gbf = GaussianLayer(
                        self.attn_weight_input_dim
                    )  # default output_dim = 128
                    self.pos_embedding_proj = nn.Linear(
                        self.attn_weight_input_dim, self.scalar_dim
                    )
                    self.node_scalar_proj = nn.Linear(self.scalar_dim, self.scalar_dim)
                else:
                    raise ValueError(
                        f"attn_type should be None or v0 or v1 or v2 but got {self.attn_type}"
                    )

        self.proj = IrrepsLinear(
            (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep

        self.q_ir2scalar2 = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )
        self.k_ir2scalar2 = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )

        self.v_irlinear2 = IrrepsLinear(
            self.irreps_node_input,
            self.multiheadatten_irreps,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irreps

        self.attn_weight2heads2 = nn.Sequential(
            nn.Linear(attn_weight_input_dim + 32 + 32, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, num_attn_heads),
        )

        self.proj2 = IrrepsLinear(
            (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep

        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout()
        self.add_rope = add_rope
        if add_rope:
            self.rot_emb = RotaryEmbedding(dim=self.attn_scalar_head)

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """
        irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
        B,L = 4,20
        node_pos = torch.randn(B,L,3)
        node_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))

        attn_scalar_head = 32
        func = GraphAttention(
            irreps_node_input = irreps_in,
            attn_weight_input_dim = 32, # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads = 8,
            attn_scalar_head = attn_scalar_head,
            irreps_head = "32x0e+8x1e+4x2e",
            rescale_degree=False,
            nonlinear_message=False,
            alpha_drop=0.1,
            proj_drop=0.1,
        )
        out = func(node_pos,
            irreps_in.randn(B,L,-1),
            node_dis,
            torch.randn(B,L,L,attn_scalar_head))
        print(out.shape)
        """

        # attn_weight f(|r_ij|): B*L*L*heads
        # edge_dis: B*L*L
        # node_input: B*L*irreps (irreps e.g. 128x0e+64x1e+32x2e)
        # ir2scalar: B*L*irreps -> N*L*hidden (hidden e.g. head*32)
        # attn_weight: B*L*L*rbf_dim
        B, L, _ = node_irreps_input.shape
        query = self.q_ir2scalar(node_irreps_input)
        key = self.k_ir2scalar(node_irreps_input)
        value = self.v_irlinear(node_irreps_input)  # B*L*irreps
        # embed = self.node_embedding(atomic_numbers)
        # if self.attn_type.startswith('v3'):
        # attn_weight = self.gbf_attn(edge_dis, batched_data["node_type_edge"].long()) # B*L*L*-1
        # torch.cat(
        #     [
        #         attn_weight,
        #         embed.reshape(B, L, 1, -1)
        #         .repeat(1, 1, L, 1),
        #         embed.reshape(B, 1, L, -1)
        #         .repeat(1, L, 1, 1),
        #     ],
        #     dim=-1,
        # ))

        if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
            outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
            outcell_index_0 = batched_data["outcell_index_0"]
            key = torch.cat([key, key[outcell_index_0, outcell_index]], dim=1)
            value = torch.cat([value, value[outcell_index_0, outcell_index]], dim=1)
            node_pos = torch.cat([node_pos, batched_data["expand_pos"].float()], dim=1)

        elif self.add_rope and batched_data["mol_type"] == 2:
            query = query.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
            key = key.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)
            query, key = self.rot_emb(
                query.transpose(1, 2).reshape(
                    B * self.num_attention_heads, L, self.attn_scalar_head
                ),
                key.transpose(1, 2).reshape(
                    B * self.num_attention_heads, L, self.attn_scalar_head
                ),
            )
            query = query.reshape(
                B, self.num_attention_heads, L, self.attn_scalar_head
            ).transpose(1, 2)
            key = key.reshape(
                B, self.num_attention_heads, L, self.attn_scalar_head
            ).transpose(1, 2)

        query = query.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
        key = key.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)

        attn_weight_new = self.attn_weight2heads(attn_weight)
        alpha = attn_weight_new[..., : self.num_attention_heads] * torch.einsum(
            "bmhs,bnhs->bmnh", query, key
        )

        alpha = self.alpha_act(alpha)

        # if attn_mask is not None:
        #     # alpha = alpha.masked_fill(attn_mask, float('-inf'))
        alpha = alpha.masked_fill(attn_mask, -1e7)
        alpha = torch.nn.functional.softmax(alpha, dim=2)
        alpha = self.alpha_dropout(alpha)
        # alpha = alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8)  # B*L*L*head
        alpha = alpha * attn_weight_new[..., self.num_attention_heads :]
        # add attn weight in messages
        # alpha = alpha.masked_fill(attn_mask, 0)

        if self.tp_type is None or self.tp_type == "None":
            node_scalars = None
        elif self.tp_type == "v2":
            edge_feature = self.gbf(
                edge_dis, batched_data["node_type_edge"].long()
            )  # B*L*L*-1
            edge_feature = edge_feature.sum(dim=-2)  # B*L*-1
            node_scalars = self.pos_embedding_proj(
                edge_feature
            ) + self.node_scalar_proj(node_irreps_input[..., : self.scalar_dim])
            if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
                outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
                outcell_index_0 = batched_data["outcell_index_0"]
                node_scalars = torch.cat(
                    [node_scalars, node_scalars[outcell_index_0, outcell_index]], dim=1
                )
        else:
            raise ValueError(f"tp_type should be v0 or v1 or v2 but got {self.tp_type}")

        node_pos_abs = torch.sqrt(torch.sum(node_pos**2, dim=-1, keepdim=True) + 1e-8)
        node_pos = node_pos / node_pos_abs
        value_1 = self.sep_tp(
            value, node_pos, node_scalars
        )  # B*N*irreps, B*N*3 uvw - > B*N*irreps
        value_1 = self.vec2heads(value_1)  # B*L*heads*irreps_head
        value_1 = -torch.einsum(
            "bmlh,blhi->bmhi",
            alpha
            * (node_pos_abs.unsqueeze(dim=1) / (edge_dis.unsqueeze(dim=-1) + 1e-8)),
            value_1,
        )
        value_1 = self.heads2vec(value_1)
        # alpha = alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8)  # B*L*L*head

        value = self.vec2heads(value)  # B*L*heads*irreps_head
        value = torch.einsum(
            "bmlh,blhi->bmhi",
            alpha
            * (
                node_pos_abs[:, :L].unsqueeze(dim=2)
                / (edge_dis.unsqueeze(dim=-1) + 1e-8)
            ),
            value,
        )
        value = self.heads2vec(value)
        value_0 = self.sep_tp(
            value[:, :L],
            node_pos[:, :L],
            None if node_scalars is None else node_scalars[:, :L],
        )  # TODO: different scalars?

        node_output = value_0 + value_1  # + value
        node_output = self.proj(node_output)

        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)

        node_output_res = node_output

        query2 = self.q_ir2scalar(node_output)
        key2 = self.k_ir2scalar(node_output)
        value2 = self.v_irlinear(node_output)  # B*L*irreps
        query2 = query2.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
        key2 = key2.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)

        embed = self.node_embedding(atomic_numbers)
        attn_weight2 = torch.cat(
            [
                attn_weight,
                embed.reshape(B, L, 1, -1).repeat(1, 1, L, 1),
                embed.reshape(B, 1, L, -1).repeat(1, L, 1, 1),
            ],
            dim=-1,
        )
        attn_weight2 = self.attn_weight2heads2(attn_weight2)
        alpha2 = attn_weight2[..., : self.num_attention_heads] * torch.einsum(
            "bmhs,bnhs->bmnh", query2, key2
        )
        alpha2 = torch.nn.functional.softmax(alpha2, dim=3)
        alpha2 = alpha2.masked_fill(attn_mask, 0)
        value2 = self.vec2heads(value2)  # B*L*heads*irreps_head
        value2 = torch.einsum(
            "bmlh,blhi->bmhi",
            alpha2,
            value2,
        )
        value2 = self.heads2vec(value2)

        node_output = self.proj2(value2)
        return node_output + node_output_res

        # B*L*L*head

    def extra_repr(self):
        output_str = "rescale_degree={}, ".format(self.rescale_degree)
        return output_str


@compile_mode("script")
class E2AttentionSO2(torch.nn.Module):
    """
    1. Message = Alpha * Value
    2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
    3. 0e -> Activation -> Inner Product -> (Alpha)
    4. (0e+1e+...) -> (Value)


    tp_type:
    v0: use irreps_node_input 0e
    v1: use sum of edge features
    v2: use sum of edge features and irreps_node_input 0e

    attn_type
    v0: initial implementation with attn_weight multiplication
    v1: GATv2 implementation without attn bias


    """

    def __init__(
        self,
        irreps_node_input="256x0e+64x1e+32x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+8x1e+4x2e",
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="v2",
        add_rope=True,
        **kwargs,
    ):
        super().__init__()

        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e
        self.num_attention_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        self.irreps_head = (
            o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )
        # irreps_node_output,  attention will not change the input shape/embeding length
        self.irreps_node_output = self.irreps_node_input
        # new params
        self.tp_type = tp_type
        self.attn_type = attn_type

        self.gbf_attn = GaussianLayer(
            self.attn_weight_input_dim
        )  # default output_dim = 128
        self.node_embedding = nn.Embedding(256, 32)
        nn.init.uniform_(self.node_embedding.weight.data, -0.001, 0.001)

        self.attn_weight2heads = nn.Sequential(
            # nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            # nn.LayerNorm(attn_weight_input_dim),
            # torch.nn.SiLU(),
            # nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            # nn.LayerNorm(attn_weight_input_dim),
            # torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, 2 * num_attn_heads),
        )
        # self.attn_weight2heads = nn.Sequential(
        #     nn.Linear(attn_weight_input_dim, 2*num_attn_heads),
        #     # torch.nn.SiLU(),
        #     # nn.Linear(attn_weight_input_dim, 2 * num_attn_heads),
        # )

        self.multiheadatten_irreps = (
            (self.irreps_head * num_attn_heads).sort().irreps.simplify()
        )

        self.q_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )
        self.k_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )

        self.v_irlinear = IrrepsLinear(
            self.irreps_node_input,
            self.multiheadatten_irreps,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irreps

        self.vec2heads = Vec2AttnHeads(
            self.irreps_head,
            num_attn_heads,
        )
        self.heads2vec = AttnHeads2Vec(irreps_head=self.irreps_head)

        # alpha_dot, used for scalar to num_heads dimension
        # self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        self.alpha_act = SmoothLeakyReLU(0.2)  # used for attention_ij
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        self.sep_tp = None
        if self.nonlinear_message or self.rescale_degree:
            raise ValueError(
                "sorry, rescale_degree or non linear message passing is not supported in tp transformer"
            )

        else:
            self.rot_func = SO3_Rotation(
                lmax=max([i[1].l for i in self.multiheadatten_irreps]),
                irreps=self.multiheadatten_irreps,
            )
            self.so2_tp = SO2_Convolution(irreps_in=self.multiheadatten_irreps)

        self.proj = IrrepsLinear(
            (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout()
        self.add_rope = add_rope
        if add_rope:
            self.rot_emb = RotaryEmbedding(dim=self.attn_scalar_head)

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """
        irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
        B,L = 4,20
        node_pos = torch.randn(B,L,3)
        node_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))

        attn_scalar_head = 32
        func = GraphAttention(
            irreps_node_input = irreps_in,
            attn_weight_input_dim = 32, # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads = 8,
            attn_scalar_head = attn_scalar_head,
            irreps_head = "32x0e+8x1e+4x2e",
            rescale_degree=False,
            nonlinear_message=False,
            alpha_drop=0.1,
            proj_drop=0.1,
        )
        out = func(node_pos,
            irreps_in.randn(B,L,-1),
            node_dis,
            torch.randn(B,L,L,attn_scalar_head))
        print(out.shape)
        """

        # attn_weight f(|r_ij|): B*L*L*heads
        # edge_dis: B*L*L
        # node_input: B*L*irreps (irreps e.g. 128x0e+64x1e+32x2e)
        # ir2scalar: B*L*irreps -> N*L*hidden (hidden e.g. head*32)
        # attn_weight: B*L*L*rbf_dim
        B, L, _ = node_irreps_input.shape
        query = self.q_ir2scalar(node_irreps_input)
        key = self.k_ir2scalar(node_irreps_input)
        value = self.v_irlinear(node_irreps_input)  # B*L*irreps

        if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
            outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
            outcell_index_0 = batched_data["outcell_index_0"]
            key = torch.cat([key, key[outcell_index_0, outcell_index]], dim=1)
            value = torch.cat([value, value[outcell_index_0, outcell_index]], dim=1)
            node_pos = torch.cat([node_pos, batched_data["expand_pos"].float()], dim=1)

        elif self.add_rope and batched_data["mol_type"] == 2:
            query = query.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
            key = key.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)
            query, key = self.rot_emb(
                query.transpose(1, 2).reshape(
                    B * self.num_attention_heads, L, self.attn_scalar_head
                ),
                key.transpose(1, 2).reshape(
                    B * self.num_attention_heads, L, self.attn_scalar_head
                ),
            )
            query = query.reshape(
                B, self.num_attention_heads, L, self.attn_scalar_head
            ).transpose(1, 2)
            key = key.reshape(
                B, self.num_attention_heads, L, self.attn_scalar_head
            ).transpose(1, 2)

        self.rot_func.set_wigner(
            self.rot_func.init_edge_rot_mat(node_pos.reshape(-1, 3))
        )

        query = query.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
        key = key.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)

        # attn_weight: RBF(|r_ij|)
        if self.attn_type == "nb":
            alpha = torch.einsum("bmhs,bnhs->bmnh", query, key)
        else:
            attn_weight = self.attn_weight2heads(attn_weight)
            alpha = attn_weight[..., : self.num_attention_heads] * torch.einsum(
                "bmhs,bnhs->bmnh", query, key
            )

        alpha = self.alpha_act(alpha)

        # if attn_mask is not None:
        #     # alpha = alpha.masked_fill(attn_mask, float('-inf'))
        alpha = alpha.masked_fill(attn_mask, -1e7)
        alpha = torch.nn.functional.softmax(alpha, dim=2)
        alpha = self.alpha_dropout(alpha)
        # alpha = alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8)  # B*L*L*head
        if self.attn_type != "nb":
            alpha = (
                alpha * attn_weight[..., self.num_attention_heads :]
            )  # add attn weight in messages
            # alpha = alpha.masked_fill(attn_mask, 0)

        node_pos_abs = torch.sqrt(torch.sum(node_pos**2, dim=-1, keepdim=True) + 1e-8)
        node_pos = node_pos / node_pos_abs
        value_1_rot = self.rot_func.rotate(value)
        value_1_rot = self.so2_tp(value_1_rot)
        value_1 = self.rot_func.rotate_inv(value_1_rot)

        value_1 = self.vec2heads(value_1)  # B*L*heads*irreps_head
        value_1 = -torch.einsum(
            "bmlh,blhi->bmhi",
            alpha
            * (node_pos_abs.unsqueeze(dim=1) / (edge_dis.unsqueeze(dim=-1) + 1e-8)),
            value_1,
        )
        value_1 = self.heads2vec(value_1)
        # alpha = alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8)  # B*L*L*head

        value = self.vec2heads(value)  # B*L*heads*irreps_head
        value = torch.einsum(
            "bmlh,blhi->bmhi",
            alpha
            * (
                node_pos_abs[:, :L].unsqueeze(dim=2)
                / (edge_dis.unsqueeze(dim=-1) + 1e-8)
            ),
            value,
        )
        value = self.heads2vec(value)

        value_0_rot = self.rot_func.rotate(value)
        value_0_rot = self.so2_tp(value_0_rot)
        value_0 = self.rot_func.rotate_inv(value_0_rot)

        node_output = value_0 + value_1 + value
        node_output = self.proj(node_output)

        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)

        return node_output

        # B*L*L*head

    def extra_repr(self):
        output_str = "rescale_degree={}, ".format(self.rescale_degree)
        return output_str


@compile_mode("script")
class E2AttentionOrigin(torch.nn.Module):
    """
    1. Message = Alpha * Value
    2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
    3. 0e -> Activation -> Inner Product -> (Alpha)
    4. (0e+1e+...) -> (Value)


    tp_type:
    v0: use irreps_node_input 0e
    v1: use sum of edge features
    v2: use sum of edge features and irreps_node_input 0e

    attn_type
    v0: initial implementation with attn_weight multiplication
    v1: GATv2 implementation without attn bias


    """

    def __init__(
        self,
        irreps_node_input="256x0e+64x1e+32x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+8x1e+4x2e",
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="v2",
        add_rope=True,
        irreps_origin="1x0e+1x1e+1x2e",
        neighbor_weight=None,
        **kwargs,
    ):
        super().__init__()
        self.neighbor_weight = neighbor_weight
        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.irreps_origin = (
            o3.Irreps(irreps_origin)
            if isinstance(irreps_origin, str)
            else irreps_origin
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e
        self.num_attention_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        self.irreps_head = (
            o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )
        # irreps_node_output,  attention will not change the input shape/embeding length
        self.irreps_node_output = self.irreps_node_input
        # new params
        self.tp_type = tp_type
        self.attn_type = attn_type
        self.neighbor_weight = neighbor_weight
        if self.neighbor_weight is None:
            self.origin_tp = SeparableFCTP(
                self.irreps_node_input,
                self.irreps_origin,
                self.irreps_node_input,
                fc_neurons=[self.scalar_dim, self.scalar_dim, self.scalar_dim],
                use_activation=True,
                norm_layer=None,
                internal_weights=False,
            )
            self.origin_rbf = GaussianRadialBasisLayer(self.scalar_dim, cutoff=1.0)
        self.input_proj = IrrepsLinear(
            self.irreps_node_input,
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep
        self.message_proj1 = IrrepsLinear(
            self.irreps_node_input,
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep
        self.message_proj2 = IrrepsLinear(
            self.irreps_node_input,
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep
        # irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_input)
        # self.gate = Gate(
        #         irreps_scalars,
        #         [torch.nn.functional.silu for _, ir in irreps_scalars],  # scalar
        #         irreps_gates,
        #         [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
        #         irreps_gated,  # gated tensors
        #     )

        self.gbf_attn = GaussianLayer(
            self.attn_weight_input_dim
        )  # default output_dim = 128
        self.node_embedding = nn.Embedding(256, 32)
        nn.init.uniform_(self.node_embedding.weight.data, -0.001, 0.001)
        self.attn_weight2weight = nn.Sequential(
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, 2),
        )

        self.attn_weight2heads = nn.Sequential(
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, 2 * num_attn_heads),
        )

        self.multiheadatten_irreps = (
            (self.irreps_head * num_attn_heads).sort().irreps.simplify()
        )

        self.q_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )
        self.k_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )

        self.v_irlinear = IrrepsLinear(
            self.irreps_node_input,
            self.multiheadatten_irreps,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irreps

        self.vec2heads = Vec2AttnHeads(
            self.irreps_head,
            num_attn_heads,
        )
        self.heads2vec = AttnHeads2Vec(irreps_head=self.irreps_head)

        # alpha_dot, used for scalar to num_heads dimension
        # self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        self.alpha_act = SmoothLeakyReLU(0.2)  # used for attention_ij
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        self.sep_tp = None
        if self.nonlinear_message or self.rescale_degree:
            raise ValueError(
                "sorry, rescale_degree or non linear message passing is not supported in tp transformer"
            )

        else:
            if self.tp_type is None or self.tp_type == "None":
                self.sep_tp = SeparableFCTP(
                    self.multiheadatten_irreps,
                    "1x1e",
                    self.multiheadatten_irreps,
                    fc_neurons=None,
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=True,
                )
            else:
                self.sep_tp = SeparableFCTP(
                    self.multiheadatten_irreps,
                    "1x1e",
                    self.multiheadatten_irreps,
                    fc_neurons=[self.scalar_dim, self.scalar_dim, self.scalar_dim],
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=False,
                )

                if self.tp_type == "v2":
                    self.gbf = GaussianLayer(
                        self.attn_weight_input_dim
                    )  # default output_dim = 128
                    self.pos_embedding_proj = nn.Linear(
                        self.attn_weight_input_dim, self.scalar_dim
                    )
                    self.node_scalar_proj = nn.Linear(self.scalar_dim, self.scalar_dim)
                else:
                    raise ValueError(
                        f"attn_type should be None or v0 or v1 or v2 but got {self.attn_type}"
                    )

        self.proj = IrrepsLinear(
            (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout()
        self.add_rope = add_rope
        if add_rope:
            self.rot_emb = RotaryEmbedding(dim=self.attn_scalar_head)

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """
        irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
        B,L = 4,20
        node_pos = torch.randn(B,L,3)
        node_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))

        attn_scalar_head = 32
        func = GraphAttention(
            irreps_node_input = irreps_in,
            attn_weight_input_dim = 32, # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads = 8,
            attn_scalar_head = attn_scalar_head,
            irreps_head = "32x0e+8x1e+4x2e",
            rescale_degree=False,
            nonlinear_message=False,
            alpha_drop=0.1,
            proj_drop=0.1,
        )
        out = func(node_pos,
            irreps_in.randn(B,L,-1),
            node_dis,
            torch.randn(B,L,L,attn_scalar_head))
        print(out.shape)
        """

        # attn_weight f(|r_ij|): B*L*L*heads
        # edge_dis: B*L*L
        # node_input: B*L*irreps (irreps e.g. 128x0e+64x1e+32x2e)
        # ir2scalar: B*L*irreps -> N*L*hidden (hidden e.g. head*32)
        # attn_weight: B*L*L*rbf_dim
        B, L, _ = node_irreps_input.shape
        # # origin self tp
        if self.neighbor_weight is None:
            origin_scalars = self.origin_rbf(node_pos.norm(dim=-1).reshape(-1)).reshape(
                B, L, -1
            )
            origin_sh = o3.spherical_harmonics(
                l=self.irreps_origin,
                x=node_pos,  # [:, [1, 2, 0]],
                normalize=True,
                normalization="component",
            )
            node_irreps_input = self.origin_tp(
                node_irreps_input, origin_sh, origin_scalars
            )
        else:
            # neighbor convolution
            if self.neighbor_weight == "identity":
                alpha = torch.ones_like(edge_dis)
                alpha = alpha.masked_fill(attn_mask.squeeze(-1), 0.0)
                alpha = alpha / torch.sum(alpha, 2, keepdim=True)  # B*L*L*head
            elif self.neighbor_weight == "distance":
                alpha = edge_dis.masked_fill(attn_mask.squeeze(-1), -1e6)
                alpha = torch.softmax(alpha, 2)  # B*L*L*head
                alpha = self.alpha_dropout(alpha)
            elif self.neighbor_weight == "attention0":
                dist_weight = self.attn_weight2weight(attn_weight)
                alpha = dist_weight[..., 0]
                alpha = alpha.masked_fill(attn_mask.squeeze(-1), -1e6)
                alpha = torch.softmax(alpha, 2)
                alpha = self.alpha_dropout(alpha)
            elif self.neighbor_weight == "attention1":
                dist_weight = self.attn_weight2weight(attn_weight)
                alpha = dist_weight[..., 0]
                alpha = alpha.masked_fill(attn_mask.squeeze(-1), -1e6)
                alpha = torch.softmax(alpha, 2)
                alpha = self.alpha_dropout(alpha)
                alpha = alpha * attn_weight[..., 1]
            else:
                raise ValueError(
                    f"neighbor_weight should be identity or distance but got {self.neighbor_weight}"
                )
            message = self.message_proj1(node_irreps_input)  # B*L*irreps
            message = torch.einsum("bmn,bnd->bmd", alpha, message)
            node_irreps_input = self.input_proj(node_irreps_input) + self.message_proj2(
                message
            )

        query = self.q_ir2scalar(node_irreps_input)
        key = self.k_ir2scalar(node_irreps_input)
        value = self.v_irlinear(node_irreps_input)  # B*L*irreps
        # embed = self.node_embedding(atomic_numbers)
        # if self.attn_type.startswith('v3'):
        # attn_weight = self.gbf_attn(edge_dis, batched_data["node_type_edge"].long()) # B*L*L*-1
        attn_weight = self.attn_weight2heads(attn_weight)
        # torch.cat(
        #     [
        #         attn_weight,
        #         embed.reshape(B, L, 1, -1)
        #         .repeat(1, 1, L, 1),
        #         embed.reshape(B, 1, L, -1)
        #         .repeat(1, L, 1, 1),
        #     ],
        #     dim=-1,
        # ))

        if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
            outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
            outcell_index_0 = batched_data["outcell_index_0"]
            key = torch.cat([key, key[outcell_index_0, outcell_index]], dim=1)
            value = torch.cat([value, value[outcell_index_0, outcell_index]], dim=1)
            node_pos = torch.cat([node_pos, batched_data["expand_pos"].float()], dim=1)

        elif self.add_rope and batched_data["mol_type"] == 2:
            query = query.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
            key = key.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)
            query, key = self.rot_emb(
                query.transpose(1, 2).reshape(
                    B * self.num_attention_heads, L, self.attn_scalar_head
                ),
                key.transpose(1, 2).reshape(
                    B * self.num_attention_heads, L, self.attn_scalar_head
                ),
            )
            query = query.reshape(
                B, self.num_attention_heads, L, self.attn_scalar_head
            ).transpose(1, 2)
            key = key.reshape(
                B, self.num_attention_heads, L, self.attn_scalar_head
            ).transpose(1, 2)

        query = query.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
        key = key.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)

        head_dim = key.size(-1)  # or query.size(-1), since they should be the same
        scaling_factor = math.sqrt(head_dim)  # sqrt(d_k)
        # scaling_factor = 1.0
        # attn_weight: RBF(|r_ij|)
        alpha = (
            attn_weight[..., : self.num_attention_heads]
            * torch.einsum("bmhs,bnhs->bmnh", query, key)
            / scaling_factor
        )
        alpha = self.alpha_act(alpha)

        # if attn_mask is not None:
        #     # alpha = alpha.masked_fill(attn_mask, float('-inf'))
        alpha = alpha.masked_fill(attn_mask, -1e6)
        alpha = torch.nn.functional.softmax(alpha, dim=2)
        alpha = self.alpha_dropout(alpha)
        alpha = alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8)  # B*L*L*head
        alpha = (
            alpha * attn_weight[..., self.num_attention_heads :]
        )  # add attn weight in messages
        # alpha = alpha.masked_fill(attn_mask, 0)

        if self.tp_type is None or self.tp_type == "None":
            node_scalars = None
        elif self.tp_type == "v2":
            edge_feature = self.gbf(
                edge_dis, batched_data["node_type_edge"].long()
            )  # B*L*L*-1
            edge_feature = edge_feature.sum(dim=-2)  # B*L*-1
            node_scalars = self.pos_embedding_proj(
                edge_feature
            ) + self.node_scalar_proj(node_irreps_input[..., : self.scalar_dim])
            if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
                outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
                outcell_index_0 = batched_data["outcell_index_0"]
                node_scalars = torch.cat(
                    [node_scalars, node_scalars[outcell_index_0, outcell_index]], dim=1
                )
        else:
            raise ValueError(f"tp_type should be v0 or v1 or v2 but got {self.tp_type}")

        value_1 = self.sep_tp(
            value, node_pos, node_scalars
        )  # B*N*irreps, B*N*3 uvw - > B*N*irreps
        value_1 = self.vec2heads(value_1)  # B*L*heads*irreps_head
        value_1 = -torch.einsum("bmlh,blhi->bmhi", alpha, value_1)
        value_1 = self.heads2vec(value_1)

        # value_0 = self.vec2heads(value) # B*L*heads*irreps_head
        # value_0 = torch.einsum("bmlh,blhi->bmhi",alpha,value_0)
        # value_0 = self.heads2vec(value_0)
        # value_0 = self.sep_tp(value_0,node_pos,node_scalars) # TODO: different scalars?
        value = self.vec2heads(value)  # B*L*heads*irreps_head
        value = torch.einsum("bmlh,blhi->bmhi", alpha, value)
        value = self.heads2vec(value)
        value_0 = self.sep_tp(
            value[:, :L],
            node_pos[:, :L],
            None if node_scalars is None else node_scalars[:, :L],
        )  # TODO: different scalars?

        node_output = value_0 + value_1 + value
        # if self.rescale_degree:
        #     degree = torch_geometric.utils.degree(
        #         edge_dst, num_nodes=node_input.shape[0], dtype=node_input.dtype
        #     )
        #     degree = degree.view(-1, 1)
        #     node_output = node_output * degree
        node_output = self.proj(node_output)

        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)

        return node_output

        # B*L*L*head

    def extra_repr(self):
        output_str = super(E2AttentionOrigin, self).extra_repr()
        output_str = output_str + "rescale_degree={}, ".format(self.rescale_degree)
        return output_str


@compile_mode("script")
class E2AttentionTriple(torch.nn.Module):
    """
    1. Message = Alpha * Value
    2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
    3. 0e -> Activation -> Inner Product -> (Alpha)
    4. (0e+1e+...) -> (Value)


    tp_type:
    v0: use irreps_node_input 0e
    v1: use sum of edge features
    v2: use sum of edge features and irreps_node_input 0e

    attn_type
    v0: initial implementation with attn_weight multiplication
    v1: GATv2 implementation without attn bias


    """

    def __init__(
        self,
        irreps_node_input="256x0e+64x1e+32x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+8x1e+4x2e",
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="v2",
        add_rope=True,
        irreps_origin="1x0e+1x1e+1x2e",
        neighbor_weight=None,
        **kwargs,
    ):
        super().__init__()
        self.neighbor_weight = neighbor_weight
        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.irreps_origin = (
            o3.Irreps(irreps_origin)
            if isinstance(irreps_origin, str)
            else irreps_origin
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e
        self.num_attention_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        self.irreps_head = (
            o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )
        # irreps_node_output,  attention will not change the input shape/embeding length
        self.irreps_node_output = self.irreps_node_input
        # new params
        self.tp_type = tp_type
        self.attn_type = attn_type
        self.neighbor_weight = neighbor_weight
        if self.neighbor_weight is None:
            self.origin_tp = SeparableFCTP(
                self.irreps_node_input,
                self.irreps_origin,
                self.irreps_node_input,
                fc_neurons=[self.scalar_dim, self.scalar_dim, self.scalar_dim],
                use_activation=True,
                norm_layer=None,
                internal_weights=False,
            )
            self.origin_rbf = GaussianRadialBasisLayer(self.scalar_dim, cutoff=1.0)
        self.input_proj = IrrepsLinear(
            self.irreps_node_input,
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep
        self.message_proj1 = IrrepsLinear(
            self.irreps_node_input,
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep
        self.message_proj2 = IrrepsLinear(
            self.irreps_node_input,
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep
        # irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_input)
        # self.gate = Gate(
        #         irreps_scalars,
        #         [torch.nn.functional.silu for _, ir in irreps_scalars],  # scalar
        #         irreps_gates,
        #         [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
        #         irreps_gated,  # gated tensors
        #     )

        self.gbf_attn = GaussianLayer(
            self.attn_weight_input_dim
        )  # default output_dim = 128
        self.node_embedding = nn.Embedding(256, 32)
        nn.init.uniform_(self.node_embedding.weight.data, -0.001, 0.001)
        self.attn_weight2weight = nn.Sequential(
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, 2),
        )

        self.attn_weight2heads = nn.Sequential(
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, 2 * num_attn_heads),
        )

        self.multiheadatten_irreps = (
            (self.irreps_head * num_attn_heads).sort().irreps.simplify()
        )

        self.q_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )
        self.k_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )

        self.v_irlinear = IrrepsLinear(
            self.irreps_node_input,
            self.multiheadatten_irreps,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irreps

        self.vec2heads = Vec2AttnHeads(
            self.irreps_head,
            num_attn_heads,
        )
        self.heads2vec = AttnHeads2Vec(irreps_head=self.irreps_head)

        # alpha_dot, used for scalar to num_heads dimension
        # self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        self.alpha_act = SmoothLeakyReLU(0.2)  # used for attention_ij
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        self.sep_tp = None
        if self.nonlinear_message or self.rescale_degree:
            raise ValueError(
                "sorry, rescale_degree or non linear message passing is not supported in tp transformer"
            )

        else:
            if self.tp_type is None or self.tp_type == "None":
                self.sep_tp = SeparableFCTP(
                    self.multiheadatten_irreps,
                    "1x1e",
                    self.multiheadatten_irreps,
                    fc_neurons=None,
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=True,
                )
            else:
                self.sep_tp0 = SeparableFCTP(
                    self.multiheadatten_irreps,
                    self.irreps_origin,
                    self.multiheadatten_irreps,
                    fc_neurons=[self.scalar_dim, self.scalar_dim, self.scalar_dim],
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=False,
                )
                self.sep_tp1 = SeparableFCTP(
                    self.multiheadatten_irreps,
                    self.irreps_origin,
                    self.multiheadatten_irreps,
                    fc_neurons=[self.scalar_dim, self.scalar_dim, self.scalar_dim],
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=False,
                )
                self.sep_tp2 = SeparableFCTP(
                    self.multiheadatten_irreps,
                    "1x1e",
                    self.multiheadatten_irreps,
                    fc_neurons=None,
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=True,
                )
                if self.tp_type == "v2":
                    self.gbf = GaussianLayer(
                        self.attn_weight_input_dim
                    )  # default output_dim = 128
                    self.pos_embedding_proj = nn.Linear(
                        self.attn_weight_input_dim, self.scalar_dim
                    )
                    self.node_scalar_proj = nn.Linear(self.scalar_dim, self.scalar_dim)
                else:
                    raise ValueError(
                        f"attn_type should be None or v0 or v1 or v2 but got {self.attn_type}"
                    )
        self.proj0 = IrrepsLinear(
            (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
            self.irreps_node_input,
            bias=True,
            act="identity",
        )  # B*L*irrep
        self.proj1 = IrrepsLinear(
            (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
            self.irreps_node_input,
            bias=True,
            act="identity",
        )  # B*L*irrep
        self.proj2 = IrrepsLinear(
            (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
            self.irreps_node_input,
            bias=True,
            act="identity",
        )  # B*L*irrep
        self.proj = IrrepsLinear(
            # self.irreps_node_input,
            (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
            self.irreps_node_input,
            bias=True,
            act="identity",
        )  # B*L*irrep   for value

        self.proj_out = IrrepsLinear(
            self.irreps_node_input,
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout()
        self.add_rope = add_rope
        if add_rope:
            self.rot_emb = RotaryEmbedding(dim=self.attn_scalar_head)

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """
        irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
        B,L = 4,20
        node_pos = torch.randn(B,L,3)
        node_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))

        attn_scalar_head = 32
        func = GraphAttention(
            irreps_node_input = irreps_in,
            attn_weight_input_dim = 32, # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads = 8,
            attn_scalar_head = attn_scalar_head,
            irreps_head = "32x0e+8x1e+4x2e",
            rescale_degree=False,
            nonlinear_message=False,
            alpha_drop=0.1,
            proj_drop=0.1,
        )
        out = func(node_pos,
            irreps_in.randn(B,L,-1),
            node_dis,
            torch.randn(B,L,L,attn_scalar_head))
        print(out.shape)
        """

        # attn_weight f(|r_ij|): B*L*L*heads
        # edge_dis: B*L*L
        # node_input: B*L*irreps (irreps e.g. 128x0e+64x1e+32x2e)
        # ir2scalar: B*L*irreps -> N*L*hidden (hidden e.g. head*32)
        # attn_weight: B*L*L*rbf_dim
        B, L, _ = node_irreps_input.shape
        # # origin self tp
        # if self.neighbor_weight is None:
        #     origin_scalars = self.origin_rbf(node_pos.norm(dim=-1).reshape(-1)).reshape(B, L, -1)
        #     origin_sh = o3.spherical_harmonics(
        #         l=self.irreps_origin,
        #         x=node_pos,  # [:, [1, 2, 0]],
        #         normalize=True,
        #         normalization="component",
        #     )
        #     node_irreps_input = self.origin_tp(node_irreps_input, origin_sh, origin_scalars)
        # else:
        #     # neighbor convolution
        #     if  self.neighbor_weight=="identity":
        #         alpha = torch.ones_like(edge_dis)
        #         alpha = alpha.masked_fill(attn_mask.squeeze(-1), 0.0)
        #         alpha = alpha / torch.sum(alpha, 2, keepdim=True) # B*L*L*head
        #     elif self.neighbor_weight == "distance":
        #         alpha = edge_dis.masked_fill(attn_mask.squeeze(-1), -1e6)
        #         alpha = torch.softmax(alpha, 2) # B*L*L*head
        #         alpha = self.alpha_dropout(alpha)
        #     elif self.neighbor_weight == "attention0":
        #         dist_weight = self.attn_weight2weight(attn_weight)
        #         alpha = dist_weight[..., 0]
        #         alpha = alpha.masked_fill(attn_mask.squeeze(-1), -1e6)
        #         alpha = torch.softmax(alpha, 2)
        #         alpha = self.alpha_dropout(alpha)
        #     elif self.neighbor_weight == "attention1":
        #         dist_weight = self.attn_weight2weight(attn_weight)
        #         alpha = dist_weight[..., 0]
        #         alpha = alpha.masked_fill(attn_mask.squeeze(-1), -1e6)
        #         alpha = torch.softmax(alpha, 2)
        #         alpha = self.alpha_dropout(alpha)
        #         alpha = alpha * attn_weight[..., 1]
        #     else:
        #         raise ValueError(f"neighbor_weight should be identity or distance but got {self.neighbor_weight}")
        #     message = self.message_proj1(node_irreps_input) # B*L*irreps
        #     message = torch.einsum("bmn,bnd->bmd", alpha, message)
        #     node_irreps_input = self.input_proj(node_irreps_input) + self.message_proj2(message)

        node_sh = o3.spherical_harmonics(
            l=self.irreps_origin,
            x=node_pos,  # [:, [1, 2, 0]],
            normalize=True,
            normalization="component",
        )

        query = self.q_ir2scalar(node_irreps_input)
        key = self.k_ir2scalar(node_irreps_input)
        value = self.v_irlinear(node_irreps_input)  # B*L*irreps
        # embed = self.node_embedding(atomic_numbers)
        # if self.attn_type.startswith('v3'):
        # attn_weight = self.gbf_attn(edge_dis, batched_data["node_type_edge"].long()) # B*L*L*-1
        attn_weight = self.attn_weight2heads(attn_weight)
        # torch.cat(
        #     [
        #         attn_weight,
        #         embed.reshape(B, L, 1, -1)
        #         .repeat(1, 1, L, 1),
        #         embed.reshape(B, 1, L, -1)
        #         .repeat(1, L, 1, 1),
        #     ],
        #     dim=-1,
        # ))

        if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
            outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
            outcell_index_0 = batched_data["outcell_index_0"]
            key = torch.cat([key, key[outcell_index_0, outcell_index]], dim=1)
            value = torch.cat([value, value[outcell_index_0, outcell_index]], dim=1)
            node_pos = torch.cat([node_pos, batched_data["expand_pos"].float()], dim=1)

        elif self.add_rope and batched_data["mol_type"] == 2:
            query = query.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
            key = key.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)
            query, key = self.rot_emb(
                query.transpose(1, 2).reshape(
                    B * self.num_attention_heads, L, self.attn_scalar_head
                ),
                key.transpose(1, 2).reshape(
                    B * self.num_attention_heads, L, self.attn_scalar_head
                ),
            )
            query = query.reshape(
                B, self.num_attention_heads, L, self.attn_scalar_head
            ).transpose(1, 2)
            key = key.reshape(
                B, self.num_attention_heads, L, self.attn_scalar_head
            ).transpose(1, 2)

        query = query.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
        key = key.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)

        head_dim = key.size(-1)  # or query.size(-1), since they should be the same
        scaling_factor = math.sqrt(head_dim)  # sqrt(d_k)
        # scaling_factor = 1.0
        # attn_weight: RBF(|r_ij|)
        alpha = (
            attn_weight[..., : self.num_attention_heads]
            * torch.einsum("bmhs,bnhs->bmnh", query, key)
            / scaling_factor
        )
        alpha = self.alpha_act(alpha)

        # if attn_mask is not None:
        #     # alpha = alpha.masked_fill(attn_mask, float('-inf'))
        alpha = alpha.masked_fill(attn_mask, -1e6)
        alpha = torch.nn.functional.softmax(alpha, dim=2)
        alpha = self.alpha_dropout(alpha)
        alpha = alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8)  # B*L*L*head
        alpha = (
            alpha * attn_weight[..., self.num_attention_heads :]
        )  # add attn weight in messages
        # alpha = alpha.masked_fill(attn_mask, 0)

        if self.tp_type is None or self.tp_type == "None":
            node_scalars = None
        elif self.tp_type == "v2":
            edge_feature = self.gbf(
                edge_dis, batched_data["node_type_edge"].long()
            )  # B*L*L*-1
            edge_feature = edge_feature.sum(dim=-2)  # B*L*-1
            node_scalars = self.pos_embedding_proj(
                edge_feature
            ) + self.node_scalar_proj(node_irreps_input[..., : self.scalar_dim])
            if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
                outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
                outcell_index_0 = batched_data["outcell_index_0"]
                node_scalars = torch.cat(
                    [node_scalars, node_scalars[outcell_index_0, outcell_index]], dim=1
                )
        else:
            raise ValueError(f"tp_type should be v0 or v1 or v2 but got {self.tp_type}")

        value_1 = self.sep_tp1(
            value, node_sh, node_scalars
        )  # B*N*irreps, B*N*3 uvw - > B*N*irreps
        value_1 = self.vec2heads(value_1)  # B*L*heads*irreps_head
        value_1 = torch.einsum("bmlh,blhi->bmhi", alpha, value_1)
        value_1 = self.heads2vec(value_1)

        value_21 = self.sep_tp2(
            value, node_pos, None
        )  # B*N*irreps, B*N*3 uvw - > B*N*irreps
        value_21 = self.vec2heads(value_21)  # B*L*heads*irreps_head
        value_21 = -torch.einsum("bmlh,blhi->bmhi", alpha, value_21)
        value_21 = self.heads2vec(value_21)

        value = self.vec2heads(value)  # B*L*heads*irreps_head
        value = torch.einsum("bmlh,blhi->bmhi", alpha, value)
        value = self.heads2vec(value)
        value_0 = self.sep_tp0(
            value[:, :L],
            node_sh[:, :L],
            None if node_scalars is None else node_scalars[:, :L],
        )  # TODO: different scalars?

        value_20 = self.sep_tp2(
            value[:, :L], node_pos[:, :L], None
        )  # TODO: different scalars?

        # node_output = self.proj0(value_0) + self.proj1(value_1)+ self.proj2(value_20 + value_21)  + value # v0
        # node_output = self.proj0(value_0) + self.proj1(value_1) + value  # v1
        # node_output = value_0 + value_1 + value  # v1.1
        # node_output = self.proj0(value_0) + self.proj1(value_1) + self.proj(value)  # v1.2
        # node_output = self.proj0(value_0) + self.proj(value)  # v1.3
        # node_output = self.proj1(value_1) + self.proj(value)  # v1.4
        # node_output = self.proj2(value_20 + value_21)  + value  # v2
        # node_output = value_20 + value_21  + value  # v2.1
        # node_output = self.proj2(value_20 + value_21) + self.proj(value)  # v2.2
        # node_output = value_0 + value_1+ self.proj2(value_20 + value_21)  + value # v3
        # node_output = value_0 + value_1+ value_20 + value_21  + value # v3.1
        node_output = (
            self.proj0(value_0)
            + self.proj1(value_1)
            + self.proj2(value_20 + value_21)
            + self.proj(value)
        )  # v3.2 (Current SOTA)
        # if self.rescale_degree:
        #     degree = torch_geometric.utils.degree(
        #         edge_dst, num_nodes=node_input.shape[0], dtype=node_input.dtype
        #     )
        #     degree = degree.view(-1, 1)
        #     node_output = node_output * degree
        node_output = self.proj_out(node_output)

        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)

        return node_output

        # B*L*L*head

    def extra_repr(self):
        output_str = super(E2AttentionTriple, self).extra_repr()
        output_str = output_str + "rescale_degree={}, ".format(self.rescale_degree)
        return output_str


class CosineCutoff(torch.nn.Module):
    r"""Appies a cosine cutoff to the input distances.

    .. math::
        \text{cutoffs} =
        \begin{cases}
        0.5 * (\cos(\frac{\text{distances} * \pi}{\text{cutoff}}) + 1.0),
        & \text{if } \text{distances} < \text{cutoff} \\
        0, & \text{otherwise}
        \end{cases}

    Args:
        cutoff (float): A scalar that determines the point at which the cutoff
            is applied.
    """

    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances):
        r"""Applies a cosine cutoff to the input distances.

        Args:
            distances (torch.Tensor): A tensor of distances.

        Returns:
            cutoffs (torch.Tensor): A tensor where the cosine function
                has been applied to the distances,
                but any values that exceed the cutoff are set to 0.
        """
        cutoffs = 0.5 * ((distances * math.pi / self.cutoff).cos() + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class E2AttentionLinear(torch.nn.Module):
    """
    Use IrrepsLinear with external weights W(|r_i|)


    """

    def __init__(
        self,
        irreps_node_input="256x0e+64x1e+32x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+8x1e+4x2e",
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="v2",
        add_rope=True,
        irreps_origin="1x0e+1x1e+1x2e",
        neighbor_weight=None,
        atom_type_cnt=256,
        **kwargs,
    ):
        super().__init__()
        self.atom_type_cnt = atom_type_cnt
        self.neighbor_weight = neighbor_weight
        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.irreps_origin = (
            o3.Irreps(irreps_origin)
            if isinstance(irreps_origin, str)
            else irreps_origin
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e
        self.num_attention_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        self.irreps_head = (
            o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )
        # irreps_node_output,  attention will not change the input shape/embeding length
        self.irreps_node_output = self.irreps_node_input
        # new params
        self.tp_type = tp_type
        self.attn_type = attn_type
        self.neighbor_weight = neighbor_weight
        # if self.neighbor_weight is None:
        #     self.origin_tp = SeparableFCTP(self.irreps_node_input,
        #                                 self.irreps_origin,
        #                                 self.irreps_node_input,
        #                                 fc_neurons=[self.scalar_dim,self.scalar_dim,self.scalar_dim],
        #                                 use_activation=True,
        #                                 norm_layer=None,
        #                                 internal_weights=False)
        #     self.origin_rbf =  GaussianRadialBasisLayer(self.scalar_dim, cutoff=1.0)
        # self.input_proj = IrrepsLinear(
        #     self.irreps_node_input,
        #     self.irreps_node_input,
        #     bias=True,
        #     act="smoothleakyrelu",
        # )  # B*L*irrep
        # self.message_proj1 = IrrepsLinear(
        #     self.irreps_node_input,
        #     self.irreps_node_input,
        #     bias=True,
        #     act="smoothleakyrelu",
        # )  # B*L*irrep
        # self.message_proj2 = IrrepsLinear(
        #     self.irreps_node_input,
        #     self.irreps_node_input,
        #     bias=True,
        #     act="smoothleakyrelu",
        # )  # B*L*irrep
        # irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_input)
        # self.gate = Gate(
        #         irreps_scalars,
        #         [torch.nn.functional.silu for _, ir in irreps_scalars],  # scalar
        #         irreps_gates,
        #         [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
        #         irreps_gated,  # gated tensors
        #     )

        self.gbf_attn = GaussianLayer(
            self.attn_weight_input_dim
        )  # default output_dim = 128
        self.node_embedding = nn.Embedding(256, 32)
        nn.init.uniform_(self.node_embedding.weight.data, -0.001, 0.001)
        self.attn_weight2weight = nn.Sequential(
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, 2),
        )

        self.attn_weight2heads = nn.Sequential(
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
            nn.LayerNorm(attn_weight_input_dim),
            torch.nn.SiLU(),
            nn.Linear(attn_weight_input_dim, 2 * num_attn_heads),
        )

        self.multiheadatten_irreps = (
            (self.irreps_head * num_attn_heads).sort().irreps.simplify()
        )

        self.q_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )
        self.k_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )

        # self.v_irlinear = IrrepsLinear(
        #     self.irreps_node_input,
        #     self.multiheadatten_irreps,
        #     bias=True,
        #     act="smoothleakyrelu",
        # )  # B*L*irreps v0

        self.vec2heads = Vec2AttnHeads(
            self.irreps_head,
            num_attn_heads,
        )
        self.heads2vec = AttnHeads2Vec(irreps_head=self.irreps_head)

        # alpha_dot, used for scalar to num_heads dimension
        # self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        self.alpha_act = SmoothLeakyReLU(0.2)  # used for attention_ij
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        self.sep_tp = None
        if self.nonlinear_message or self.rescale_degree:
            raise ValueError(
                "sorry, rescale_degree or non linear message passing is not supported in tp transformer"
            )

        else:
            if self.tp_type is None or self.tp_type == "None":
                self.sep_tp = SeparableFCTP(
                    self.multiheadatten_irreps,
                    "1x1e",
                    self.multiheadatten_irreps,
                    fc_neurons=None,
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=True,
                )
            else:
                self.sep_tp2 = SeparableFCTP(
                    self.multiheadatten_irreps,
                    "1x1e",
                    self.multiheadatten_irreps,
                    fc_neurons=None,
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=True,
                )
                self.tp_linear = SeparableFCTP(
                    #    self.multiheadatten_irreps,
                    self.irreps_node_input,  # v0.1
                    o3.Irreps("1x0e"),
                    self.multiheadatten_irreps,
                    fc_neurons=[
                        self.attn_weight_input_dim,
                        self.scalar_dim,
                        self.scalar_dim,
                    ]
                    if self.tp_type.startswith("v3")
                    else [self.scalar_dim, self.scalar_dim, self.scalar_dim],
                    use_activation=True,
                    norm_layer=None,
                    internal_weights=False,
                )  # v1.0

                self.gbf = GaussianLayer(
                    self.attn_weight_input_dim
                )  # default output_dim = 128
                if self.tp_type.startswith("v2"):
                    self.pos_embedding_proj = nn.Linear(
                        self.attn_weight_input_dim, self.scalar_dim
                    )
                    self.node_scalar_proj = nn.Linear(self.scalar_dim, self.scalar_dim)
        self.proj2 = IrrepsLinear(
            # self.irreps_node_input,
            (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
            self.irreps_node_input,
            bias=True,
            act="identity",
        )  # B*L*irrep   for value
        self.proj = IrrepsLinear(
            # self.irreps_node_input,
            (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
            self.irreps_node_input,
            bias=True,
            act="identity",
        )  # B*L*irrep   for value

        self.proj_out = IrrepsLinear(
            self.irreps_node_input,
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout()
        self.add_rope = add_rope
        if add_rope:
            self.rot_emb = RotaryEmbedding(dim=self.attn_scalar_head)

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """
        irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
        B,L = 4,20
        node_pos = torch.randn(B,L,3)
        node_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))

        attn_scalar_head = 32
        func = GraphAttention(
            irreps_node_input = irreps_in,
            attn_weight_input_dim = 32, # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads = 8,
            attn_scalar_head = attn_scalar_head,
            irreps_head = "32x0e+8x1e+4x2e",
            rescale_degree=False,
            nonlinear_message=False,
            alpha_drop=0.1,
            proj_drop=0.1,
        )
        out = func(node_pos,
            irreps_in.randn(B,L,-1),
            node_dis,
            torch.randn(B,L,L,attn_scalar_head))
        print(out.shape)
        """

        # attn_weight f(|r_ij|): B*L*L*heads
        # edge_dis: B*L*L
        # node_input: B*L*irreps (irreps e.g. 128x0e+64x1e+32x2e)
        # ir2scalar: B*L*irreps -> N*L*hidden (hidden e.g. head*32)
        # attn_weight: B*L*L*rbf_dim
        B, L, _ = node_irreps_input.shape
        # # origin self tp
        # if self.neighbor_weight is None:
        #     origin_scalars = self.origin_rbf(node_pos.norm(dim=-1).reshape(-1)).reshape(B, L, -1)
        #     origin_sh = o3.spherical_harmonics(
        #         l=self.irreps_origin,
        #         x=node_pos,  # [:, [1, 2, 0]],
        #         normalize=True,
        #         normalization="component",
        #     )
        #     node_irreps_input = self.origin_tp(node_irreps_input, origin_sh, origin_scalars)
        # else:
        #     # neighbor convolution
        #     if  self.neighbor_weight=="identity":
        #         alpha = torch.ones_like(edge_dis)
        #         alpha = alpha.masked_fill(attn_mask.squeeze(-1), 0.0)
        #         alpha = alpha / torch.sum(alpha, 2, keepdim=True) # B*L*L*head
        #     elif self.neighbor_weight == "distance":
        #         alpha = edge_dis.masked_fill(attn_mask.squeeze(-1), -1e6)
        #         alpha = torch.softmax(alpha, 2) # B*L*L*head
        #         alpha = self.alpha_dropout(alpha)
        #     elif self.neighbor_weight == "attention0":
        #         dist_weight = self.attn_weight2weight(attn_weight)
        #         alpha = dist_weight[..., 0]
        #         alpha = alpha.masked_fill(attn_mask.squeeze(-1), -1e6)
        #         alpha = torch.softmax(alpha, 2)
        #         alpha = self.alpha_dropout(alpha)
        #     elif self.neighbor_weight == "attention1":
        #         dist_weight = self.attn_weight2weight(attn_weight)
        #         alpha = dist_weight[..., 0]
        #         alpha = alpha.masked_fill(attn_mask.squeeze(-1), -1e6)
        #         alpha = torch.softmax(alpha, 2)
        #         alpha = self.alpha_dropout(alpha)
        #         alpha = alpha * attn_weight[..., 1]
        #     else:
        #         raise ValueError(f"neighbor_weight should be identity or distance but got {self.neighbor_weight}")
        #     message = self.message_proj1(node_irreps_input) # B*L*irreps
        #     message = torch.einsum("bmn,bnd->bmd", alpha, message)
        #     node_irreps_input = self.input_proj(node_irreps_input) + self.message_proj2(message)

        o3.spherical_harmonics(
            l=self.irreps_origin,
            x=node_pos,  # [:, [1, 2, 0]],
            normalize=True,
            normalization="component",
        )

        query = self.q_ir2scalar(node_irreps_input)
        key = self.k_ir2scalar(node_irreps_input)
        # value = self.v_irlinear(node_irreps_input)  # B*L*irreps
        # embed = self.node_embedding(atomic_numbers)
        # if self.attn_type.startswith('v3'):
        # attn_weight = self.gbf_attn(edge_dis, batched_data["node_type_edge"].long()) # B*L*L*-1
        attn_weight = self.attn_weight2heads(attn_weight)
        # torch.cat(
        #     [
        #         attn_weight,
        #         embed.reshape(B, L, 1, -1)
        #         .repeat(1, 1, L, 1),
        #         embed.reshape(B, 1, L, -1)
        #         .repeat(1, L, 1, 1),
        #     ],
        #     dim=-1,
        # ))

        if self.tp_type is None or self.tp_type == "None":
            node_scalars = None
        elif self.tp_type == "v2":  # sum pool edge_feature => node_scalar
            edge_feature = self.gbf(
                edge_dis, batched_data["node_type_edge"].long()
            )  # B*L*L*-1
            edge_feature = edge_feature.sum(dim=-2)  # B*L*-1
            node_scalars = self.pos_embedding_proj(
                edge_feature
            ) + self.node_scalar_proj(node_irreps_input[..., : self.scalar_dim])
            # TODO: nonlinear
            if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
                outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
                outcell_index_0 = batched_data["outcell_index_0"]
                node_scalars = torch.cat(
                    [node_scalars, node_scalars[outcell_index_0, outcell_index]], dim=1
                )
        elif self.tp_type == "v2.1":  # mean pool edge_feature => node_scalar
            edge_feature = self.gbf(
                edge_dis, batched_data["node_type_edge"].long()
            )  # B*L*L*-1
            edge_feature = edge_feature.mean(dim=-2)  # B*L*-1
            node_scalars = self.pos_embedding_proj(
                edge_feature
            ) + self.node_scalar_proj(node_irreps_input[..., : self.scalar_dim])
            if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
                outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
                outcell_index_0 = batched_data["outcell_index_0"]
                node_scalars = torch.cat(
                    [node_scalars, node_scalars[outcell_index_0, outcell_index]], dim=1
                )
        elif self.tp_type == "v3":
            edge_feature = self.gbf(
                edge_dis, batched_data["node_type_edge"].long()
            )  # B*L*L*-1
            node_scalars = edge_feature.mean(dim=-2)  # B*L*-1
        else:
            raise ValueError(f"tp_type should be v0 or v1 or v2 but got {self.tp_type}")

        # value = self.tp_linear(value, torch.ones_like(value.narrow(-1, 0, 1)), node_scalars) # v0 linear with external weights
        value = self.tp_linear(
            node_irreps_input,
            torch.ones_like(node_irreps_input.narrow(-1, 0, 1)),
            node_scalars,
        )  # v0.1, 0.2, 1.0

        if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
            outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
            outcell_index_0 = batched_data["outcell_index_0"]
            key = torch.cat([key, key[outcell_index_0, outcell_index]], dim=1)
            value = torch.cat([value, value[outcell_index_0, outcell_index]], dim=1)
            node_pos = torch.cat([node_pos, batched_data["expand_pos"].float()], dim=1)

        elif self.add_rope and batched_data["mol_type"] == 2:
            query = query.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
            key = key.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)
            query, key = self.rot_emb(
                query.transpose(1, 2).reshape(
                    B * self.num_attention_heads, L, self.attn_scalar_head
                ),
                key.transpose(1, 2).reshape(
                    B * self.num_attention_heads, L, self.attn_scalar_head
                ),
            )
            query = query.reshape(
                B, self.num_attention_heads, L, self.attn_scalar_head
            ).transpose(1, 2)
            key = key.reshape(
                B, self.num_attention_heads, L, self.attn_scalar_head
            ).transpose(1, 2)

        query = query.reshape(B, L, self.num_attention_heads, self.attn_scalar_head)
        key = key.reshape(B, -1, self.num_attention_heads, self.attn_scalar_head)

        head_dim = key.size(-1)  # or query.size(-1), since they should be the same
        scaling_factor = math.sqrt(head_dim)  # sqrt(d_k)
        # scaling_factor = 1.0
        # attn_weight: RBF(|r_ij|)
        alpha = (
            attn_weight[..., : self.num_attention_heads]
            * torch.einsum("bmhs,bnhs->bmnh", query, key)
            / scaling_factor
        )
        alpha = self.alpha_act(alpha)

        # if attn_mask is not None:
        #     # alpha = alpha.masked_fill(attn_mask, float('-inf'))
        alpha = alpha.masked_fill(attn_mask, -1e6)
        alpha = torch.nn.functional.softmax(alpha, dim=2)
        alpha = self.alpha_dropout(alpha)
        alpha = alpha / (edge_dis.unsqueeze(dim=-1) + 1e-8)  # B*L*L*head
        alpha = (
            alpha * attn_weight[..., self.num_attention_heads :]
        )  # add attn weight in messages
        # alpha = alpha.masked_fill(attn_mask, 0)

        value_21 = self.sep_tp2(
            value, node_pos, None
        )  # B*N*irreps, B*N*3 uvw - > B*N*irreps
        value_21 = self.vec2heads(value_21)  # B*L*heads*irreps_head
        value_21 = -torch.einsum("bmlh,blhi->bmhi", alpha, value_21)
        value_21 = self.heads2vec(value_21)

        value = self.vec2heads(value)  # B*L*heads*irreps_head
        value = torch.einsum("bmlh,blhi->bmhi", alpha, value)
        value = self.heads2vec(value)

        value_20 = self.sep_tp2(
            value[:, :L], node_pos[:, :L], None
        )  # TODO: different scalars?

        node_output = self.proj2(value_20 + value_21) + self.proj(value)  # v0, v0.1
        # node_output = value_20 + value_21  + value # v0.2
        # if self.rescale_degree:
        #     degree = torch_geometric.utils.degree(
        #         edge_dst, num_nodes=node_input.shape[0], dtype=node_input.dtype
        #     )
        #     degree = degree.view(-1, 1)
        #     node_output = node_output * degree
        node_output = self.proj_out(node_output)

        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)

        return node_output

        # B*L*L*head

    def extra_repr(self):
        output_str = super(E2AttentionLinear, self).extra_repr()
        output_str = output_str + "rescale_degree={}, ".format(self.rescale_degree)
        return output_str


@compile_mode("script")
class E2Attention_pbias(torch.nn.Module):
    """
    1. Message = Alpha * Value
    2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
    3. 0e -> Activation -> Inner Product -> (Alpha)
    4. (0e+1e+...) -> (Value)


    tp_type:
    v0: use irreps_node_input 0e
    v1: use sum of edge features
    v2: use sum of edge features and irreps_node_input 0e

    attn_type
    v0: initial implementation with attn_weight multiplication
    v1: GATv2 implementation without attn bias


    """

    def __init__(
        self,
        irreps_node_input="256x0e+64x1e+32x2e",
        attn_weight_input_dim: int = 32,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads: int = 8,
        attn_scalar_head: int = 32,
        irreps_head="32x0e+8x1e+4x2e",
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
        tp_type="v1",
        attn_type="v2",
        add_rope=True,
        **kwargs,
    ):
        super().__init__()

        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.scalar_dim = self.irreps_node_input[0][0]  # scalar_dim x 0e
        self.num_attention_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.attn_weight_input_dim = attn_weight_input_dim
        self.irreps_head = (
            o3.Irreps(irreps_head) if isinstance(irreps_head, str) else irreps_head
        )
        # irreps_node_output,  attention will not change the input shape/embeding length
        self.irreps_node_output = self.irreps_node_input
        # new params
        self.tp_type = tp_type
        self.attn_type = attn_type

        self.gbf_attn = GaussianLayer(
            self.attn_weight_input_dim
        )  # default output_dim = 128
        self.node_embedding = nn.Embedding(256, 32)
        nn.init.uniform_(self.node_embedding.weight.data, -0.001, 0.001)

        # self.attn_weight2heads = nn.Sequential(
        #     nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
        #     nn.LayerNorm(attn_weight_input_dim),
        #     torch.nn.SiLU(),
        #     # nn.Linear(attn_weight_input_dim, attn_weight_input_dim),
        #     # nn.LayerNorm(attn_weight_input_dim),
        #     # torch.nn.SiLU(),
        #     nn.Linear(attn_weight_input_dim, 2 * num_attn_heads),
        # )
        # self.attn_weight2heads = nn.Sequential(
        #     nn.Linear(attn_weight_input_dim, 2*num_attn_heads),
        #     # torch.nn.SiLU(),
        #     # nn.Linear(attn_weight_input_dim, 2 * num_attn_heads),
        # )

        self.multiheadatten_irreps = (
            (self.irreps_head * num_attn_heads).sort().irreps.simplify()
        )

        self.q_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )
        self.k_ir2scalar = Irreps2Scalar(
            self.irreps_node_input,
            num_attn_heads * attn_scalar_head,
            bias=True,
            act="smoothleakyrelu",
        )

        self.v_irlinear = IrrepsLinear(
            self.irreps_node_input,
            self.multiheadatten_irreps,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irreps

        self.vec2heads = Vec2AttnHeads(
            self.irreps_head,
            num_attn_heads,
        )
        self.heads2vec = AttnHeads2Vec(irreps_head=self.irreps_head)

        # alpha_dot, used for scalar to num_heads dimension
        # self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        # self.alpha_act = SmoothLeakyReLU(0.2)  # used for attention_ij
        # self.alpha_dropout = None
        # if alpha_drop != 0.0:
        #     self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        self.sep_tp = None
        if self.nonlinear_message or self.rescale_degree:
            raise ValueError(
                "sorry, rescale_degree or non linear message passing is not supported in tp transformer"
            )

        else:
            if self.tp_type is None or self.tp_type == "None":
                self.sep_tp = SeparableFCTP(
                    self.multiheadatten_irreps,
                    "1x1e",
                    self.multiheadatten_irreps,
                    fc_neurons=None,
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=True,
                )
            else:
                self.sep_tp = SeparableFCTP(
                    self.multiheadatten_irreps,
                    "1x1e",
                    self.multiheadatten_irreps,
                    fc_neurons=[self.scalar_dim, self.scalar_dim, self.scalar_dim],
                    use_activation=False,
                    norm_layer=None,
                    internal_weights=False,
                )

                if self.tp_type == "v2":
                    self.gbf = GaussianLayer(
                        self.attn_weight_input_dim
                    )  # default output_dim = 128
                    self.pos_embedding_proj = nn.Linear(
                        self.attn_weight_input_dim, self.scalar_dim
                    )
                    self.node_scalar_proj = nn.Linear(self.scalar_dim, self.scalar_dim)
                else:
                    raise ValueError(
                        f"attn_type should be None or v0 or v1 or v2 but got {self.attn_type}"
                    )

        self.proj = IrrepsLinear(
            (self.irreps_head * num_attn_heads).sort().irreps.simplify(),
            self.irreps_node_input,
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irrep
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout()
        self.add_rope = add_rope
        if add_rope:
            self.rot_emb = RotaryEmbedding(dim=self.attn_scalar_head)

    def forward(
        self,
        node_pos,
        node_irreps_input,
        edge_dis,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        attn_mask=None,  # non-adj is True
        batch=None,
        batched_data=None,
        **kwargs,
    ):
        """
        irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
        B,L = 4,20
        node_pos = torch.randn(B,L,3)
        node_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))

        attn_scalar_head = 32
        func = GraphAttention(
            irreps_node_input = irreps_in,
            attn_weight_input_dim = 32, # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads = 8,
            attn_scalar_head = attn_scalar_head,
            irreps_head = "32x0e+8x1e+4x2e",
            rescale_degree=False,
            nonlinear_message=False,
            alpha_drop=0.1,
            proj_drop=0.1,
        )
        out = func(node_pos,
            irreps_in.randn(B,L,-1),
            node_dis,
            torch.randn(B,L,L,attn_scalar_head))
        print(out.shape)
        """

        # attn_weight f(|r_ij|): B*L*L*heads
        # edge_dis: B*L*L
        # node_input: B*L*irreps (irreps e.g. 128x0e+64x1e+32x2e)
        # ir2scalar: B*L*irreps -> N*L*hidden (hidden e.g. head*32)
        # attn_weight: B*L*L*rbf_dim
        B, L, _ = node_irreps_input.shape
        query = self.q_ir2scalar(node_irreps_input)
        key = self.k_ir2scalar(node_irreps_input)
        value = self.v_irlinear(node_irreps_input)  # B*L*irreps

        if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
            outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
            outcell_index_0 = batched_data["outcell_index_0"]
            key = torch.cat([key, key[outcell_index_0, outcell_index]], dim=1)
            value = torch.cat([value, value[outcell_index_0, outcell_index]], dim=1)
            node_pos = torch.cat([node_pos, batched_data["expand_pos"].float()], dim=1)

            query = query.reshape(
                B, -1, self.num_attention_heads, self.attn_scalar_head
            ).transpose(1, 2)
            key = key.reshape(
                B, -1, self.num_attention_heads, self.attn_scalar_head
            ).transpose(1, 2)

        elif self.add_rope and batched_data["mol_type"] == 2:
            query = query.reshape(
                B, -1, self.num_attention_heads, self.attn_scalar_head
            ).transpose(1, 2)
            key = key.reshape(
                B, -1, self.num_attention_heads, self.attn_scalar_head
            ).transpose(1, 2)
            query, key = self.rot_emb(
                query.reshape(B * self.num_attention_heads, L, self.attn_scalar_head),
                key.reshape(B * self.num_attention_heads, L, self.attn_scalar_head),
            )
            query = query.reshape(B, self.num_attention_heads, L, self.attn_scalar_head)
            key = key.reshape(B, self.num_attention_heads, L, self.attn_scalar_head)

        else:
            query = query.reshape(
                B, -1, self.num_attention_heads, self.attn_scalar_head
            ).transpose(1, 2)
            key = key.reshape(
                B, -1, self.num_attention_heads, self.attn_scalar_head
            ).transpose(1, 2)

        attn_mask = attn_mask.squeeze(-1).unsqueeze(
            1
        )  # [batch_size, 1, seq_len, seq_len]
        attn_mask = attn_mask.expand(
            -1, query.size(1), -1, -1
        )  # [batch_size, num_heads, seq_len, seq_len]
        attn_mask = attn_weight.permute(0, 3, 1, 2).masked_fill(attn_mask, -1e6)

        if self.tp_type is None or self.tp_type == "None":
            node_scalars = None
        elif self.tp_type == "v2":
            edge_feature = self.gbf(
                edge_dis, batched_data["node_type_edge"].long()
            )  # B*L*L*-1
            edge_feature = edge_feature.sum(dim=-2)  # B*L*-1
            node_scalars = self.pos_embedding_proj(
                edge_feature
            ) + self.node_scalar_proj(node_irreps_input[..., : self.scalar_dim])
            if batched_data["mol_type"] == 1:  # mol_type 0 mol, 1 material, 2 protein
                outcell_index = batched_data["outcell_index"]  # B*(L2-L1)
                outcell_index_0 = batched_data["outcell_index_0"]
                node_scalars = torch.cat(
                    [node_scalars, node_scalars[outcell_index_0, outcell_index]], dim=1
                )
        else:
            raise ValueError(f"tp_type should be v0 or v1 or v2 but got {self.tp_type}")

        node_pos_abs = torch.sqrt(torch.sum(node_pos**2, dim=-1, keepdim=True) + 1e-8)
        node_pos = node_pos / node_pos_abs
        value_0 = self.sep_tp(
            value, node_pos, node_scalars
        )  # B*N*irreps, B*N*3 uvw - > B*N*irreps
        value_0 = self.vec2heads(value_0).transpose(1, 2)  # B*L*heads*irreps_head
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            if attn_mask is not None:
                attn_mask = attn_mask.to(dtype=query.dtype)
            value_0 = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value_0,
                dropout_p=0.1,
                attn_mask=attn_mask,
                is_causal=False,
            )
        value_0 = self.heads2vec(value_0.transpose(1, 2))

        value = self.vec2heads(value).transpose(1, 2)  # B*L*heads*irreps_head
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            if attn_mask is not None:
                attn_mask = attn_mask.to(dtype=query.dtype)
            value = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.1,
                attn_mask=attn_mask,
                is_causal=False,
            )
        value = self.heads2vec(value.transpose(1, 2))
        value_1 = self.sep_tp(
            value[:, :L],
            node_pos[:, :L],
            None if node_scalars is None else node_scalars[:, :L],
        )  # TODO: different scalars?

        node_output = value_0 + value_1 + value

        node_output = self.proj(node_output)

        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)

        return node_output

        # B*L*L*head

    def extra_repr(self):
        output_str = "rescale_degree={}, ".format(self.rescale_degree)
        return output_str


def get_mul_0(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


def DepthwiseTensorProduct(
    irreps_node_input,
    irreps_edge_attr,
    irreps_node_output,
    internal_weights=False,
    bias=True,
    mode="default",
):
    """
    The irreps of output is pre-determined.
    `irreps_node_output` is used to get certain types of vectors.
    """
    irreps_output = []
    instructions = []

    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))

    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output)  # irreps_output.sort()
    instructions = [
        (i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions
    ]
    if mode != "default":
        if bias is True or internal_weights is False:
            raise ValueError("tp not support some parameter, please check your code.")
    tp = TensorProductRescale(
        irreps_node_input,
        irreps_edge_attr,
        irreps_output,
        instructions,
        internal_weights=internal_weights,
        shared_weights=internal_weights,
        bias=bias,
        rescale=_RESCALE,
        mode=mode,
    )
    return tp


@compile_mode("trace")
class Activation(torch.nn.Module):
    """
    Directly apply activation when irreps is type-0.
    """

    def __init__(self, irreps_in, acts):
        super().__init__()
        if isinstance(irreps_in, str):
            irreps_in = o3.Irreps(irreps_in)
        assert len(irreps_in) == len(acts), (irreps_in, acts)

        # normalize the second moment
        acts = [
            e3nn.math.normalize2mom(act) if act is not None else None for act in acts
        ]

        from e3nn.util._argtools import _get_device

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError(
                        "Activation: cannot apply an activation function to a non-scalar input."
                    )

                x = torch.linspace(0, 10, 256, device=_get_device(act))

                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs().max() < 1e-5:
                    p_act = 1
                elif (a1 + a2).abs().max() < 1e-5:
                    p_act = -1
                else:
                    p_act = 0

                p_out = p_act if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError(
                        "Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd."
                    )
            else:
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(irreps_out)
        self.acts = torch.nn.ModuleList(acts)
        assert len(self.irreps_in) == len(self.acts)

    # def __repr__(self):
    #    acts = "".join(["x" if a is not None else " " for a in self.acts])
    #    return f"{self.__class__.__name__} [{self.acts}] ({self.irreps_in} -> {self.irreps_out})"
    def extra_repr(self):
        output_str = super(Activation, self).extra_repr()
        output_str = output_str + "{} -> {}, ".format(self.irreps_in, self.irreps_out)
        return output_str

    def forward(self, features, dim=-1):
        # directly apply activation without narrow
        if len(self.acts) == 1:
            return self.acts[0](features)

        output = []
        index = 0
        for (mul, ir), act in zip(self.irreps_in, self.acts):
            if act is not None:
                output.append(act(features.narrow(dim, index, mul)))
            else:
                output.append(features.narrow(dim, index, mul * ir.dim))
            index += mul * ir.dim

        if len(output) > 1:
            return torch.cat(output, dim=dim)
        elif len(output) == 1:
            return output[0]
        else:
            return torch.zeros_like(features)


@compile_mode("script")
class Gate(torch.nn.Module):
    """
    TODO: to be optimized.  Toooooo ugly
    1. Use `narrow` to split tensor.
    2. Use `Activation` in this file.
    """

    def __init__(
        self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated
    ):
        super().__init__()
        irreps_scalars = o3.Irreps(irreps_scalars)
        irreps_gates = o3.Irreps(irreps_gates)
        irreps_gated = o3.Irreps(irreps_gated)

        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError(
                f"Gate scalars must be scalars, instead got irreps_gates = {irreps_gates}"
            )
        if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
            raise ValueError(
                f"Scalars must be scalars, instead got irreps_scalars = {irreps_scalars}"
            )
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(
                f"There are {irreps_gated.num_irreps} irreps in irreps_gated, but a different number ({irreps_gates.num_irreps}) of gate scalars in irreps_gates"
            )
        # assert len(irreps_scalars) == 1
        # assert len(irreps_gates) == 1

        self.irreps_scalars = irreps_scalars
        self.irreps_gates = irreps_gates
        self.irreps_gated = irreps_gated
        self._irreps_in = (irreps_scalars + irreps_gates + irreps_gated).simplify()

        self.act_scalars = Activation(irreps_scalars, act_scalars)
        irreps_scalars = self.act_scalars.irreps_out

        self.act_gates = Activation(irreps_gates, act_gates)
        irreps_gates = self.act_gates.irreps_out

        self.mul = o3.ElementwiseTensorProduct(irreps_gated, irreps_gates)
        irreps_gated = self.mul.irreps_out

        self._irreps_out = irreps_scalars + irreps_gated

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features):
        scalars_dim = self.irreps_scalars.dim
        gates_dim = self.irreps_gates.dim
        input_dim = self.irreps_in.dim

        scalars = features.narrow(-1, 0, scalars_dim)
        gates = features.narrow(-1, scalars_dim, gates_dim)
        gated = features.narrow(
            -1, (scalars_dim + gates_dim), (input_dim - scalars_dim - gates_dim)
        )

        scalars = self.act_scalars(scalars)
        if gates.shape[-1]:
            gates = self.act_gates(gates)
            gated = self.mul(gated, gates)
            features = torch.cat([scalars, gated], dim=-1)
        else:
            features = scalars
        return features

    @property
    def irreps_in(self):
        """Input representations."""
        return self._irreps_in

    @property
    def irreps_out(self):
        """Output representations."""
        return self._irreps_out


@compile_mode("script")
class FeedForwardNetwork(torch.nn.Module):
    """
    Use two (FCTP + Gate)
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_output,
        proj_drop=0.1,
    ):
        super().__init__()
        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )

        self.irreps_node_output = (
            o3.Irreps(irreps_node_output)
            if isinstance(irreps_node_output, str)
            else irreps_node_output
        )

        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_input)
        self.irreps_mlp_mid = (
            (self.irreps_node_input + irreps_gates).sort().irreps.simplify()
        )

        # warnings.warn(f"FeedForwardNetwork:GATE is tooooooo ugly, please refine this later")

        self.slinear_1 = IrrepsLinear(
            self.irreps_node_input, self.irreps_mlp_mid, bias=True, act=None
        )
        # TODO: to be optimized.  Toooooo ugly
        if irreps_gated.num_irreps == 0:
            self.gate = Activation(self.irreps_mlp_mid, acts=[torch.nn.functional.silu])
        else:
            self.gate = Gate(
                irreps_scalars,
                [torch.nn.functional.silu for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )

        self.slinear_2 = IrrepsLinear(
            self.irreps_node_input, self.irreps_node_output, bias=True, act=None
        )
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(
                self.irreps_node_output, drop_prob=proj_drop
            )

    def forward(self, node_input, **kwargs):
        """
        irreps_in = o3.Irreps("128x0e+32x1e")
        func =  FeedForwardNetwork(
                irreps_in,
                irreps_in,
                proj_drop=0.1,
            )
        out = func(irreps_in.randn(10,20,-1))
        """
        node_output = self.slinear_1(node_input)
        node_output = self.gate(node_output)
        node_output = self.slinear_2(node_output)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output


@compile_mode("script")
class Gate_OP(torch.nn.Module):
    """
    TODO: to be optimized.  Toooooo ugly
    1. Use `narrow` to split tensor.
    2. Use `Activation` in this file.
    """

    def __init__(self, node_irreps, act_scalars="silu", act_vector="sigmoid"):
        super().__init__()
        node_irreps = (
            o3.Irreps(node_irreps) if isinstance(node_irreps, str) else node_irreps
        )
        self.node_irreps = node_irreps
        if node_irreps[0][1][0] != 0:
            raise ValueError("The irreps in for Gate must have 0e")
        elif len(node_irreps) <= 1:
            raise ValueError("Gate must have 1/2/3/4/5e at least.")
        self.scalars_channel = node_irreps[0][0]

        self.gates = torch.nn.Linear(node_irreps[0][0], node_irreps.num_irreps)
        bound = 1 / math.sqrt(node_irreps[0][0])
        torch.nn.init.uniform_(self.gates.weight, -bound, bound)

        if act_scalars == "silu":
            self.act_scalars = e3nn.math.normalize2mom(torch.nn.SiLU())
        else:
            raise ValueError("in Gate, only support silu")

        if act_vector == "sigmoid":
            self.act_vector = e3nn.math.normalize2mom(torch.nn.Sigmoid())
        else:
            raise ValueError("in Gate, only support sigmoid for vector")

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.node_irreps} -> {self.node_irreps})"

    def forward(self, features):
        input_shape = features.shape
        input_dim = features.shape[-1]
        features = features.reshape(-1, input_dim)
        bs = features.shape[0]
        scalars = features[..., : self.scalars_channel]
        features[..., self.scalars_channel :]

        scalars = self.gates(scalars)
        out = [self.act_scalars(scalars[..., : self.scalars_channel])]

        start_dim = self.scalars_channel
        start = self.scalars_channel
        for i in range(1, len(self.node_irreps)):
            l = self.node_irreps[i][1].l
            mul = self.node_irreps[i][0]
            out.append(
                self.act_vector(
                    scalars[:, start_dim : start_dim + mul].unsqueeze(dim=-1)
                    * (
                        features[:, start_dim : start_dim + mul * (2 * l + 1)].reshape(
                            -1, mul, 2 * l + 1
                        )
                    )
                ).reshape(bs, -1)
            )
            start += mul
            start_dim += self.node_irreps[i][0]
        out = torch.cat(out, dim=-1)
        return out.reshape(input_shape)

    @property
    def irreps_in(self):
        """Input representations."""
        return self.out

    @property
    def irreps_out(self):
        """Output representations."""
        return self.out


@compile_mode("script")
class FeedForwardNetwork_GateOP(torch.nn.Module):
    """
    Use two (FCTP + Gate)
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_output,
        proj_drop=0.1,
    ):
        super().__init__()
        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )

        self.irreps_node_output = (
            o3.Irreps(irreps_node_output)
            if isinstance(irreps_node_output, str)
            else irreps_node_output
        )

        self.slinear_1 = IrrepsLinear(
            self.irreps_node_input, self.irreps_node_input, bias=True, act=None
        )

        self.gate = Gate_OP(
            self.irreps_node_input, act_scalars="silu", act_vector="sigmoid"
        )

        self.slinear_2 = IrrepsLinear(
            self.irreps_node_input, self.irreps_node_output, bias=True, act=None
        )
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(
                self.irreps_node_output, drop_prob=proj_drop
            )

    def forward(self, node_input, **kwargs):
        """
        irreps_in = o3.Irreps("128x0e+32x1e")
        func =  FeedForwardNetwork(
                irreps_in,
                irreps_in,
                proj_drop=0.1,
            )
        out = func(irreps_in.randn(10,20,-1))
        """
        node_output = self.slinear_1(node_input)
        node_output = self.gate(node_output)
        node_output = self.slinear_2(node_output)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output


@compile_mode("script")
class TransBlock(torch.nn.Module):
    """
    1. Layer Norm 1 -> E2Attention -> Layer Norm 2 -> FeedForwardNetwork
    2. Use pre-norm architecture
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_output,
        attn_weight_input_dim,  # e.g. rbf(|r_ij|) or relative pos in sequence
        num_attn_heads,
        attn_scalar_head,
        irreps_head,
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
        drop_path_rate=0.0,
        norm_layer="layer",  # used for norm 1 and norm2
        layer_id=0,
        attn_type=0,
        tp_type="v2",
        ffn_type="default",
        add_rope=True,
        max_radius=15,
    ):
        super().__init__()
        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.irreps_node_output = (
            o3.Irreps(irreps_node_output)
            if isinstance(irreps_node_output, str)
            else irreps_node_output
        )

        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        self.norm_1 = get_norm_layer(norm_layer)(self.irreps_node_input)
        func = None
        if attn_type in [None, "None", "nb", "w2head3"]:
            func = E2Attention
        elif attn_type in [17]:
            func = E2Attention_pbias
        elif isinstance(attn_type, str) and attn_type.startswith("so2"):
            func = E2AttentionSO2
        elif isinstance(attn_type, str) and attn_type.startswith("headatt"):
            func = E2Attention_headatt
        elif isinstance(attn_type, str) and attn_type.startswith("headatt2"):
            func = E2Attention_headatt2
        elif isinstance(attn_type, str) and attn_type.startswith("origin"):
            func = E2AttentionOrigin
        elif isinstance(attn_type, str) and attn_type.startswith("linear"):
            func = E2AttentionLinear
        elif isinstance(attn_type, str) and attn_type.startswith("triple"):
            func = E2AttentionTriple
        else:
            raise ValueError(
                f" sorry, the attn type is not support, please check {attn_type}"
            )

        self.ga = func(
            irreps_node_input,
            attn_weight_input_dim,  # e.g. rbf(|r_ij|) or relative pos in sequence
            num_attn_heads,
            attn_scalar_head,
            irreps_head,
            rescale_degree=rescale_degree,
            nonlinear_message=nonlinear_message,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
            layer_id=layer_id,
            attn_type=attn_type,
            tp_type=tp_type,
            add_rope=add_rope,
            max_radius=max_radius,
        )
        if drop_path_rate > 0.0:
            self.drop_path = DropPath(drop_path_rate)
        else:
            self.drop_path = nn.Identity()

        self.norm_2 = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.so2_ffn = None
        if ffn_type == "gate_op":
            self.ffn = FeedForwardNetwork_GateOP(
                irreps_node_input=self.irreps_node_input,  # self.concat_norm_output.irreps_out,
                irreps_node_output=self.irreps_node_input,
                proj_drop=proj_drop,
            )
        else:
            self.ffn = FeedForwardNetwork(
                irreps_node_input=self.irreps_node_input,  # self.concat_norm_output.irreps_out,
                irreps_node_output=self.irreps_node_input,
                proj_drop=proj_drop,
            )

        if ffn_type == "so2":
            self.norm_3 = get_norm_layer(norm_layer)(self.irreps_node_input)
            self.rot_func = SO3_Rotation(
                lmax=max([i[1].l for i in self.irreps_node_input]),
                irreps=self.irreps_node_input,
            )
            self.so2_ffn = SO2_Convolution(irreps_in=self.irreps_node_input)
        elif ffn_type == "newso2":
            self.norm_3 = get_norm_layer(norm_layer)(self.irreps_node_input)
            self.rot_func = SO3_Rotation(
                lmax=max([i[1].l for i in self.irreps_node_input]),
                irreps=self.irreps_node_input,
            )
            self.so2_ffn = SO2_Convolution_sameorder(irreps_in=self.irreps_node_input)

        self.manybody_ffn = None
        if ffn_type == "2body":
            self.norm_manybody = get_norm_layer(norm_layer)(self.irreps_node_input)
            self.manybody_ffn = Body2_interaction(
                self.irreps_node_input, internal_weights=True
            )
        elif ffn_type == "3body":
            self.norm_manybody = get_norm_layer(norm_layer)(self.irreps_node_input)
            self.manybody_ffn = Body3_interaction_MACE(
                self.irreps_node_input, internal_weights=True
            )

        # self.norm_3 = get_norm_layer(norm_layer)(self.irreps_node_input)
        # self.ffn_vec2scalar = FeedForwardVec2Scalar(
        #     irreps_node_input=self.irreps_node_input,  # self.concat_norm_output.irreps_out,
        #     irreps_node_output=self.irreps_node_output,
        # )

        self.add_rope = add_rope

    def forward(
        self,
        node_pos,
        node_irreps,
        edge_dis,
        attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
        atomic_numbers,
        attn_mask,
        batch,
        batched_data=None,
        **kwargs,
    ):
        """
        only norm=layer is supported here, that is EquivariantLayerNormV2, --batch-- variable is not needed
        example:


        irreps_in = o3.Irreps("256x0e+64x1e+32x2e")
        B,L = 4,100
        dis_embedding_dim = 32
        node_pos = torch.randn(B,L,3)
        edge_dis = torch.sqrt(torch.sum((node_pos.view(B,L,1,3)-node_pos.view(B,1,L,3))**2,dim = -1))
        dis_embedding = torch.randn(B,L,L,dis_embedding_dim)
        func = TransBlock(
                irreps_in,
                irreps_in,
                attn_weight_input_dim=dis_embedding_dim, # e.g. rbf(|r_ij|) or relative pos in sequence
                num_attn_heads=8,
                attn_scalar_head = 48,
                irreps_head="32x0e+16x1e+8x2e",
                rescale_degree=False,
                nonlinear_message=False,
                alpha_drop=0.1,
                proj_drop=0.1,
                drop_path_rate=0.0,
                norm_layer="layer", # used for norm 1 and norm2
            )

        out = func.forward(
                node_pos,
                irreps_in.randn(B,L,-1),
                edge_dis,
                dis_embedding, # e.g. rbf(|r_ij|) or relative pos in sequence
                batch=None)
        """

        ## residual connection
        node_irreps_res = node_irreps
        node_irreps = self.norm_1(node_irreps, batch=batch)
        node_irreps = self.ga(
            node_pos=node_pos,
            node_irreps_input=node_irreps,
            edge_dis=edge_dis,
            attn_weight=attn_weight,
            atomic_numbers=atomic_numbers,
            attn_mask=attn_mask,
            batched_data=batched_data,
            add_rope=self.add_rope,
        )

        node_irreps = self.drop_path(node_irreps)

        node_irreps = node_irreps + node_irreps_res

        ## residual connection
        node_irreps_res = node_irreps
        node_irreps = self.norm_2(node_irreps, batch=batch)
        node_irreps = self.ffn(node_irreps)
        node_irreps = self.drop_path(node_irreps)
        node_irreps = node_irreps_res + node_irreps

        if self.so2_ffn is not None:
            node_irreps_res = node_irreps
            self.rot_func.set_wigner(
                self.rot_func.init_edge_rot_mat(node_pos.reshape(-1, 3))
            )

            node_irreps = self.norm_3(node_irreps, batch=batch)
            node_irreps = self.rot_func.rotate(node_irreps)
            node_irreps = self.so2_ffn(node_irreps)
            node_irreps = self.rot_func.rotate_inv(node_irreps)

            node_irreps = node_irreps_res + node_irreps

        if self.manybody_ffn is not None:
            node_irreps_res = node_irreps
            node_irreps = self.norm_manybody(node_irreps, batch=batch)
            node_irreps = self.manybody_ffn(node_irreps, atomic_numbers)
            node_irreps = node_irreps_res + node_irreps

        # node_irreps_res = node_irreps
        # node_irreps = self.norm_3(node_irreps, batch=batch)
        # node_irreps = self.ffn_vec2scalar(node_irreps)
        # node_irreps = node_irreps_res + node_irreps
        return node_irreps


class EdgeDegreeEmbeddingNetwork_Dense(torch.nn.Module):
    def __init__(self, irreps_node_embedding, avg_aggregate_num, cutoff=None, **kwargs):
        super().__init__()
        self.cutoff = cutoff
        self.irreps_node_embedding = (
            o3.Irreps(irreps_node_embedding)
            if isinstance(irreps_node_embedding, str)
            else irreps_node_embedding
        )
        if self.irreps_node_embedding[0][1].l != 0:
            raise ValueError("node embedding must have sph order 0 embedding.")
        self.exp = IrrepsLinear(
            o3.Irreps("1x0e"),
            f"{self.irreps_node_embedding[0][0]}x0e",
        )

        self._node_vec_dim = (
            self.irreps_node_embedding.dim - self.irreps_node_embedding[0][0]
        )

        self.dw = DepthwiseTensorProduct(
            self.irreps_node_embedding,
            o3.Irreps("1x1e"),
            self.irreps_node_embedding,
            internal_weights=True,
            bias=False,
        )
        self.linear_0 = IrrepsLinear(
            self.irreps_node_embedding,
            self.dw.tp.irreps_out.simplify(),
        )

        self.proj = IrrepsLinear(
            self.dw.irreps_out.simplify(), self.irreps_node_embedding
        )
        self.avg_aggregate_num = avg_aggregate_num

    def forward(
        self, node_input, node_pos, edge_dis, edge_scalars=None, batch=None, **kwargs
    ):
        """
        Parameters
        ----------
        node_pos : postiion
            tensor of shape ``(B, L, 3)``

        edge_scalars : rbf of distance
            tensor of shape ``(B, L, L, number_of_basis)``

        edge_dis : distances of all node pairs
            tensor of shape ``(B, L, L)``
        """

        # B,L,_ = node_input.shape
        # edge_index = torch.vstack([edge_src, edge_dst])
        # adj = to_dense_adj(edge_index, batch, max_num_nodes=L) # adj matrix in shape[B, L, L]

        adj = 1 / (edge_dis + 1e-8)  # 1/|r_ij|
        if self.cutoff is not None:
            adj = torch.where(adj < 1 / self.cutoff, 0.0, adj)  # mask out distant edges
        B, L = node_pos.shape[:2]
        node_features = torch.ones(
            (B, L, 1), device=node_pos.device, dtype=node_pos.dtype
        )
        node_features = self.exp(node_features)  # 1x0e => irreps
        node_features = torch.cat(
            [
                node_features,
                torch.zeros(
                    B,
                    L,
                    self._node_vec_dim,
                    dtype=node_features.dtype,
                    device=node_features.device,
                ),
            ],
            dim=2,
        )
        # if self.internal_weights:
        #     weight = None
        # else:
        #     weight = self.rad(edge_scalars)
        node_features_a = self.dw(adj @ node_features, node_pos, weight=None)
        node_features_new = self.dw(node_features, node_pos, weight=None)  # h_new
        node_features_0 = self.linear_0(node_features)  # for tp with order 0
        node_features = (
            node_features_a - adj @ node_features_new + adj @ node_features_0
        )  # TODO: sparse matrix mul
        node_features = self.proj(node_features)
        return node_features / self.avg_aggregate_num**0.5


class EdgeDegreeEmbeddingNetwork_higherorder(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding,
        avg_aggregate_num=10,
        number_of_basis=32,
        cutoff=None,
        time_embed=False,
        **kwargs,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.irreps_node_embedding = (
            o3.Irreps(irreps_node_embedding)
            if isinstance(irreps_node_embedding, str)
            else irreps_node_embedding
        )
        if self.irreps_node_embedding[0][1].l != 0:
            raise ValueError("node embedding must have sph order 0 embedding.")
        self.gbf = GaussianLayer(number_of_basis)  # default output_dim = 128
        self.gbf_projs = nn.ModuleList()

        self.scalar_dim = self.irreps_node_embedding[0][0]
        self.embed = nn.Embedding(256, self.scalar_dim)
        self.embed.weight.data.normal_(0, 1 / self.scalar_dim**0.5)
        if time_embed:
            self.time_embed_proj = nn.Sequential(
                nn.Linear(self.scalar_dim, self.scalar_dim, bias=True),
                nn.SiLU(),
                nn.Linear(self.scalar_dim, number_of_basis, bias=True),
            )

        self.weight_list = nn.ParameterList()
        for idx in range(len(self.irreps_node_embedding)):
            self.gbf_projs.append(
                RadialProfile([number_of_basis, number_of_basis], use_layer_norm=False)
            )

            out_feature = self.irreps_node_embedding[idx][0]
            weight = torch.nn.Parameter(torch.randn(out_feature, number_of_basis))
            bound = 1 / math.sqrt(number_of_basis)
            torch.nn.init.uniform_(weight, -bound, bound)
            self.weight_list.append(weight)

        self.proj = IrrepsLinear(self.irreps_node_embedding, self.irreps_node_embedding)
        self.avg_aggregate_num = avg_aggregate_num

    def forward(
        self,
        node_input,
        node_pos,
        edge_dis,
        atomic_numbers,
        edge_vec,
        batched_data,
        attn_mask,
        time_embed=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        node_pos : postiion
            tensor of shape ``(B, L, 3)``

        edge_scalars : rbf of distance
            tensor of shape ``(B, L, L, number_of_basis)``

        edge_dis : distances of all node pairs
            tensor of shape ``(B, L, L)``
        """

        B, L = node_pos.shape[:2]
        self.embed(atomic_numbers)  # B*L*hidden

        # edge_vec = node_pos.unsqueeze(2) - node_pos.unsqueeze(1)  # B, L, L, 3
        node_type_edge = batched_data["node_type_edge"]
        edge_dis_embed = self.gbf(edge_dis, node_type_edge.long())
        if time_embed is not None:
            edge_dis_embed += self.time_embed_proj(time_embed).unsqueeze(-2)
        edge_dis_embed = torch.where(attn_mask, 0, edge_dis_embed)
        node_features = []
        for idx in range(len(self.irreps_node_embedding)):
            # if self.irreps_node_embedding[idx][1].l ==0:
            #     node_features.append(torch.zeros(
            #                         (B,L,self.irreps_node_embedding[idx][0]),
            #                          dtype=edge_dis.dtype,
            #                          device = edge_dis.device))
            #     continue

            lx = o3.spherical_harmonics(
                l=self.irreps_node_embedding[idx][1].l,
                x=edge_vec,
                normalize=True,
                normalization="norm",
            )  # * adj.reshape(B,L,L,1) #B*L*L*(2l+1)
            edge_fea = edge_dis_embed
            edge_fea = self.gbf_projs[idx](edge_dis_embed)
            # lx_embed = torch.einsum("bmnd,bnh->bmhd",lx,node_embed) #lx:B*L*L*(2l+1)  node_embed:B*L*hidden
            lx_embed = torch.einsum(
                "bmnd,bmnh->bmhd", lx, edge_fea
            )  # lx:B*L*L*(2l+1)  edge_fea:B*L*L*number of basis

            lx_embed = torch.matmul(self.weight_list[idx], lx_embed).reshape(
                B, L, -1
            )  # self.weight_list[idx]:irreps_channel*hidden, lx_embed:B*L*hidden*(2l+1)
            node_features.append(lx_embed)

        node_features = torch.cat(node_features, dim=-1) / self.avg_aggregate_num**0.5
        node_features = self.proj(node_features)
        return node_features


class EdgeDegreeEmbeddingNetwork_higherorder_bias(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding,
        avg_aggregate_num=10,
        number_of_basis=32,
        cutoff=None,
        time_embed=False,
        **kwargs,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.irreps_node_embedding = (
            o3.Irreps(irreps_node_embedding)
            if isinstance(irreps_node_embedding, str)
            else irreps_node_embedding
        )
        if self.irreps_node_embedding[0][1].l != 0:
            raise ValueError("node embedding must have sph order 0 embedding.")
        self.gbf = GaussianLayer(number_of_basis)  # default output_dim = 128
        self.gbf_projs = nn.ModuleList()

        self.scalar_dim = self.irreps_node_embedding[0][0]
        self.embed = nn.Embedding(256, self.scalar_dim)
        self.embed.weight.data.normal_(0, 1 / self.scalar_dim**0.5)
        if time_embed:
            self.time_embed_proj = nn.Sequential(
                nn.Linear(self.scalar_dim, self.scalar_dim, bias=True),
                nn.SiLU(),
                nn.Linear(self.scalar_dim, number_of_basis, bias=True),
            )

        self.weight_list = nn.ParameterList()
        self.bias_list = nn.ParameterList()
        for idx in range(len(self.irreps_node_embedding)):
            self.gbf_projs.append(
                RadialProfile([number_of_basis, number_of_basis], use_layer_norm=False)
            )

            out_feature = self.irreps_node_embedding[idx][0]
            weight = torch.nn.Parameter(torch.randn(out_feature, number_of_basis))
            bound = 1 / math.sqrt(number_of_basis)
            torch.nn.init.uniform_(weight, -bound, bound)
            self.weight_list.append(weight)

            bias = torch.nn.Parameter(torch.zeros(1, 1, out_feature, 1))
            self.bias_list.append(bias)

        self.proj = IrrepsLinear(self.irreps_node_embedding, self.irreps_node_embedding)
        self.avg_aggregate_num = avg_aggregate_num

    def forward(
        self,
        node_input,
        node_pos,
        edge_dis,
        atomic_numbers,
        edge_vec,
        batched_data,
        attn_mask,
        time_embed=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        node_pos : postiion
            tensor of shape ``(B, L, 3)``

        edge_scalars : rbf of distance
            tensor of shape ``(B, L, L, number_of_basis)``

        edge_dis : distances of all node pairs
            tensor of shape ``(B, L, L)``
        """

        B, L = node_pos.shape[:2]
        self.embed(atomic_numbers)  # B*L*hidden

        # edge_vec = node_pos.unsqueeze(2) - node_pos.unsqueeze(1)  # B, L, L, 3
        node_type_edge = batched_data["node_type_edge"]
        edge_dis_embed = self.gbf(edge_dis, node_type_edge.long())
        if time_embed is not None:
            edge_dis_embed += self.time_embed_proj(time_embed).unsqueeze(-2)
        edge_dis_embed = torch.where(attn_mask, 0, edge_dis_embed)
        node_features = []
        for idx in range(len(self.irreps_node_embedding)):
            # if self.irreps_node_embedding[idx][1].l ==0:
            #     node_features.append(torch.zeros(
            #                         (B,L,self.irreps_node_embedding[idx][0]),
            #                          dtype=edge_dis.dtype,
            #                          device = edge_dis.device))
            #     continue

            lx = o3.spherical_harmonics(
                l=self.irreps_node_embedding[idx][1].l,
                x=edge_vec,
                normalize=True,
                normalization="norm",
            )  # * adj.reshape(B,L,L,1) #B*L*L*(2l+1)
            edge_fea = self.gbf_projs[idx](edge_dis_embed)
            # lx_embed = torch.einsum("bmnd,bnh->bmhd",lx,node_embed) #lx:B*L*L*(2l+1)  node_embed:B*L*hidden
            lx_embed = torch.einsum(
                "bmnd,bmnh->bmhd", lx, edge_fea
            )  # lx:B*L*L*(2l+1)  edge_fea:B*L*L*number of basis

            lx_embed = (
                torch.matmul(self.weight_list[idx], lx_embed) + self.bias_list[idx]
            ).reshape(
                B, L, -1
            )  # self.weight_list[idx]:irreps_channel*hidden, lx_embed:B*L*hidden*(2l+1)
            node_features.append(lx_embed)

        node_features = torch.cat(node_features, dim=-1) / self.avg_aggregate_num**0.5
        node_features = self.proj(node_features)
        return node_features


class E2former(torch.nn.Module):
    def __init__(
        self,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        pbc_max_radius=5,
        max_radius=15.0,
        basis_type="gaussian",
        number_of_basis=128,
        num_attn_heads=4,
        attn_scalar_head=32,
        irreps_head="32x0e+16x1e+8x2e",
        rescale_degree=False,
        nonlinear_message=False,
        norm_layer="layer",
        alpha_drop=0.2,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        atom_type_cnt=256,
        tp_type=None,
        attn_type="v0",
        edge_embedtype="default",
        attn_biastype="share",  # add
        ffn_type="default",
        add_rope=True,
        time_embed=False,
        # mean=None,
        # std=None,
        # scale=None,
        # atomref=None,
        **kwargs,
    ):
        super().__init__()
        self.tp_type = tp_type
        self.attn_type = attn_type
        self.pbc_max_radius = pbc_max_radius  #
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.add_rope = add_rope
        self.time_embed = time_embed
        # self.task_mean = mean
        # self.task_std = std
        # self.scale = scale
        # self.register_buffer("atomref", atomref)

        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.num_layers = num_layers
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.irreps_head = irreps_head
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        if "0e" not in self.irreps_node_embedding:
            raise ValueError("sorry, the irreps node embedding must have 0e embedding")

        self.default_node_embedding = torch.nn.Embedding(
            atom_type_cnt, self.irreps_node_embedding[0][0]
        )
        self._node_scalar_dim = self.irreps_node_embedding[0][0]
        self._node_vec_dim = (
            self.irreps_node_embedding.dim - self.irreps_node_embedding[0][0]
        )

        ## this is for f( r_ij )
        self.basis_type = basis_type
        self.attn_biastype = attn_biastype
        self.heads2basis = nn.Linear(
            self.num_attn_heads, self.number_of_basis, bias=True
        )
        if self.basis_type == "gaussian":
            self.rbf = GaussianRadialBasisLayer(
                self.number_of_basis, cutoff=self.max_radius
            )
        elif self.basis_type == "gaussiansmear":
            self.rbf = GaussianSmearing(
                self.number_of_basis, cutoff=self.max_radius, basis_width_scalar=2
            )
        elif self.basis_type == "bessel":
            self.rbf = RadialBasis(
                self.number_of_basis,
                cutoff=self.max_radius,
                rbf={"name": "spherical_bessel"},
            )
        else:
            raise ValueError

        # edge
        if edge_embedtype == "default" or edge_embedtype == "highorder":
            self.edge_deg_embed_dense = EdgeDegreeEmbeddingNetwork_higherorder(
                self.irreps_node_embedding,
                _AVG_DEGREE,
                cutoff=self.max_radius,
                number_of_basis=self.number_of_basis,
                time_embed=self.time_embed,
            )
        elif edge_embedtype == "highorder_bias":
            self.edge_deg_embed_dense = EdgeDegreeEmbeddingNetwork_higherorder_bias(
                self.irreps_node_embedding,
                _AVG_DEGREE,
                cutoff=self.max_radius,
                number_of_basis=self.number_of_basis,
                time_embed=self.time_embed,
            )
        else:
            raise ValueError("please check edge embedtype")

        self.blocks = torch.nn.ModuleList()
        for i in range(self.num_layers):
            blk = TransBlock(
                irreps_node_input=self.irreps_node_embedding,
                irreps_node_output=self.irreps_node_embedding,
                attn_weight_input_dim=self.number_of_basis,
                num_attn_heads=self.num_attn_heads,
                attn_scalar_head=self.attn_scalar_head,
                irreps_head=self.irreps_head,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                norm_layer=self.norm_layer,
                tp_type=self.tp_type,
                attn_type=self.attn_type,
                ffn_type=ffn_type,
                layer_id=i,
                add_rope=self.add_rope,
                max_radius=max_radius,
            )
            self.blocks.append(blk)

        self.scalar_dim = self.irreps_node_embedding[0][0]
        self.norm_final_noise = get_norm_layer(norm_layer)(self.irreps_node_embedding)
        self.output_proj_noise = IrrepsLinear(
            self.irreps_node_embedding,
            f"{self.scalar_dim}x0e+{self.scalar_dim}x1e",
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irreps

        self.norm_final = get_norm_layer(norm_layer)(self.irreps_node_embedding)
        self.output_proj = IrrepsLinear(
            self.irreps_node_embedding,
            f"{self.scalar_dim}x0e+{self.scalar_dim}x1e",
            bias=True,
            act="smoothleakyrelu",
        )  # B*L*irreps
        # self.output_ffn = FeedForwardNetwork(
        #     irreps_node_input=self.irreps_node_embedding,  # self.concat_norm_output.irreps_out,
        #     irreps_node_output=o3.Irreps(f"{self.irreps_node_embedding[0][0]}x1e"),
        #     proj_drop=proj_drop,
        # )
        self.apply(self._init_weights)

    def reset_parameters(self):
        warnings.warn("sorry, output model not implement reset parameters")

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (
                isinstance(module, torch.nn.Linear)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, RMS_Norm_SH)
                # or isinstance(module, EquivariantGraphNorm) # TODO
                or isinstance(module, GaussianRadialBasisLayer)
                or isinstance(module, RadialBasis)
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, torch.nn.Linear)
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)

    def forward(
        self,
        batched_data: Dict,
        token_embedding: torch.Tensor,
        mixed_attn_bias=None,
        padding_mask: torch.Tensor = None,
        pbc_expand_batched: Optional[Dict] = None,
        time_embed: Optional[torch.Tensor] = None,
        sepFN=False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of the PSMEncoder class.
        Args:
            x (torch.Tensor): Input tensor, [L, B, H].
            padding_mask (torch.Tensor): Padding mask, [B, L].
            batched_data (Dict): Input data for the forward pass.
            masked_token_type (torch.Tensor): The masked token type, [B, L].
        Returns:
            torch.Tensor: Encoded tensor, [B, L, H].
        example:
        batch: attn_bias torch.Size([4, 65, 65])
        batch: attn_edge_type torch.Size([4, 64, 64, 1])
        batch: spatial_pos torch.Size([4, 64, 64]) -> shortest path
        batch: in_degree torch.Size([4, 64])
        batch: out_degree torch.Size([4, 64])
        batch: token_id torch.Size([4, 64])
        batch: node_attr torch.Size([4, 64, 1])
        batch: edge_input torch.Size([4, 64, 64, 5, 1])
        batch: energy torch.Size([4])
        batch: forces torch.Size([4, 64, 3])
        batch: pos torch.Size([4, 64, 3])
        batch: node_type_edge torch.Size([4, 64, 64, 2])
        batch: pbc torch.Size([4, 3])
        batch: cell torch.Size([4, 3, 3])
        batch: num_atoms torch.Size([4])
        batch: is_periodic torch.Size([4])
        batch: is_molecule torch.Size([4])
        batch: is_protein torch.Size([4])
        batch: protein_masked_pos torch.Size([4, 64, 3])
        batch: protein_masked_aa torch.Size([4, 64])
        batch: protein_mask torch.Size([4, 64, 3])
        batch: init_pos torch.Size([4, 64, 3])
        batch: time_embed torch.Size([4, 64, node_embedding_scalar])

        example:
        import torch
        from sfm.models.psm.equivariant.e2former import E2former
        func = E2former(
                irreps_node_embedding="24x0e+16x1e+8x2e",
                num_layers=6,
                max_radius=5.0,
                basis_type="gaussian",
                number_of_basis=128,
                num_attn_heads=4,
                attn_scalar_head = 32,
                irreps_head="32x0e+16x1e+8x2e",
                alpha_drop=0,)
        batched_data = {}
        B,L = 4,100

        node_pos = torch.randn(B,L,3)*100
        token_embedding = torch.randn(B,L,24)
        token_id = torch.randint(0,100,(B,L))
        batched_data = {
            "pos":node_pos,
            # "token_embedding":token_embedding,
            "token_id":token_id,
        }
        out = func(batched_data, None, None)


        tr = o3._wigner.wigner_D(1,
                            torch.Tensor([0.8]),
                            torch.Tensor([0.5]),
                            torch.Tensor([0.3]))
        pos = (tr@(node_pos.unsqueeze(-1))).squeeze(-1)
        batched_data = {
            "pos":pos,
            # "token_embedding":token_embedding,
            "token_id":token_id,
        }
        out_tr = func(batched_data, None, None)


        print(torch.max(torch.abs(out_tr[0]-out[0])),
            torch.max(torch.abs(out_tr[1]-(tr@out[1]))))
        """

        tensortype = self.default_node_embedding.weight.dtype
        node_pos = batched_data["pos"].to(tensortype)
        device = node_pos.device
        B, L, _ = node_pos.shape
        atomic_numbers = batched_data["masked_token_type"].reshape(B, L)
        if (time_embed is not None) and self.time_embed:
            time_embed = time_embed.to(dtype=tensortype)
        else:
            time_embed = None
        mol_type = 0  # torch.any(batched_data["is_molecule"]):
        node_pos_right = node_pos
        if torch.any(batched_data["is_periodic"]):
            mol_type = 1
            #  batched_data["outcell_index"] # B*L2
            # batched_data["outcell_index_0"] # B*L2
            batched_data.update(pbc_expand_batched)
            L2 = batched_data["outcell_index"].shape[1]
            batched_data["outcell_index_0"] = (
                torch.arange(B).reshape(B, 1).repeat(1, L2)
            )
            node_pos_right = torch.cat(
                [node_pos, batched_data["expand_pos"].float()], dim=1
            )

            atomic_numbers_i = (
                atomic_numbers.unsqueeze(2).repeat(1, 1, L + L2).unsqueeze(-1)
            )
            atomic_numbers_j = torch.cat(
                [
                    atomic_numbers,
                    atomic_numbers[
                        batched_data["outcell_index_0"], batched_data["outcell_index"]
                    ],
                ],
                dim=1,
            )
            atomic_numbers_j = (
                atomic_numbers_j.unsqueeze(1).repeat(1, L, 1).unsqueeze(-1)
            )

            batched_data["node_type_edge"] = torch.cat(
                [atomic_numbers_i, atomic_numbers_j], dim=-1
            )

        if torch.any(batched_data["is_protein"]):
            mol_type = 2
        batched_data["mol_type"] = mol_type

        edge_vec = node_pos.unsqueeze(2) - node_pos_right.unsqueeze(1)
        dist = torch.norm(edge_vec, dim=-1)  # B*L*L Attention: ego-connection is 0 here
        dist = torch.where(dist < 1e-4, 1000, dist)
        dist_embedding = self.rbf(dist.reshape(-1)).reshape(
            B, L, -1, self.number_of_basis
        )  # [B, L, L, number_of_basis]
        if mol_type != 1:
            attn_mask = dist > self.max_radius
            attn_mask = (
                attn_mask + padding_mask.unsqueeze(1) + padding_mask.unsqueeze(2)
            ).unsqueeze(-1)
        else:
            attn_mask = dist > self.pbc_max_radius
            attn_mask = (
                attn_mask
                + torch.cat(
                    [padding_mask, batched_data["expand_mask"]], dim=-1
                ).unsqueeze(1)
                + padding_mask.unsqueeze(2)
            ).unsqueeze(-1)

        if token_embedding is not None:
            atom_embedding = token_embedding.permute(1, 0, 2)  # [L, B, D] => [B, L, D]
        else:
            atom_embedding = self.default_node_embedding(atomic_numbers)

        # if torch.any(torch.isnan(atom_embedding)):assert(False)

        edge_degree_embedding_dense = self.edge_deg_embed_dense(
            atom_embedding,
            node_pos,
            dist,
            edge_scalars=dist_embedding,
            batch=None,
            attn_mask=attn_mask,
            atomic_numbers=atomic_numbers,
            edge_vec=edge_vec,
            batched_data=batched_data,
            time_embed=time_embed,
        )
        if token_embedding is not None:
            node_irreps = torch.cat(
                [
                    atom_embedding[..., : self._node_scalar_dim],
                    edge_degree_embedding_dense[..., self._node_scalar_dim :],
                ],
                dim=-1,
            )
        else:
            atom_embedding = torch.cat(
                [
                    atom_embedding,
                    torch.zeros(
                        B, L, self._node_vec_dim, dtype=tensortype, device=device
                    ),
                ],
                dim=2,
            )
            node_irreps = atom_embedding + edge_degree_embedding_dense
        # if torch.any(torch.isnan(node_irreps)):assert(False)
        if mixed_attn_bias is not None:
            if len(mixed_attn_bias.shape) != 5:  # (bsz, num_heads, tgt_len, src_len)
                mixed_attn_bias = mixed_attn_bias.permute(0, 2, 3, 1)
            else:  # (layers,bsz, num_heads, tgt_len, src_len)
                mixed_attn_bias = mixed_attn_bias.permute(0, 1, 3, 4, 2)
            mixed_attn_bias = torch.where(
                mixed_attn_bias == -torch.inf, 0, mixed_attn_bias
            )
            # mixed_attn_bias = self.heads2basis(mixed_attn_bias)
            if self.attn_biastype == "add":
                mixed_attn_bias = mixed_attn_bias + dist_embedding
        node_irreps_noise = node_irreps
        for i, blk in enumerate(self.blocks):
            if mixed_attn_bias is not None:
                if len(mixed_attn_bias.shape) != 5:
                    dist_embedding_i = mixed_attn_bias
                else:
                    dist_embedding_i = mixed_attn_bias[i]
            else:
                dist_embedding_i = dist_embedding
            node_irreps = blk(
                node_pos=node_pos,
                node_irreps=node_irreps,
                edge_dis=dist,
                attn_weight=dist_embedding_i,
                atomic_numbers=atomic_numbers,
                batch=None,  #
                attn_mask=attn_mask,
                batched_data=batched_data,
                add_rope=self.add_rope,
            )
            if i == self.num_layers - 2:
                node_irreps_noise = node_irreps
            # if torch.any(torch.isnan(node_irreps)):assert(False)

        node_irreps = self.output_proj(self.norm_final(node_irreps))
        node_embedding_ef = node_irreps[..., : self.scalar_dim]  # the part of order 0
        node_vec_ef = (
            node_irreps[..., self.scalar_dim :]
            .reshape(B, L, self.scalar_dim, 3)
            .permute(0, 1, 3, 2)
        )
        if sepFN:
            node_irreps_noise = self.output_proj_noise(
                self.norm_final_noise(node_irreps_noise)
            )
            node_embedding_noise = node_irreps_noise[
                ..., : self.scalar_dim
            ]  # the part of order 0
            node_vec_noise = (
                node_irreps_noise[..., self.scalar_dim :]
                .reshape(B, L, self.scalar_dim, 3)
                .permute(0, 1, 3, 2)
            )
            return node_embedding_noise, node_vec_noise, node_embedding_ef, node_vec_ef

        else:
            return node_embedding_ef, node_vec_ef
