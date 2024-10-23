# -*- coding: utf-8 -*-
import math

import torch
from e3nn import o3
from e3nn.o3 import so3_generators

from sfm.models.psm.equivariant.equiformer_v2.equiformer_v2_oc20 import (
    CoefficientMappingModule,
)


def wigner_D(l, alpha, beta, gamma):
    r"""Wigner D matrix representation of :math:`SO(3)`.

    It satisfies the following properties:

    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \circ R_2) = D(R_1) \circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`
    * :math:`D(\text{rotation around Y axis})` has some property that allows us to use FFT in `ToS2Grid`

    Parameters
    ----------
    l : int
        :math:`l`

    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\alpha` around Y axis, applied third.

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\beta` around X axis, applied second.

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\gamma` around Y axis, applied first.

    Returns
    -------
    `torch.Tensor`
        tensor :math:`D^l(\alpha, \beta, \gamma)` of shape :math:`(2l+1, 2l+1)`
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    alpha = alpha[..., None, None] % (2 * math.pi)
    beta = beta[..., None, None] % (2 * math.pi)
    gamma = gamma[..., None, None] % (2 * math.pi)
    X = so3_generators(l).to(alpha.device)
    return (
        torch.matrix_exp(alpha * X[1])
        @ torch.matrix_exp(beta * X[0])
        @ torch.matrix_exp(gamma * X[1])
    )


class SO3_Rotation(torch.nn.Module):
    """
    Helper functions for Wigner-D rotations

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
    """

    def __init__(self, lmax, irreps="128x1e+64x1e"):
        super().__init__()
        self.lmax = lmax
        self.mapping = CoefficientMappingModule([self.lmax], [self.lmax])

        self.irreps = o3.Irreps(irreps) if isinstance(irreps, str) else irreps

    def set_wigner(self, rot_mat3x3):
        self.device, self.dtype = rot_mat3x3.device, rot_mat3x3.dtype

        start_lmax, end_lmax = 0, self.lmax
        # Compute Wigner matrices from rotation matrix
        x = rot_mat3x3 @ rot_mat3x3.new_tensor([0.0, 1.0, 0.0])
        alpha, beta = o3.xyz_to_angles(x)
        R = (
            o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
            @ rot_mat3x3
        )
        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

        (end_lmax + 1) ** 2 - (start_lmax) ** 2
        start = 0
        self.wigner = []
        self.wigner_inv = []

        for lmax in range(start_lmax, end_lmax + 1):
            block = wigner_D(lmax, alpha, beta, gamma)
            end = start + block.size()[1]
            self.wigner.append(block.detach())
            self.wigner_inv.append(torch.transpose(block.detach(), 1, 2).contiguous())
            start = end

    def init_edge_rot_mat(self, edge_distance_vec):
        edge_vec_0 = edge_distance_vec
        edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

        norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1) + 1e-8)

        edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
        edge_vec_2 = edge_vec_2 / (
            torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
        )
        # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
        # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
        edge_vec_2b = edge_vec_2.clone()
        edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
        edge_vec_2b[:, 1] = edge_vec_2[:, 0]
        edge_vec_2c = edge_vec_2.clone()
        edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
        edge_vec_2c[:, 2] = edge_vec_2[:, 1]
        vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
        vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
        edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

        vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
        # Check the vectors aren't aligned
        assert torch.max(vec_dot) < 0.99

        norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
        norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True)))
        norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1))
        norm_y = torch.cross(norm_x, norm_z, dim=1)
        norm_y = norm_y / (torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True)))

        # Construct the 3D rotation matrix
        norm_x = norm_x.view(-1, 3, 1)
        norm_y = -norm_y.view(-1, 3, 1)
        norm_z = norm_z.view(-1, 3, 1)

        edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
        edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)
        # Make sure the atoms are far enough apart
        # assert torch.min(edge_vec_0_distance) < 0.0001
        if torch.min(edge_vec_0_distance) < 0.0001:
            edge_rot_mat[edge_vec_0_distance < 0.0001] = torch.eye(
                3, device=edge_rot_mat.device
            )[None, :, :]
            # print("Error edge_vec_0_distance: {}".format(torch.min(edge_vec_0_distance)))

        return edge_rot_mat.detach()

    # Rotate the embedding
    def rotate(self, embedding, irreps=None):
        shape = list(embedding.shape[:-1])
        num = embedding.shape[:-1].numel()
        embedding = embedding.reshape(num, -1)

        # embedding: N*(128x0e+64x1e+32x2e)
        irreps = self.irreps if irreps is None else irreps
        start = 0
        out = []
        for i in range(len(irreps)):
            l = irreps[i][1].l
            mul = irreps[i][0]
            cur = self.wigner[l].unsqueeze(dim=1) @ (
                embedding[:, start : start + mul * (2 * l + 1)].reshape(
                    num, mul, 2 * l + 1, 1
                )
            )
            out.append(cur.reshape(num, -1))
            start += mul * (2 * l + 1)

        embedding = torch.cat(out, dim=-1).reshape(shape + [-1])
        return embedding

    # Rotate the embedding by the inverse of the rotation matrix
    def rotate_inv(self, embedding, irreps=None):
        shape = list(embedding.shape[:-1])
        num = embedding.shape[:-1].numel()
        embedding = embedding.reshape(num, -1)

        # embedding: N*(128x0e+64x1e+32x2e)
        irreps = self.irreps if irreps is None else irreps
        start = 0
        out = []
        for i in range(len(irreps)):
            l = irreps[i][1].l
            mul = irreps[i][0]
            cur = self.wigner_inv[l].unsqueeze(dim=1) @ (
                embedding[:, start : start + mul * (2 * l + 1)].reshape(
                    num, mul, 2 * l + 1, 1
                )
            )
            out.append(cur.reshape(num, -1))
            start += mul * (2 * l + 1)

        embedding = torch.cat(out, dim=-1).reshape(shape + [-1])
        return embedding


class SO2_Convolution(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
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
        irreps_type = []
        for i in range(len(self.irreps_in)):
            l = self.irreps_in[i][1].l
            mul = self.irreps_in[i][0]
            irreps_type.extend(mul * list(range(-l, l + 1)))

        self.irreps_type = torch.tensor(
            irreps_type, dtype=torch.long, requires_grad=False
        )

        self.fc_list = torch.nn.ModuleList()
        self.mlengths = []
        for m in range(len(self.irreps_in)):
            mlength = torch.sum(self.irreps_type == m)
            if m == 0:
                self.fc_list.append(torch.nn.Linear(mlength, mlength, bias=True))
            else:
                self.fc_list.append(torch.nn.Linear(mlength, mlength * 2, bias=False))
            self.mlengths.append(mlength)

    def forward(self, embedding):
        shape = list(embedding.shape[:-1])
        num = embedding.shape[:-1].numel()
        embedding = embedding.reshape(num, -1)

        out = torch.zeros_like(embedding)
        # embedding rot
        for m in range(len(self.irreps_in)):
            if m == 0:
                out[:, self.irreps_type == m] = self.fc_list[m](
                    embedding[:, self.irreps_type == m]
                )
            else:
                m_plus = self.fc_list[m](embedding[:, self.irreps_type == m])
                m_minus = self.fc_list[m](embedding[:, self.irreps_type == -m])

                fealen = self.mlengths[m]
                x_m_r = (
                    m_plus[:, :fealen] - m_minus[:, fealen:]
                )  # x_r[:, 0] - x_i[:, 1]
                x_m_i = (
                    m_minus[:, :fealen] + m_plus[:, fealen:]
                )  # x_r[:, 1] + x_i[:, 0]

                out[:, self.irreps_type == m] = x_m_r
                out[:, self.irreps_type == -m] = x_m_i
        return out.reshape(shape + [-1])
        # embedding  rot back


class SO2_Convolution_sameorder(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
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
        irreps_type = []
        for i in range(len(self.irreps_in)):
            l = self.irreps_in[i][1].l
            mul = self.irreps_in[i][0]
            irreps_type.extend(mul * list(range(-l, l + 1)))

        self.irreps_type = torch.tensor(
            irreps_type, dtype=torch.long, requires_grad=False
        )

        self.fc_list = torch.nn.ModuleList()
        self.mlengths = []
        for m in range(len(self.irreps_in)):
            l = self.irreps_in[m][1].l
            mul = self.irreps_in[m][0]
            if l == 0:
                self.fc_list.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            mul * (2 * l + 1), mul * (2 * l + 1), bias=True
                        ),
                        torch.nn.LeakyReLU(negative_slope=0.1),
                    ),
                    torch.nn.Linear(mul * (2 * l + 1), mul * (2 * l + 1), bias=True),
                )
            else:
                self.fc_list.append(
                    torch.nn.Linear(mul * (2 * l + 1), mul * (2 * l + 1), bias=False)
                )

    def forward(self, input_embedding):
        shape = list(input_embedding.shape[:-1])
        num = input_embedding.shape[:-1].numel()
        input_embedding = input_embedding.reshape(num, -1)

        output_embedding = []
        start = 0
        for m in range(len(self.irreps_in)):
            l = self.irreps_in[m][1].l
            mul = self.irreps_in[m][0]

            out = self.fc_list[m](input_embedding[:, start : start + mul * (2 * l + 1)])
            start += mul * (2 * l + 1)
            output_embedding.append(out)

        output_embedding = torch.cat(output_embedding, dim=1)
        output_embedding = output_embedding.reshape(shape + [-1])
        return output_embedding
