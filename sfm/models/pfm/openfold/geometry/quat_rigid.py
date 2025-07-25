# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from sfm.models.pfm.openfold.geometry.rigid_matrix_vector import Rigid3Array
from sfm.models.pfm.openfold.geometry.rotation_matrix import Rot3Array
from sfm.models.pfm.openfold.geometry.vector import Vec3Array
from sfm.models.pfm.openfold.primitives import Linear


class QuatRigid(nn.Module):
    def __init__(self, c_hidden, full_quat):
        super().__init__()
        self.full_quat = full_quat
        if self.full_quat:
            rigid_dim = 7
        else:
            rigid_dim = 6

        self.linear = Linear(c_hidden, rigid_dim, init="final", precision=torch.float32)

    def forward(self, activations: torch.Tensor) -> Rigid3Array:
        # NOTE: During training, this needs to be run in higher precision
        rigid_flat = self.linear(activations)

        rigid_flat = torch.unbind(rigid_flat, dim=-1)
        if self.full_quat:
            qw, qx, qy, qz = rigid_flat[:4]
            translation = rigid_flat[4:]
        else:
            qx, qy, qz = rigid_flat[:3]
            qw = torch.ones_like(qx)
            translation = rigid_flat[3:]

        rotation = Rot3Array.from_quaternion(
            qw,
            qx,
            qy,
            qz,
            normalize=True,
        )
        translation = Vec3Array(*translation)
        return Rigid3Array(rotation, translation)
