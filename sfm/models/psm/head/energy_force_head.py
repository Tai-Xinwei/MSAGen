# -*- coding: utf-8 -*-
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple

import torch
from e3nn import o3
from torch import nn
from torch.autograd import grad
from torch_scatter import scatter

from sfm.models.psm.equivariant.equiformer.dp_attention_transformer_md17 import _RESCALE
from sfm.models.psm.equivariant.equiformer.fast_activation import Activation, Gate
from sfm.models.psm.equivariant.equiformer.graph_attention_transformer import (
    FeedForwardNetwork,
)
from sfm.models.psm.equivariant.equiformer.tensor_product_rescale import (
    FullyConnectedTensorProductRescale,
    LinearRS,
    TensorProductRescale,
    irreps2gate,
)

##
# Scalar and EquivariantScalar is for visnet kindof model (EGNN)
# EquivariantScalar_viaTP is for equiformer kind of tensor product model
# __all__ = ["EquivariantScalar_viaTP"]


class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, allow_prior_model):
        super(OutputModel, self).__init__()
        self.allow_prior_model = allow_prior_model

    def reset_parameters(self):
        warnings.warn("sorry, output model not implement reset parameters")


class InvariantHeadviaTP(OutputModel):
    def __init__(
        self,
        irrep_features,
        node_attr_dim=None,
        activation="silu",
        allow_prior_model=True,
    ):
        super().__init__(allow_prior_model=allow_prior_model)

        if activation != "silu":
            raise ValueError("This model supports only 'silu' activation.")

        if not isinstance(irrep_features, o3.Irreps):
            irrep_features = o3.Irreps(irrep_features)

        self.irreps_feature = irrep_features
        if node_attr_dim is None:
            self.irreps_node_attr = o3.Irreps(f"{self.irreps_feature[0][0]}x0e")
        else:
            self.irreps_node_attr = o3.Irreps(f"{node_attr_dim}x0e")

        self.equivariant_layer = FeedForwardNetwork(
            irreps_node_input=self.irreps_feature,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_attr,
            irreps_mlp_mid=self.irreps_feature,
            proj_drop=0.0,
        )

        self.output_network = nn.Sequential(
            LinearRS(self.irreps_node_attr, self.irreps_node_attr, _RESCALE),
            Activation(
                self.irreps_node_attr, [torch.nn.SiLU()] * len(self.irreps_node_attr)
            ),
            LinearRS(self.irreps_node_attr, o3.Irreps("1x0e"), _RESCALE),
        )

        self.reset_parameters()

    def forward(self, batch):
        features = batch["node_vec"]  # features after NodeEncoder
        attributes = batch["node_embedding"]  # just embedding

        outputs = self.equivariant_layer(features, attributes)
        outputs = self.output_network(outputs)

        return outputs


class EquivariantHeadviaTP(OutputModel):
    def __init__(
        self,
        irrep_features,
        out_irreps="1x1e",
        node_attr_dim=None,
        activation="silu",
        allow_prior_model=True,
    ):
        super().__init__(allow_prior_model=allow_prior_model)

        if activation != "silu":
            raise ValueError("This model supports only 'silu' activation.")

        if not isinstance(irrep_features, o3.Irreps):
            irrep_features = o3.Irreps(irrep_features)

        self.irreps_feature = irrep_features
        if node_attr_dim is None:
            self.irreps_node_attr = o3.Irreps(f"{self.irreps_feature[0][0]}x0e")
        else:
            self.irreps_node_attr = o3.Irreps(f"{node_attr_dim}x0e")

        self.equivariant_layer = FeedForwardNetwork(
            irreps_node_input=self.irreps_feature,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_attr,
            irreps_mlp_mid=self.irreps_feature,
            proj_drop=0.0,
        )

        self.output_network = nn.Sequential(
            LinearRS(self.irreps_node_attr, self.irreps_node_attr, _RESCALE),
            Activation(
                self.irreps_node_attr, [torch.nn.SiLU()] * len(self.irreps_node_attr)
            ),
            LinearRS(self.irreps_node_attr, o3.Irreps("1x0e"), _RESCALE),
        )

        self.reset_parameters()

    def forward(self, batch):
        features = batch["node_vec"]  # features after NodeEncoder
        attributes = batch["node_embedding"]  # just embedding

        outputs = self.equivariant_layer(features, attributes)
        outputs = self.output_network(outputs)

        return outputs


class EnergyForceHead(nn.Module):
    def __init__(
        self,
        irreps_node_embedding=None,
        embedding_dimension=None,
        activation="silu",
        output_model=None,
        hami_model=None,
        prior_model=None,
        mean=None,
        std=None,
        output_model_noise=None,
        position_noise_scale=0.0,
        enable_energy=False,
        enable_forces=False,
        enable_hami=False,
    ):
        super(EnergyForceHead, self).__init__()
        self.output_model = getattr(output_model)(
            irrep_features=irreps_node_embedding,
            node_attr_dim=embedding_dimension,
            activation=activation,
        )

        self.prior_model = prior_model
        self.enable_energy = enable_energy
        self.enable_forces = enable_forces
        self.output_model_noise = output_model_noise
        self.position_noise_scale = position_noise_scale

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        if self.position_noise_scale > 0:
            self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))

        self.reset_parameters()

    def reset_parameters(self):
        if self.output_model is not None:
            self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self, batch_data):
        pos = batch_data["pos"]
        batch = batch_data["batch"]
        batch_data["atomic_numbers"] = batch_data["atomic_numbers"].reshape(-1)
        z = batch_data["atomic_numbers"]
        assert z.dim() == 1 and (z.dtype == torch.long or z.dtype == torch.int32)

        if self.enable_forces:
            pos.requires_grad_(True)

        if self.enable_energy or self.enable_forces:
            # apply the output network
            batch_data = self.output_model.pre_reduce(batch_data)

            pred_energy = batch_data["pred_energy"]
            # # apply prior model
            # if self.prior_model is not None:
            #     pred_energy = self.prior_model(pred_energy, z, pos, batch)
            # aggregate atoms
            pred_energy = scatter(pred_energy, batch, dim=0, reduce="add")
            # shift by data mean
            # scale by data standard deviation
            if self.std is not None:
                pred_energy = pred_energy * self.std
            if self.mean is not None:
                pred_energy = pred_energy + self.mean
            batch_data["pred_energy"] = pred_energy

            # # apply output model after reduction
            # out = self.output_model.post_reduce(out)

        # compute gradients with respect to coordinates
        if self.enable_forces:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(pred_energy)]
            dy = (
                -1
                * grad(
                    [pred_energy],
                    [pos],
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                )[0]
            )
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            batch_data["pred_forces"] = dy
        return batch_data


class AccumulatedNormalization(nn.Module):
    """Running normalization of a tensor."""

    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
        super(AccumulatedNormalization, self).__init__()

        self._epsilon = epsilon
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def update_statistics(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        return torch.sqrt(
            (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        ).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor):
        if self.training:
            self.update_statistics(batch)
        return (batch - self.mean) / self.std
