# -*- coding: utf-8 -*-
import math
import time
import warnings
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import Activation, FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, Linear, TensorProduct

try:
    from fairchem.core.models.base import BaseModel
except:
    from fairchem.core.models.base import HydraModel as BaseModel

from torch import logical_not, nn
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_scatter import scatter

from .equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20Backbone
from .qhnet import (
    ConvNetLayer,
    ExponentialBernsteinRadialBasisFunctions,
    InnerProduct,
    NormGate,
    SelfNetLayer,
    get_feasible_irrep,
    get_nonlinear,
)

# from .utils import construct_o3irrps, construct_o3irrps_base, generate_graph
from .utils import construct_o3irrps, construct_o3irrps_base, generate_graph


class PairNetLayer_symmetry(torch.nn.Module):
    def __init__(
        self,
        irrep_in_node,
        irrep_bottle_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        resnet: bool = True,
        invariant_layers=1,
        invariant_neurons=8,
        tp_mode="uuu",
        nonlinear="ssp",
    ):
        super().__init__()
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        self.irrep_in_node = (
            irrep_in_node
            if isinstance(irrep_in_node, o3.Irreps)
            else o3.Irreps(irrep_in_node)
        )
        self.irrep_bottle_hidden = (
            irrep_bottle_hidden
            if isinstance(irrep_bottle_hidden, o3.Irreps)
            else o3.Irreps(irrep_bottle_hidden)
        )
        self.irrep_out = (
            irrep_out if isinstance(irrep_out, o3.Irreps) else o3.Irreps(irrep_out)
        )
        self.sh_irrep = (
            sh_irrep if isinstance(sh_irrep, o3.Irreps) else o3.Irreps(sh_irrep)
        )

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_in_node, _ = get_feasible_irrep(
            self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden
        )

        self.norm_gate_pre = NormGate(self.irrep_in_node)
        self.linear_node_pair_input = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_tp_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

        self.irrep_tp_out_node_pair, instruction_node_pair = get_feasible_irrep(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_bottle_hidden,
            tp_mode=tp_mode,
        )

        self.linear_node_pair_inner = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )
        self.inner_product = InnerProduct(self.irrep_in_node)

        # tensor product for node pair : left and right
        self.tp_node_pair = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node_pair,
            instruction_node_pair,
            shared_weights=False,
            internal_weights=False,
        )

        self.fc_node_pair = FullyConnectedNet(
            [self.edge_attr_dim]
            + invariant_layers * [invariant_neurons]
            + [self.tp_node_pair.weight_numel],
            self.nonlinear_layer,
        )

        if self.irrep_in_node == self.irrep_out and resnet:
            self.resnet = True
        else:
            self.resnet = False

        self.node_residual = Linear(
            irreps_in=self.irrep_tp_out_node_pair,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

        self.norm_gate = NormGate(self.irrep_tp_out_node_pair)
        num_mul = 0
        for mul, ir in self.irrep_in_node:
            num_mul = num_mul + mul

        self.fc = nn.Sequential(
            # nn.Linear(self.irrep_in_node[0][0] + num_mul, self.irrep_in_node[0][0]),
            nn.Linear(num_mul, self.irrep_in_node[0][0]),
            nn.SiLU(),
            nn.Linear(self.irrep_in_node[0][0], self.tp_node_pair.weight_numel),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, node_attr, node_pair_attr=None):
        dst, src = data["full_edge_index"]
        node_attr_0 = self.linear_node_pair_inner(node_attr)
        s0 = self.inner_product(node_attr_0[dst], node_attr_0[src])[
            :, self.irrep_in_node.slices()[0].stop :
        ]
        s0 = torch.cat(
            [
                0.5 * node_attr_0[dst][:, self.irrep_in_node.slices()[0]]
                + 0.5 * node_attr_0[src][:, self.irrep_in_node.slices()[0]],
                s0,
            ],
            dim=-1,
        )

        node_attr = self.norm_gate_pre(node_attr)
        node_attr = self.linear_node_pair_input(node_attr)

        node_pair = self.tp_node_pair(
            node_attr[src],
            node_attr[dst],
            self.fc_node_pair(data["full_edge_attr"]) * self.fc(s0),
        )

        node_pair = self.norm_gate(node_pair)
        node_pair = self.node_residual(node_pair)

        if self.resnet and node_pair_attr is not None:
            node_pair = node_pair + node_pair_attr
        return node_pair


class QHNet_backbone_MADFT(nn.Module):
    def __init__(
        self,
        order=4,
        embedding_dimension=128,
        bottle_hidden_size=32,
        num_gnn_layers=5,
        max_radius=15,
        num_nodes=20,
        radius_embed_dim=32,
        use_equi_norm=False,
        **kwargs,
    ):  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        """
        Initialize the QHNet_backbone model.

        Args:
            order (int): The order of the spherical harmonics.
            embedding_dimension (int): The size of the hidden layer.
            bottle_hidden_size (int): The size of the bottleneck hidden layer.
            num_gnn_layers (int): The number of GNN layers.
            max_radius (int): The maximum radius cutoff.
            num_nodes (int): The number of nodes.
            radius_embed_dim (int): The dimension of the radius embedding.
        """

        super().__init__()
        self.order = order
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.hs = embedding_dimension
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.num_gnn_layers = num_gnn_layers
        self.node_embedding = nn.Embedding(num_nodes, self.hs)

        super().__init__()
        self.order = order
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.hs = embedding_dimension
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.num_gnn_layers = num_gnn_layers
        self.node_embedding = nn.Embedding(num_nodes, self.hs)

        self.init_sph_irrep = o3.Irreps(construct_o3irrps(1, order=order))

        self.init_sph_irrep = o3.Irreps(construct_o3irrps(1, order=order))

        self.irreps_node_embedding = construct_o3irrps_base(self.hs, order=order)
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs, order=order))
        self.hidden_irrep_base = o3.Irreps(self.irreps_node_embedding)
        self.hidden_bottle_irrep = o3.Irreps(construct_o3irrps(self.hbs, order=order))
        self.hidden_bottle_irrep_base = o3.Irreps(
            construct_o3irrps_base(self.hbs, order=order)
        )

        self.irreps_node_embedding = construct_o3irrps_base(self.hs, order=order)
        self.hidden_irrep = o3.Irreps(construct_o3irrps(self.hs, order=order))
        self.hidden_irrep_base = o3.Irreps(self.irreps_node_embedding)
        self.hidden_bottle_irrep = o3.Irreps(construct_o3irrps(self.hbs, order=order))
        self.hidden_bottle_irrep_base = o3.Irreps(
            construct_o3irrps_base(self.hbs, order=order)
        )

        self.input_irrep = o3.Irreps(f"{self.hs}x0e")
        self.radial_basis_functions = ExponentialBernsteinRadialBasisFunctions(
            self.radius_embed_dim, self.max_radius
        )
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}
        self.num_fc_layer = 1

        self.e3_gnn_layer = nn.ModuleList()
        for i in range(self.num_gnn_layers):
            input_irrep = self.input_irrep if i == 0 else self.hidden_irrep
            self.e3_gnn_layer.append(
                ConvNetLayer(
                    irrep_in_node=input_irrep,
                    irrep_hidden=self.hidden_irrep,
                    irrep_out=self.hidden_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hs,
                    sh_irrep=self.sh_irrep,
                    resnet=True,
                    use_norm_gate=True if i != 0 else False,
                    use_equi_norm=use_equi_norm,
                )
            )

    def reset_parameters(self):
        warnings.warn("reset parameter is not init in qhnet backbone model")

    def forward(self, batch_data):
        batch_data["ptr"] = torch.cat(
            [
                torch.Tensor([0]).to(batch_data["molecule_size"].device).int(),
                torch.cumsum(batch_data["molecule_size"], dim=0),
            ],
            dim=0,
        )

        edge_index = radius_graph(batch_data.pos, self.max_radius, batch_data.batch)
        edge_vec = (
            batch_data.pos[edge_index[0].long()] - batch_data.pos[edge_index[1].long()]
        )
        rbf_new = (
            self.radial_basis_functions(edge_vec.norm(dim=-1).unsqueeze(-1))
            .squeeze()
            .type(batch_data.pos.type())
        )
        edge_sh = o3.spherical_harmonics(
            self.sh_irrep,
            edge_vec[:, [1, 2, 0]],
            normalize=True,
            normalization="component",
        ).type(batch_data.pos.type())
        node_attr = self.node_embedding(batch_data.atomic_numbers.squeeze())

        (
            batch_data.node_attr,
            batch_data.edge_index,
            batch_data.edge_attr,
            batch_data.edge_sh,
        ) = (node_attr, edge_index, rbf_new, edge_sh)

        for layer_idx, layer in enumerate(self.e3_gnn_layer):
            node_attr = layer(batch_data, node_attr)

        batch_data["node_vec"] = node_attr
        batch_data["node_embedding"] = batch_data.node_attr

        return batch_data


class Equiformerv2SO2(BaseModel):
    def __init__(
        self,
        order=4,
        embedding_dim=128,
        bottle_hidden_size=32,
        num_gnn_layers=5,
        max_radius=12,
        num_nodes=200,
        radius_embed_dim=32,
        max_neighbors=100,
        load_pretrain="",
        **kwargs,
    ):  # maximum nuclear charge (+1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        """
        Initialize the QHNet_backbone model.

        Args:
            order (int): The order of the spherical harmonics.
            embedding_dimension (int): The size of the hidden layer.
            bottle_hidden_size (int): The size of the bottleneck hidden layer.
            num_gnn_layers (int): The number of GNN layers.
            max_radius (int): The maximum radius cutoff.
            num_nodes (int): The number of nodes.
            radius_embed_dim (int): The dimension of the radius embedding.
        """

        super().__init__()
        self.order = order
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.hs = embedding_dim
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.max_neighbors = max_neighbors
        self.num_gnn_layers = num_gnn_layers
        self.node_embedding = nn.Embedding(num_nodes, self.hs)

        self.init_sph_irrep = o3.Irreps(construct_o3irrps(1, order=order))

        self.irreps_node_embedding = construct_o3irrps_base(self.hs, order=order)

        # prevent double kwargs
        [
            kwargs.pop(x, None)
            for x in ["max_raius", "num_layers", "sphere_channels", "lmax_list"]
        ]
        self.node_attr_encoder = EquiformerV2_OC20Backbone(
            None,
            None,
            None,
            max_radius=max_radius,
            lmax_list=[order],
            sphere_channels=embedding_dim,
            num_layers=num_gnn_layers,
            **kwargs,
        )

        if load_pretrain != "":
            loaded_state_dict = torch.load(load_pretrain)["state_dict"]
            state_dict = {
                k.replace("module.module.", ""): v for k, v in loaded_state_dict.items()
            }
            self.node_attr_encoder.load_state_dict(state_dict, strict=False)

    def reset_parameters(self):
        warnings.warn("reset parameter is not init in qhnet backbone model")

    def forward(
        self,
        batched_data: Dict,
        token_embedding: torch.Tensor,
        mixed_attn_bias,
        padding_mask: torch.Tensor,
        pbc_expand_batched: Optional[Dict] = None,
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
        batch: spatial_pos torch.Size([4, 64, 64])
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
        """
        tensortype = self.node_embedding.weight.dtype
        # print("tensor type", tensortype)
        device = token_embedding.device
        token_embedding = token_embedding.transpose(0, 1)
        if (
            "pbc" in batched_data
            and batched_data["pbc"] is not None
            and torch.any(batched_data["pbc"])
        ):
            #
            # padding_mask: True for padding data
            # token_mask: True for token data
            bs, length = padding_mask.shape
            token_mask = logical_not(padding_mask).reshape(-1)
            new_batched_data = Data()
            new_batched_data.atomic_numbers = batched_data["masked_token_type"].reshape(
                -1
            )[token_mask]
            new_batched_data.ori_atomic_numbers = batched_data["masked_token_type"]
            new_batched_data.token_mask = token_mask
            new_batched_data.batch = (
                torch.arange(bs)
                .reshape(-1, 1)
                .repeat(1, length)
                .reshape(-1)
                .to(device)[token_mask]
                .to(device)
            )

            # 9999 means: for padding data, the pos will set a big number, thus no neighbor is included
            new_batched_data.pos = (
                batched_data["pos"].reshape(bs * length, 3).float()[token_mask]
            )

            new_batched_data.natoms = torch.sum(token_mask.reshape(bs, length), dim=1)
            new_batched_data.ptr = torch.cumsum(new_batched_data.natoms, dim=0)
            new_batched_data.cell = batched_data["cell"].float()
            (
                edge_index,
                edge_distance,
                edge_distance_vec,
                cell_offsets,
                cell_offset_distances,
                neighbors,
            ) = generate_graph(
                new_batched_data,
                self.max_radius,
                max_neighbors=self.max_neighbors,
                use_pbc=True,
                otf_graph=True,  # oc 20 , material data, these kind of data is saved or otf mode?
                enforce_max_neighbors_strictly=True,
            )

            new_batched_data.edge_index = edge_index
            new_batched_data.pos = new_batched_data.pos.to(tensortype)
            new_batched_data.cell = new_batched_data.cell.to(tensortype)
            new_batched_data.edge_distance = edge_distance.to(tensortype)
            new_batched_data.edge_distance_vec = edge_distance_vec.to(tensortype)
            # new_batched_data.cell_offsets = cell_offsets
            # new_batched_data.cell_offset_distances = cell_offset_distances
            # new_batched_data.neighbors = neighbors
            # print(edge_distance.shape,new_batched_data.natoms)
            # remove padding number
            new_batched_data.token_embedding = token_embedding.reshape(bs * length, -1)[
                token_mask
            ]

            node_attr = torch.zeros((bs * length, self.hs), device=device)
            node_vec = torch.zeros((bs * length, 3 * self.hs), device=device)

            if edge_distance_vec.numel() == 0:
                warnings.warn(
                    f"Edge_distance_vec is empty, skip batch and return zero matrix, please check. "
                    f"token is protein? {torch.any(batched_data['is_protein']).item()}, "
                    f"periodic? {torch.any(batched_data['is_periodic']).item()}, "
                    f"molecular? {torch.any(batched_data['is_molecule']).item()}"
                )
                return node_attr.reshape(bs, length, -1), node_vec.reshape(
                    bs, length, 3, -1
                )

            _node_vec = self.node_attr_encoder(new_batched_data)
            node_attr[token_mask] = _node_vec[:, : self.hs]
            node_vec[token_mask] = _node_vec[:, self.hs : 4 * self.hs]
            return node_attr.reshape(bs, length, -1), node_vec.reshape(
                bs, length, 3, -1
            )

        else:
            #
            # padding_mask: True for padding data
            # token_mask: True for token data

            bs, length = padding_mask.shape
            token_mask = logical_not(padding_mask).reshape(-1)
            new_batched_data = Data()
            new_batched_data.atomic_numbers = batched_data["masked_token_type"].reshape(
                -1
            )[token_mask]
            new_batched_data.ori_atomic_numbers = batched_data["masked_token_type"]
            new_batched_data.token_mask = token_mask
            new_batched_data.batch = (
                torch.arange(bs)
                .reshape(-1, 1)
                .repeat(1, length)
                .reshape(-1)
                .to(device)[token_mask]
                .to(device)
            )

            # 9999 means: for padding data, the pos will set a big number, thus no neighbor is included
            new_batched_data.pos = (
                batched_data["pos"].reshape(bs * length, 3).float()[token_mask]
            )

            new_batched_data.natoms = torch.sum(token_mask.reshape(bs, length), dim=1)
            new_batched_data.ptr = torch.cumsum(new_batched_data.natoms, dim=0)

            (
                edge_index,
                edge_distance,
                edge_distance_vec,
                cell_offsets,
                cell_offset_distances,
                neighbors,
            ) = generate_graph(
                new_batched_data,
                self.max_radius,
                max_neighbors=self.max_neighbors,
                use_pbc=False,
                otf_graph=True,  # oc 20 , material data, these kind of data is saved or otf mode?
                enforce_max_neighbors_strictly=True,
            )
            new_batched_data.edge_index = edge_index
            new_batched_data.pos = new_batched_data.pos.to(tensortype)
            # new_batched_data.cell = new_batched_data.cell.to(tensortype)
            new_batched_data.edge_distance = edge_distance.to(tensortype)
            new_batched_data.edge_distance_vec = edge_distance_vec.to(tensortype)
            # new_batched_data.cell_offsets = cell_offsets
            # new_batched_data.cell_offset_distances = cell_offset_distances
            # new_batched_data.neighbors = neighbors

            # remove padding number
            new_batched_data.token_embedding = token_embedding.reshape(bs * length, -1)[
                token_mask
            ]

            node_attr = torch.zeros((bs * length, self.hs), device=device)
            node_vec = torch.zeros((bs * length, 3 * self.hs), device=device)
            if edge_distance_vec.numel() == 0:
                warnings.warn(
                    f"Edge_distance_vec is empty, skip batch and return zero matrix, please check. "
                    f"token is protein? {torch.any(batched_data['is_protein']).item()}, "
                    f"periodic? {torch.any(batched_data['is_periodic']).item()}, "
                    f"molecular? {torch.any(batched_data['is_molecule']).item()}"
                )
                return node_attr.reshape(bs, length, -1), node_vec.reshape(
                    bs, length, 3, -1
                )

            _node_vec = self.node_attr_encoder(new_batched_data)

            node_attr[token_mask] = _node_vec[:, : self.hs]
            node_vec[token_mask] = _node_vec[:, self.hs : 4 * self.hs]
            return node_attr.reshape(bs, length, -1), node_vec.reshape(
                bs, length, 3, -1
            )
