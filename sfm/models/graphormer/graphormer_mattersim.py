# -*- coding: utf-8 -*-
import math
from typing import Dict, Optional, Tuple

import e3nn
import torch
import torch.nn.functional as F
from icecream import ic
from torch import Tensor, nn

from sfm.logging.loggers import logger
from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.models.graphormer.modules.graphormer_layers import NodeTaskHead
from sfm.models.graphormer.modules.pbc import CellExpander
from sfm.models.graphormer.modules.UnifiedDecoder_backprop import UnifiedDecoder
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=512 * 3):
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
        std = self.stds.weight.float().view(-1).abs() + 3e-1
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        self_attention=False,
        d_tilde=1,
        k_bias=False,
        q_bias=True,
        v_bias=True,
        o_bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (
            (self.head_dim / d_tilde) ** 0.5
        ) / self.head_dim  # when d_tilt == 1, match with original transformer scale

        self.self_attention = self_attention

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=k_bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=q_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=v_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=o_bias)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.reset_parameters(d_tilde)

    def reset_parameters(self, d_tilde=1):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(
                self.k_proj.weight, gain=1.0 / (math.sqrt(2 * d_tilde))
            )
            nn.init.xavier_uniform_(
                self.v_proj.weight, gain=1.0 / (math.sqrt(2 * d_tilde))
            )
            nn.init.xavier_uniform_(
                self.q_proj.weight, gain=1.0 / (math.sqrt(2 * d_tilde))
            )
            if self.q_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)
            if self.k_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)
            if self.v_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)
        else:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0 / math.sqrt(d_tilde))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0 / math.sqrt(d_tilde))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))

        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.layer_norm is not None:
            self.layer_norm.reset_parameters()

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        pbc_expand_batched: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()

        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        # ic(q, k, v, self.v_proj.weight.data, self.v_proj.bias.data)

        q *= self.scaling

        if pbc_expand_batched is not None:
            outcell_index = pbc_expand_batched["outcell_index"]
            expand_mask = pbc_expand_batched["expand_mask"]
        else:
            outcell_index = None
            expand_mask = None

        if outcell_index is not None:
            outcell_index = (
                outcell_index.transpose(1, 0).unsqueeze(-1).expand(-1, -1, embed_dim)
            )
            expand_k = torch.gather(k, dim=0, index=outcell_index)
            expand_v = torch.gather(v, dim=0, index=outcell_index)

            k = torch.cat([k, expand_k], dim=0)
            v = torch.cat([v, expand_v], dim=0)

            src_len = k.size()[0]

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

            if outcell_index is not None:
                assert expand_mask is not None
                key_padding_mask = torch.cat([key_padding_mask, expand_mask], dim=1)
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.contiguous().view(
                bsz * self.num_heads, tgt_len, src_len
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = (attn_weights + 20.0) * attn_mask.unsqueeze(1) - 20.0
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights_float.type_as(attn_weights)

        if attn_mask is not None:
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                * attn_mask.unsqueeze(1)
            ).view(bsz * self.num_heads, tgt_len, src_len)
            # attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

        attn_probs = self.dropout_module(attn_weights).type_as(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        if self.layer_norm is not None:
            attn = self.layer_norm(attn)

        attn = self.out_proj(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights


class GraphormerEncoderLayer(nn.Module):
    """
    Implements a Graphormer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(self, graphormer_config: GraphormerConfig) -> None:
        super().__init__()
        self.dropout_module = nn.Dropout(graphormer_config.dropout)
        self.attention_dropout_module = nn.Dropout(graphormer_config.attention_dropout)
        self.activation_dropout_module = nn.Dropout(
            graphormer_config.activation_dropout
        )

        # Initialize blocks
        self.self_attn = MultiheadAttention(
            graphormer_config.embedding_dim,
            graphormer_config.num_attention_heads,
            dropout=graphormer_config.attention_dropout,
            self_attention=True,
            d_tilde=1,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(graphormer_config.embedding_dim)

        self.fc1 = nn.Linear(
            graphormer_config.embedding_dim, graphormer_config.ffn_embedding_dim
        )
        self.fc2 = nn.Linear(
            graphormer_config.ffn_embedding_dim, graphormer_config.embedding_dim
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(graphormer_config.embedding_dim)
        self.final_layer_norm_2 = nn.LayerNorm(graphormer_config.ffn_embedding_dim)

        self.activation_function = nn.GELU()

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.self_attn_layer_norm.reset_parameters()
        self.final_layer_norm.reset_parameters()
        self.final_layer_norm_2.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        pbc_expand_batched: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """

        # x: T x B x C
        residual = x
        x = self.self_attn_layer_norm(x)

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
            pbc_expand_batched=pbc_expand_batched,
        )

        x = self.dropout_module(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_function(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.final_layer_norm_2(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        return x, attn


class Graph3DBias(nn.Module):
    """
    Compute 3D attention bias according to the position information for each head.
    """

    def __init__(self, graphormer_config: GraphormerConfig):
        super(Graph3DBias, self).__init__()
        num_rpe_heads = graphormer_config.num_attention_heads * (
            graphormer_config.num_encoder_layers + 1
        )
        self.gbf = GaussianLayer(
            graphormer_config.num_3d_bias_kernel, graphormer_config.num_edges
        )
        self.gbf_proj = NonLinear(graphormer_config.num_3d_bias_kernel, num_rpe_heads)
        self.edge_proj = nn.Linear(
            graphormer_config.num_3d_bias_kernel, graphormer_config.embedding_dim
        )
        self.pbc_multigraph_cutoff = graphormer_config.pbc_multigraph_cutoff

    def polynomial(self, dist: torch.Tensor, cutoff: float) -> torch.Tensor:
        """
        Polynomial cutoff function,ref: https://arxiv.org/abs/2204.13639
        Args:
            dist (tf.Tensor): distance tensor
            cutoff (float): cutoff distance
        Returns: polynomial cutoff functions
        """
        ratio = torch.div(dist, cutoff)
        result = (
            1
            - 6 * torch.pow(ratio, 5)
            + 15 * torch.pow(ratio, 4)
            - 10 * torch.pow(ratio, 3)
        )
        return torch.clamp(result, min=0.0)

    def forward(self, batched_data, pos, pbc_expand_batched=None):
        x, node_type_edge = (
            batched_data["x"],
            batched_data["node_type_edge"],
        )  # pos shape: [n_graphs, n_nodes, 3]

        padding_mask = x.eq(0).all(dim=-1)
        n_node = pos.size()[1]

        if pbc_expand_batched is not None:
            expand_pos = pbc_expand_batched["expand_pos"]
            expand_mask = pbc_expand_batched["expand_mask"]

            expand_pos = expand_pos.masked_fill(
                expand_mask.unsqueeze(-1).to(torch.bool), 0.0
            )
            total_pos = torch.cat([pos, expand_pos], dim=1)

            expand_n_node = total_pos.size()[1]

            delta_pos = pos.unsqueeze(2) - total_pos.unsqueeze(
                1
            )  # B x T x (expand T) x 3
            dist = delta_pos.norm(dim=-1).view(-1, n_node, expand_n_node)
            full_mask = torch.cat([padding_mask, expand_mask], dim=-1)
            dist = dist.masked_fill(full_mask.unsqueeze(1).to(torch.bool), 1.0)
            dist = dist.masked_fill(padding_mask.unsqueeze(-1).to(torch.bool), 1.0)
            attn_mask = pbc_expand_batched["local_attention_mask"]
        else:
            delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
            dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
            attn_mask = self.polynomial(dist, self.pbc_multigraph_cutoff)

        n_graph, n_node, _ = pos.shape

        delta_pos = delta_pos / (dist.unsqueeze(-1) + 1e-2)

        # attn_mask = self.polynomial(dist, self.pbc_multigraph_cutoff)

        atomic_numbers = x[:, :, 0]
        if node_type_edge is None:
            node_type_edge = atomic_numbers.unsqueeze(
                -1
            ) * 128 + atomic_numbers.unsqueeze(1)

            if pbc_expand_batched is not None:
                outcell_index = pbc_expand_batched["outcell_index"]
                node_type_edge = torch.cat(
                    [
                        node_type_edge,
                        torch.gather(
                            node_type_edge,
                            dim=-1,
                            index=outcell_index.unsqueeze(1).repeat(1, n_node, 1),
                        ),
                    ],
                    dim=-1,
                )
            node_type_edge = node_type_edge.unsqueeze(-1)
        edge_feature = self.gbf(
            dist,
            node_type_edge.long(),
        )
        gbf_result = self.gbf_proj(edge_feature)
        graph_attn_bias = gbf_result

        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()

        if pbc_expand_batched is None:
            graph_attn_bias.masked_fill_(
                padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

            edge_feature = edge_feature.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
            )
        else:
            edge_feature = edge_feature.masked_fill(
                full_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
            )
            edge_feature = edge_feature.masked_fill(
                padding_mask.unsqueeze(-1).unsqueeze(-1).to(torch.bool), 0.0
            )

            edge_feature = (
                torch.mul(edge_feature, attn_mask.unsqueeze(-1))
                .float()
                .type_as(edge_feature)
            )

        sum_edge_features = edge_feature.sum(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features)

        merge_edge_features = merge_edge_features.masked_fill_(
            padding_mask.unsqueeze(-1), 0.0
        )

        pbc_expand_batched["expand_pos"] = expand_pos
        pbc_expand_batched["local_attention_mask"] = attn_mask
        return graph_attn_bias, merge_edge_features, delta_pos, node_type_edge


class UnifiedDecoderNoMask(UnifiedDecoder):
    def __init__(
        self,
        args,
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
        pbc_multigraph_cutoff: float,
    ):
        super().__init__(
            args,
            num_pred_attn_layer,
            embedding_dim,
            num_attention_heads,
            ffn_embedding_dim,
            dropout,
            attention_dropout,
            activation_dropout,
            num_3d_bias_kernel,
            num_edges,
            num_atoms,
            pbc_multigraph_cutoff,
        )
        self.unified_gbf_vec = GaussianLayer(num_3d_bias_kernel, num_edges)
        del self.unified_gbf_pos
        self.unified_gbf_pos = None

        del self.unified_gbf_attn_bias
        self.unified_gbf_attn_bias = GaussianLayer(num_3d_bias_kernel, num_edges)
        self.embedding_dim = embedding_dim
        self.edge_irreps = e3nn.o3.Irreps("1x0e+1x1o")
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.sph_harm = e3nn.o3.SphericalHarmonics(
            self.edge_irreps, normalize=True, normalization="component"
        )

    def polynomial(self, dist: torch.Tensor, cutoff: float) -> torch.Tensor:
        ratio = torch.div(dist, cutoff)
        result = (
            1
            - 6 * torch.pow(ratio, 5)
            + 15 * torch.pow(ratio, 4)
            - 10 * torch.pow(ratio, 3)
        )
        return torch.clamp(result, min=0.0)

    def forward(
        self,
        x,
        pos,
        node_type_edge,
        node_type,
        padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
    ):
        n_node = pos.shape[1]

        x = x.contiguous().transpose(0, 1)

        if pbc_expand_batched is not None:
            pbc_expand_batched["outcell_index"]
            expand_pos = torch.cat([pos, pbc_expand_batched["expand_pos"]], dim=1)
            expand_n_node = expand_pos.shape[1]
            uni_delta_pos = pos.unsqueeze(2) - expand_pos.unsqueeze(1)
            dist = uni_delta_pos.norm(dim=-1).view(-1, n_node, expand_n_node)
        else:
            uni_delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
            dist = uni_delta_pos.norm(dim=-1).view(-1, n_node, n_node)

        # attn_mask = self.polynomial(dist, self.pbc_multigraph_cutoff)
        # uni_delta_pos = uni_delta_pos / (dist.unsqueeze(-1) + 1e-5)

        # r_ij/||r_ij|| * gbf(||r_ij||)
        vec_gbf = self.unified_gbf_vec(
            dist, node_type_edge.long()
        )  # n_graph x n_node x expand_n_node x num_kernel
        # vec_gbf = torch.mul(torch.ones_like(vec_gbf,device=vec_gbf.device),dist.unsqueeze(-1))
        sph = self.sph_harm(uni_delta_pos)
        vec_gbf = sph.unsqueeze(-1) * vec_gbf.unsqueeze(
            -2
        )  # n_graph x n_node x expand_n_node x 4 x num_kernel

        # reduce ij -> i by \sum_j vec_ij * x_j
        if pbc_expand_batched is not None:
            expand_mask = pbc_expand_batched["expand_mask"]
            expand_mask = torch.cat([padding_mask, expand_mask], dim=-1)
            attn_mask = pbc_expand_batched["local_attention_mask"]
            vec_gbf = vec_gbf.masked_fill(
                expand_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0
            )
            vec_gbf = vec_gbf * attn_mask.unsqueeze(-1).unsqueeze(-1)
        else:
            attn_mask = self.polynomial(dist, self.pbc_multigraph_cutoff)
            vec_gbf = vec_gbf.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0
            )

        vec_gbf = vec_gbf.sum(dim=2).float().type_as(x)
        vec = self.unified_vec_proj(vec_gbf)
        # vec = vec_gbf_feature * x.unsqueeze(-2)

        pos_mean_centered_dist = pos.norm(dim=-1)
        pos_mean_centered_unit = pos / (pos_mean_centered_dist.unsqueeze(-1) + 1e-2)

        uni_gbf_feature = self.unified_gbf_attn_bias(dist, node_type_edge.long())

        uni_graph_attn_bias = (
            self.unified_bias_proj(uni_gbf_feature).permute(0, 3, 1, 2).contiguous()
        )

        if pbc_expand_batched is not None:
            attn_mask = attn_mask.masked_fill(
                expand_mask.unsqueeze(1), 0.0
            )  # other nodes don't attend to padding nodes
        else:
            attn_mask = attn_mask.masked_fill(
                padding_mask.unsqueeze(1), 0.0
            )  # other nodes don't attend to padding nodes

        output = x
        output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        for i, layer in enumerate(self.unified_encoder_layers):
            output, vec = layer(
                output,
                vec,
                uni_graph_attn_bias,
                uni_graph_attn_bias,
                uni_graph_attn_bias,
                uni_graph_attn_bias,
                attn_mask,
                [pos_mean_centered_unit, uni_delta_pos],
                [dist, node_type_edge],
                pbc_expand_batched=pbc_expand_batched,
            )

        node_output = self.unified_final_equivariant_ln(vec)
        output = self.unified_final_invariant_ln(output)

        node_output = self.unified_output_layer(node_output).squeeze(-1)
        node_output = node_output.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return node_output, output


class GradientHead(nn.Module):
    def __init__(self, force_std, stress_std):
        super(GradientHead, self).__init__()
        self.force_std = force_std
        self.stress_std = stress_std

    def forward(self, energy, pos, strain, volume):
        grad_outputs = [torch.ones_like(energy)]

        grad = torch.autograd.grad(
            outputs=[energy],
            inputs=[pos, strain],
            grad_outputs=grad_outputs,
            create_graph=self.training,
        )

        force_grad = grad[0] / self.force_std
        stress_grad = grad[1] / self.stress_std

        if force_grad is not None:
            forces = torch.neg(force_grad)

        if stress_grad is not None:
            stresses = 1 / volume[:, None, None] * stress_grad * 160.21766208
        return forces, stresses


class GraphormerMatterSim(Model):
    def __init__(
        self,
        cli_args,
        energy_mean,
        energy_std,
        force_mean,
        force_std,
        force_loss_factor,
        stress_mean,
        stress_std,
        stress_loss_factor,
        use_stress_loss,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        graphormer_config: GraphormerConfig = GraphormerConfig(cli_args)
        # atom embedding
        self.atom_encoder = nn.Embedding(
            graphormer_config.num_atoms + 1,
            graphormer_config.embedding_dim,
            padding_idx=0,
        )

        # GBF encoder
        self.graph_3d_pos_encoder = Graph3DBias(graphormer_config=graphormer_config)

        self.edge_proj = nn.Linear(
            graphormer_config.num_3d_bias_kernel, graphormer_config.embedding_dim
        )

        self.emb_layer_norm = nn.LayerNorm(graphormer_config.embedding_dim)

        self.cell_expander = CellExpander(
            graphormer_config.pbc_cutoff,
            graphormer_config.pbc_expanded_token_cutoff,
            graphormer_config.pbc_expanded_num_cell_per_direction,
            graphormer_config.pbc_multigraph_cutoff,
            backprop=True,
            original_token_count=False,
        )

        self.layers = nn.ModuleList(
            [
                GraphormerEncoderLayer(graphormer_config=graphormer_config)
                for _ in range(graphormer_config.num_encoder_layers)
            ]
        )

        self.gradient_head = GradientHead(force_std, stress_std)

        self.energy_loss = nn.L1Loss(reduction="mean")
        self.force_loss = nn.L1Loss(reduction="none")
        self.force_mae_loss = nn.L1Loss(reduction="none")
        self.stress_loss = nn.L1Loss(reduction="none")

        self.energy_huberloss = nn.HuberLoss(reduction="mean", delta=0.01)
        self.force_huberloss = nn.HuberLoss(reduction="none", delta=0.01)
        self.stress_huberloss = nn.HuberLoss(reduction="mean", delta=0.01)

        if cli_args.use_simple_head:
            self.node_head = NodeTaskHead(
                graphormer_config.embedding_dim, graphormer_config.num_attention_heads
            )
        else:
            self.node_head = UnifiedDecoderNoMask(
                cli_args,
                graphormer_config.num_pred_attn_layer,
                graphormer_config.embedding_dim,
                graphormer_config.num_attention_heads,
                graphormer_config.ffn_embedding_dim,
                graphormer_config.dropout,
                graphormer_config.attention_dropout,
                graphormer_config.activation_dropout,
                graphormer_config.num_3d_bias_kernel,
                graphormer_config.num_edges,
                graphormer_config.num_atoms,
                graphormer_config.pbc_multigraph_cutoff,
            )

        self.energy_mean = energy_mean
        self.energy_std = energy_std
        self.force_mean = force_mean
        self.force_std = force_std
        self.force_loss_factor = force_loss_factor
        self.stress_mean = stress_mean
        self.stress_std = stress_std
        self.stress_loss_factor = stress_loss_factor

        self.activation_function = nn.GELU()
        self.layer_norm = nn.LayerNorm(graphormer_config.embedding_dim)
        self.lm_head_transform_weight = nn.Linear(
            graphormer_config.embedding_dim, graphormer_config.embedding_dim
        )
        self.energy_out = nn.Linear(graphormer_config.embedding_dim, 1)

        if cli_args.loadcheck_path != "":
            self.load_state_dict(torch.load(cli_args.loadcheck_path)["model"])

        self.use_simple_head = cli_args.use_simple_head
        self.graphormer_config = graphormer_config
        self.use_stress_loss = use_stress_loss

    def forward(self, batched_data):
        atomic_numbers = batched_data["x"]
        pbc = batched_data["pbc"]
        cell = batched_data["cell"]
        pos = batched_data["pos"]

        padding_mask = (atomic_numbers[:, :, 0]).eq(0)  # B x T
        n_graph, n_node = atomic_numbers.size()[:2]
        # stress
        if self.use_stress_loss:
            volume = torch.abs(torch.linalg.det(cell))  # to avoid negative volume

        pbc_expand_batched = {}
        pbc_dict = self.cell_expander.expand(
            pos,
            pbc,
            atomic_numbers[:, :, 0],
            cell,
            batched_data["natoms"],
            use_local_attention=True,
        )
        pos = pbc_dict["pos"]
        # print(pos.shape)
        cell = pbc_dict["cell"]
        expand_pos = pbc_dict["expand_pos"]
        # print(expand_pos.shape)
        outcell_index = pbc_dict["outcell_index"]
        expand_mask = pbc_dict["expand_mask"]
        local_attention_mask = pbc_dict["local_attention_weight"]
        strain = pbc_dict["strain"]

        pbc_expand_batched["expand_pos"] = expand_pos
        pbc_expand_batched["outcell_index"] = outcell_index
        pbc_expand_batched["expand_mask"] = expand_mask

        x: Tensor = self.atom_encoder(atomic_numbers).sum(dim=-2)

        local_attention_mask = local_attention_mask.masked_fill(
            padding_mask.unsqueeze(-1), 1.0
        )
        expand_mask = torch.cat([padding_mask, expand_mask], dim=-1)
        local_attention_mask = local_attention_mask.masked_fill(
            expand_mask.unsqueeze(1), 0.0
        ).type_as(x)

        pbc_expand_batched["local_attention_mask"] = local_attention_mask

        (
            attn_bias,
            merged_edge_features,
            delta_pos,
            node_type_edge,
        ) = self.graph_3d_pos_encoder(
            batched_data, pos, pbc_expand_batched=pbc_expand_batched
        )
        attn_mask = local_attention_mask

        attn_bias = attn_bias.reshape(
            n_graph,
            len(self.layers) + 1,
            -1,
            attn_bias.size()[-2],
            attn_bias.size()[-1],
        ).type_as(x)

        x = x + merged_edge_features * 0.01

        x = self.emb_layer_norm(x)

        x = x.transpose(0, 1)

        for i, layer in enumerate(self.layers):
            x, _ = layer(
                x,
                attn_bias[:, i, :, :, :],
                self_attn_mask=attn_mask,
                self_attn_padding_mask=padding_mask,
                pbc_expand_batched=pbc_expand_batched,
            )

        node_type = atomic_numbers[:, :, 0]
        pbc_expand_batched["local_attention_mask"] = attn_mask

        if self.use_simple_head:
            forces = self.node_head(
                x,
                attn_bias[:, -1, :, :, :],
                delta_pos,
                # expand_mask,
                pbc_expand_batched,
            )
            x = x.transpose(0, 1)
        else:
            forces, x = self.node_head(
                x,
                pos,
                node_type_edge,
                node_type,
                padding_mask,
                pbc_expand_batched,
            )

        # use mean pooling
        x = torch.sum(
            x.masked_fill(padding_mask.unsqueeze(-1), 0.0), dim=1
        ) / batched_data["natoms"].unsqueeze(-1)
        x = self.layer_norm(self.activation_function(self.lm_head_transform_weight(x)))
        energy = self.energy_out(x)  # per atom energy, get from graph token

        # # additional stress:use energy gradient
        if self.use_stress_loss:
            grad_forces, grad_stresses = self.gradient_head(
                torch.mul(
                    energy * self.energy_std + self.energy_mean,
                    batched_data["natoms"].unsqueeze(-1),
                ),
                pos,
                strain,
                volume,
            )

        if self.use_simple_head:
            forces = grad_forces
        else:
            forces = grad_forces
            stresses = grad_stresses
            forces = forces[:, :n_node]
            forces = forces.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if torch.any(x.isnan()) or torch.any(forces.isnan()):
            logger.info(
                f"found nan in: {torch.any(x.isnan())}, {torch.any(forces.isnan())}"
            )

        if self.use_stress_loss:
            if torch.any(stresses.isnan()):
                logger.info(f"found nan in stress: {torch.any(stresses.isnan())}")
            return energy, forces, stresses, padding_mask

        return energy, forces, padding_mask

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        if self.use_stress_loss:
            energy, forces, stress, padding_mask = model_output
        else:
            energy, forces, padding_mask = model_output
        bs = energy.shape[0]
        pad_seq_len = forces.size()[1]
        natoms = batch_data["natoms"]

        # pred_energy = energy * self.energy_std + self.energy_mean
        # pred_forces = forces * self.force_std
        # pred_stress = stress * self.stress_std
        energy_loss = (
            self.energy_loss(
                (batch_data["y"].reshape(-1) - self.energy_mean) / self.energy_std,
                energy.reshape(-1),
            )
            * self.energy_std
        )
        # energy_loss = (
        #         self.energy_huberloss(
        #         batch_data["y"].reshape(-1).type_as(energy),
        #         pred_energy.reshape(-1),
        #     )
        # )
        energy_mae_loss = (
            self.energy_loss(
                (batch_data["y"].reshape(-1) - self.energy_mean) / self.energy_std,
                energy.reshape(-1),
            )
            * self.energy_std
        )
        # batch_data["forces"] = forces.detach()*2
        force_loss = (
            self.force_loss(
                (batch_data["forces"].reshape(-1) - self.force_mean) / self.force_std,
                forces.reshape(-1),
            ).reshape(bs, pad_seq_len, 3)
            * self.force_std
        )
        # force_loss = (
        #         self.force_huberloss(
        #         batch_data["forces"].type_as(forces),
        #         pred_forces,
        #     )
        # )
        force_loss = force_loss.masked_fill(padding_mask.unsqueeze(-1), 0.0).sum() / (
            natoms.sum() * 3.0
        )
        force_mae_loss = (
            self.force_mae_loss(
                (
                    batch_data["forces"].reshape(-1).float().type_as(energy)
                    - self.force_mean
                )
                / self.force_std,
                forces.reshape(-1),
            ).reshape(bs, pad_seq_len, 3)
            * self.force_std
        )
        force_mae_loss = (
            force_mae_loss.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            .norm(dim=-1)
            .sum()
            / natoms.sum()
        )

        if self.use_stress_loss:
            stress_mae_loss = (
                self.stress_loss(
                    (batch_data["stress"].reshape(-1)) / self.stress_std,
                    stress.reshape(-1),
                ).reshape(bs, 3, 3)
                * self.stress_std
            )
            stress_mae_loss = stress_mae_loss.norm(dim=-1).sum(dim=-1) / natoms
            stress_loss = (
                self.stress_loss(
                    (batch_data["stress"].reshape(-1)) / self.stress_std,
                    stress.reshape(-1),
                ).reshape(bs, 3, 3)
                * self.stress_std
            )
            stress_loss = stress_loss.sum() / (bs * 9.0)
            # stress_loss = (
            #         self.stress_huberloss(
            #         batch_data["stress"].type_as(stress),
            #         pred_stress,
            #     )
            # )
            stress_mae_loss = stress_mae_loss.sum() / bs
            # stress_loss = self.stress_loss(stress * self.stress_std, batch_data["stress"])
            # stress_mae_loss = nn.L1Loss()(stress * self.stress_std, batch_data["stress"])

        if torch.any(batch_data["forces"].isnan()):
            logger.warning("nan found in force labels")
        if torch.any(batch_data["y"].isnan()):
            logger.warning("nan found in energy labels")
        stress_non_flag = False
        if self.use_stress_loss and torch.any(batch_data["stress"].isnan()):
            logger.warning("nan found in stress labels")
            stress_non_flag = True
            stress_loss = 0.0
            stress_mae_loss = 0.0
        return ModelOutput(
            loss=(force_loss * self.force_loss_factor + energy_loss)
            .float()
            .type_as(energy)
            if not self.use_stress_loss or stress_non_flag
            else (
                force_loss * self.force_loss_factor
                + energy_loss
                + stress_loss * self.stress_loss_factor
            )
            .float()
            .type_as(energy),
            log_output={
                "energy_loss": energy_loss,
                "energy_mae_loss": energy_mae_loss,
                "force_loss": force_loss,
                "force_mae_loss": force_mae_loss,
                "stress_loss": stress_loss if self.use_stress_loss else 0,
                "stress_mae_loss": stress_mae_loss if self.use_stress_loss else 0,
            },
            num_examples=bs,
        )

    def config_optimizer(self):
        pass
