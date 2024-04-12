# -*- coding: utf-8 -*-
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sfm.logging.loggers import logger
from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.models.graphormer.modules.graphormer_layers import NodeTaskHead
from sfm.models.graphormer.modules.pbc import CellExpander
from sfm.models.graphormer.modules.UnifiedDecoder import UnifiedDecoder
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
        std = self.stds.weight.float().view(-1).abs() + 1e-2
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
            expand_k = torch.gather(k, dim=0, index=outcell_index + 1)
            expand_v = torch.gather(v, dim=0, index=outcell_index + 1)

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

        if key_padding_mask is not None:
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

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights_float.type_as(attn_weights)

        attn_probs = self.dropout_module(attn_weights)

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
            expand_pos = torch.cat([pos, expand_pos], dim=1)

            expand_n_node = expand_pos.size()[1]

            delta_pos = pos.unsqueeze(2) - expand_pos.unsqueeze(
                1
            )  # B x T x (expand T) x 3
            dist = delta_pos.norm(dim=-1).view(-1, n_node, expand_n_node)
            full_mask = torch.cat([padding_mask, expand_mask], dim=-1)
            dist = dist.masked_fill(full_mask.unsqueeze(1).to(torch.bool), 1.0)
            dist = dist.masked_fill(padding_mask.unsqueeze(-1).to(torch.bool), 1.0)
        else:
            delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
            dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)

        n_graph, n_node, _ = pos.shape

        delta_pos = delta_pos / (dist.unsqueeze(-1) + 1e-2)
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
            graph_attn_bias.masked_fill_(
                full_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            graph_attn_bias.masked_fill_(
                padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), float("-inf")
            )

            edge_feature = edge_feature.masked_fill(
                full_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
            )
            edge_feature = edge_feature.masked_fill(
                padding_mask.unsqueeze(-1).unsqueeze(-1).to(torch.bool), 0.0
            )

        sum_edge_features = edge_feature.sum(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features)

        merge_edge_features = merge_edge_features.masked_fill_(
            padding_mask.unsqueeze(-1), 0.0
        )

        return graph_attn_bias, merge_edge_features, delta_pos, node_type_edge


class UnifiedDecoderNoMask(UnifiedDecoder):
    def forward(self, x, pos, node_type_edge, node_type, padding_mask):
        n_node = pos.shape[1]

        uni_delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = uni_delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        uni_delta_pos /= dist.unsqueeze(-1) + 1e-2

        # r_i/||r_i|| * gbf(||r_i||)
        pos_norm = pos.norm(dim=-1)

        uni_gbf_pos_feature = self.unified_gbf_pos(pos_norm, node_type.unsqueeze(-1))
        uni_pos_feature = uni_gbf_pos_feature.masked_fill(
            padding_mask[:, 1:].unsqueeze(-1), 0.0
        )
        uni_vec_value = self.unified_vec_proj(uni_pos_feature).unsqueeze(-2)
        vec = pos.unsqueeze(-1) * uni_vec_value

        vec = vec.masked_fill(padding_mask[:, 1:].unsqueeze(-1).unsqueeze(-1), 0.0)
        pos_mean_centered_dist = pos.norm(dim=-1)
        pos_mean_centered_unit = pos / (pos_mean_centered_dist.unsqueeze(-1) + 1e-2)

        uni_gbf_feature = self.unified_gbf_attn_bias(dist, node_type_edge)

        uni_graph_attn_bias = (
            self.unified_bias_proj(uni_gbf_feature).permute(0, 3, 1, 2).contiguous()
        )
        uni_graph_attn_bias = uni_graph_attn_bias.masked_fill(
            padding_mask[:, 1:].unsqueeze(1).unsqueeze(2), float("-inf")
        )

        output = x.contiguous().transpose(0, 1)[:, 1:, :]
        output = output.masked_fill(padding_mask[:, 1:].unsqueeze(-1), 0.0)

        for i, layer in enumerate(self.unified_encoder_layers):
            output, vec = layer(
                output,
                vec,
                uni_graph_attn_bias,
                uni_graph_attn_bias,
                uni_graph_attn_bias,
                uni_graph_attn_bias,
                padding_mask[:, 1:],
                [pos_mean_centered_unit, uni_delta_pos],
                [dist, node_type_edge],
            )

        node_output = self.unified_final_equivariant_ln(vec)
        output = self.unified_final_invariant_ln(output)

        node_output = self.unified_output_layer(node_output).squeeze(-1)
        node_output = node_output.masked_fill(padding_mask[:, 1:].unsqueeze(-1), 0.0)

        return node_output, output


class GraphormerMatterSim(Model):
    def __init__(
        self,
        cli_args,
        energy_mean,
        energy_std,
        force_mean,
        force_std,
        force_loss_factor,
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

        # graph token embedding
        self.graph_token = nn.Embedding(1, graphormer_config.embedding_dim)
        self.graph_token.weight.data.zero_()

        # graph token attn bias
        self.graph_token_virtual_distance = nn.Embedding(
            1, graphormer_config.num_attention_heads
        )
        self.graph_token_virtual_distance.weight.data.zero_()

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
        )

        self.layers = nn.ModuleList(
            [
                GraphormerEncoderLayer(graphormer_config=graphormer_config)
                for _ in range(graphormer_config.num_encoder_layers)
            ]
        )

        self.energy_loss = nn.L1Loss(reduction="mean")
        self.force_loss = nn.L1Loss(reduction="none")
        self.force_mae_loss = nn.L1Loss(reduction="none")

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
            )

        self.energy_mean = energy_mean
        self.energy_std = energy_std
        self.force_mean = force_mean
        self.force_std = force_std
        self.force_loss_factor = force_loss_factor

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

    def forward(self, batched_data):
        atomic_numbers = batched_data["x"]
        pbc = batched_data["pbc"]
        cell = batched_data["cell"]
        pos = batched_data["pos"]

        pbc_expand_batched = {}
        expand_pos, _, outcell_index, expand_mask = self.cell_expander.expand(
            pos,
            pbc,
            atomic_numbers[:, :, 0],
            cell,
            batched_data["natoms"],
        )
        pbc_expand_batched["expand_pos"] = expand_pos
        pbc_expand_batched["outcell_index"] = outcell_index
        pbc_expand_batched["expand_mask"] = expand_mask

        n_graph, n_node = atomic_numbers.size()[:2]
        padding_mask = (atomic_numbers[:, :, 0]).eq(0)  # B x T
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        x: Tensor = self.atom_encoder(atomic_numbers).sum(dim=-2)

        x = torch.cat(
            [self.graph_token.weight.unsqueeze(1).repeat([n_graph, 1, 1]), x], dim=1
        )

        (
            attn_bias_3d,
            merged_edge_features,
            delta_pos,
            node_type_edge,
        ) = self.graph_3d_pos_encoder(
            batched_data, pos, pbc_expand_batched=pbc_expand_batched
        )

        num_heads, max_len = attn_bias_3d.size()[1], attn_bias_3d.size()[2] + 1

        attn_bias = torch.zeros(
            [n_graph, num_heads, 1 + n_node, 1 + n_node],
            dtype=attn_bias_3d.dtype,
            device=attn_bias_3d.device,
        )

        t = self.graph_token_virtual_distance.weight.view(
            1, self.graphormer_config.num_attention_heads, 1
        ).repeat(1, len(self.layers) + 1, 1)
        attn_bias[:, :, 1:, 0] = t
        attn_bias[:, :, 0, :] = t

        extended_bias = torch.gather(
            attn_bias[:, :, :, 1:],
            dim=3,
            index=outcell_index.unsqueeze(1)
            .unsqueeze(2)
            .repeat(1, num_heads, max_len, 1),
        )
        attn_bias = torch.cat([attn_bias, extended_bias], dim=3)

        attn_bias[:, :, 1:, 1:] += attn_bias_3d

        attn_bias = attn_bias.reshape(
            n_graph,
            len(self.layers) + 1,
            -1,
            attn_bias.size()[-2],
            attn_bias.size()[-1],
        )

        x[:, 1:, :] = x[:, 1:, :] + merged_edge_features * 0.01

        x = self.emb_layer_norm(x)

        x = x.transpose(0, 1)

        for i, layer in enumerate(self.layers):
            x, _ = layer(
                x,
                attn_bias[:, i, :, :, :],
                self_attn_padding_mask=padding_mask,
                pbc_expand_batched=pbc_expand_batched,
            )

        outcell_index = pbc_expand_batched["outcell_index"]
        expand_pos = pbc_expand_batched["expand_pos"]
        expand_pos = torch.cat([pos, expand_pos], dim=1)
        expand_x = torch.gather(
            x.transpose(0, 1),
            1,
            outcell_index.unsqueeze(-1).repeat(1, 1, x.size()[-1]) + 1,
        )
        expand_x = torch.cat([x.transpose(0, 1), expand_x], dim=1).transpose(0, 1)
        node_type = atomic_numbers[:, :, 0]
        expand_node_type = torch.gather(node_type, 1, outcell_index)
        expand_node_type = torch.cat([node_type, expand_node_type], dim=1)
        expand_mask = pbc_expand_batched["expand_mask"]
        expand_mask = torch.cat([padding_mask, expand_mask], dim=-1)
        expand_node_type_edge = torch.gather(
            node_type_edge,
            1,
            outcell_index.unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, node_type_edge.size(-2), 1),
        )
        expand_node_type_edge = torch.cat(
            [node_type_edge, expand_node_type_edge], dim=1
        )

        if self.use_simple_head:
            forces = self.node_head(
                x[1:],
                attn_bias[:, -1, :, 1:, 1:],
                delta_pos,
                expand_mask[:, 1:],
                pbc_expand_batched,
            )
            x = x.transpose(0, 1)[:, 1:]
        else:
            forces, x = self.node_head(
                expand_x,
                expand_pos,
                expand_node_type_edge,
                expand_node_type,
                expand_mask,
            )
            x = x[:, :n_node]

        # use mean pooling
        x = torch.sum(
            x.masked_fill(padding_mask[:, 1:].unsqueeze(-1), 0.0), dim=1
        ) / batched_data["natoms"].unsqueeze(-1)
        x = self.layer_norm(self.activation_function(self.lm_head_transform_weight(x)))
        energy = self.energy_out(x)  # per atom energy, get from graph token

        forces = forces[:, :n_node]
        forces = forces.masked_fill(padding_mask[:, 1:].unsqueeze(-1), 0.0)

        if torch.any(x.isnan()) or torch.any(forces.isnan()):
            logger.info(
                f"found nan in: {torch.any(x.isnan())}, {torch.any(forces.isnan())}"
            )

        return energy, forces, padding_mask[:, 1:]

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        energy, forces, padding_mask = model_output
        bs = energy.shape[0]
        pad_seq_len = forces.size()[1]
        natoms = batch_data["natoms"]
        energy_loss = (
            self.energy_loss(
                (batch_data["y"].reshape(-1) - self.energy_mean) / self.energy_std,
                energy.reshape(-1),
            )
            * self.energy_std
        )
        force_loss = (
            self.force_loss(
                (batch_data["forces"].reshape(-1) - self.force_mean) / self.force_std,
                forces.reshape(-1),
            ).reshape(bs, pad_seq_len, 3)
            * self.force_std
        )
        force_loss = force_loss.masked_fill(padding_mask.unsqueeze(-1), 0.0).sum() / (
            natoms.sum() * 3.0
        )
        force_mae_loss = (
            self.force_mae_loss(
                (batch_data["forces"].reshape(-1).float() - self.force_mean)
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
        if torch.any(batch_data["forces"].isnan()):
            logger.warning("nan found in force labels")
        if torch.any(batch_data["y"].isnan()):
            logger.warning("nan found in energy labels")
        return ModelOutput(
            loss=(force_loss * self.force_loss_factor + energy_loss).float(),
            log_output={
                "energy_loss": energy_loss,
                "force_loss": force_loss,
                "force_mae_loss": force_mae_loss,
            },
            num_examples=bs,
        )

    def config_optimizer(self):
        pass
