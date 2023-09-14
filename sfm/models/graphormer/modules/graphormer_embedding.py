# -*- coding: utf-8 -*-
import torch

from megatron.core import parallel_state
from sfm.modules.sfmmodule import SFMModule

try:
    from apex.normalization import FusedLayerNorm as LayerNormTP
except:
    raise ImportError("Please install apex from install/install_megatron.sh")

from sfm.logging import logger
from sfm.modules.FairseqDropout import FairseqDropout

from .graphormer_layers import GraphAttnBias, GraphNodeFeature
from .graphormer_layers_mp import Graph3DBiasMP, GraphAttnBiasMP, GraphNodeFeatureMP


class GraphormerEmbeddingMP(SFMModule):
    def __init__(
        self,
        graphormer_config,
        args,
        mp_config,
        init_bias: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ):
        super().__init__()
        logger.info(
            "Graphormer 3D parallel version only support generalist task, do not support 3D input"
        )

        self.layerdrop = graphormer_config.layerdrop
        self.max_seq_len = graphormer_config.max_seq_len
        self.embedding_dim = graphormer_config.embedding_dim
        self.ffn_embedding_dim = graphormer_config.ffn_embedding_dim
        self.num_attention_heads = graphormer_config.num_attention_heads
        self.num_segments = graphormer_config.num_segments
        self.use_position_embeddings = graphormer_config.use_position_embeddings
        self.apply_bert_init = graphormer_config.apply_bert_init
        self.learned_pos_embedding = graphormer_config.learned_pos_embedding

        self.embed_scale = embed_scale
        self.inner_states = None
        self.args = args
        self.config = mp_config
        self.graphormer_config = graphormer_config

        self.dropout_module = FairseqDropout(
            graphormer_config.dropout, module_name=self.__class__.__name__
        )

        if graphormer_config.encoder_normalize_before:
            self.emb_layer_norm = LayerNormTP(self.embedding_dim)
        else:
            self.emb_layer_norm = None

        self.graph_node_feature = GraphNodeFeatureMP(
            mp_config,
            args,
            num_heads=graphormer_config.num_attention_heads,
            num_atoms=graphormer_config.num_atoms,
            num_in_degree=graphormer_config.num_in_degree,
            num_out_degree=graphormer_config.num_out_degree,
            hidden_dim=graphormer_config.embedding_dim,
            n_layers=graphormer_config.num_encoder_layers,
            no_2d=graphormer_config.no_2d,
        )

        self.graph_attn_bias = GraphAttnBiasMP(
            mp_config,
            args,
            num_heads=graphormer_config.num_attention_heads
            * (graphormer_config.num_encoder_layers + 1),
            num_atoms=graphormer_config.num_atoms,
            num_edges=graphormer_config.num_edges,
            num_spatial=graphormer_config.num_spatial,
            num_edge_dis=graphormer_config.num_edge_dis,
            edge_type=graphormer_config.edge_type,
            multi_hop_max_dist=graphormer_config.multi_hop_max_dist,
            hidden_dim=graphormer_config.num_attention_heads,
            n_layers=graphormer_config.num_encoder_layers,
            no_2d=graphormer_config.no_2d,
        )

        assert graphormer_config.add_3d is False, "3D bias is not supported in MP mode"
        self.graph_3d_bias = (
            Graph3DBiasMP(
                mp_config,
                args,
                num_heads=graphormer_config.num_attention_heads
                * (graphormer_config.num_encoder_layers + 1),
                num_edges=graphormer_config.num_edges,
                n_layers=graphormer_config.num_encoder_layers,
                embed_dim=graphormer_config.embedding_dim,
                num_kernel=graphormer_config.num_3d_bias_kernel,
                no_share_rpe=False,
            )
            if graphormer_config.add_3d
            else None
        )

    def forward(self, input_batchdata: tuple):
        (
            input_ids,
            llm_mask,
            _,
            x_0,
            in_degree,
            out_degree,
            attn_bias,
            spatial_pos,
            edge_input,
            num_atoms,
            pos,
            mask3d_filter,
            node_type_edge,
        ) = input_batchdata

        # assert type(idx) == torch.Tensor
        assert type(attn_bias) == torch.Tensor
        # assert type(attn_edge_type) == torch.Tensor
        assert type(spatial_pos) == torch.Tensor
        assert type(in_degree) == torch.Tensor
        # assert type(output_degree) == torch.Tensor
        assert type(x_0) == torch.Tensor
        assert type(edge_input) == torch.Tensor
        # assert type(y) == torch.Tensor
        # assert type(pos) == torch.Tensor
        # assert type(node_type_edge) == torch.Tensor

        last_state_only = False

        n_graph, n_node = x_0.size()[:2]
        padding_mask = (x_0[:, :, 0]).eq(0)  # B x T x 1
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        Bs, Seq_len = padding_mask.shape

        # calculate node feature
        input_tuple = (x_0, in_degree, out_degree, None, None)
        x = self.graph_node_feature(input_tuple)

        # x: B x T x C
        # calculate the 2D bias
        input_tuple2 = (
            attn_bias,
            spatial_pos,
            x_0,
            edge_input,
            None,
            None,
            None,
        )

        attn_bias = self.graph_attn_bias(
            input_tuple2
        )  # B x (nhead x (nlayer+1)) x T x T

        # tp 3d bias is not implemented, it's not needed in the generalist task
        if self.graph_3d_bias is not None:
            input_tuple3 = (pos, x_0, node_type_edge, None)
            attn_bias_3d, merged_edge_features, delta_pos = self.graph_3d_bias(
                input_tuple3
            )
            attn_bias_3d = attn_bias_3d.masked_fill_(
                mask3d_filter[:, None, None, None], 0.0
            )
            merged_edge_features = merged_edge_features.masked_fill_(
                mask3d_filter[:, None, None], 0.0
            )
            attn_bias[:, :, 1:, 1:] = attn_bias[:, :, 1:, 1:] + attn_bias_3d
            x[:, 1:, :] = x[:, 1:, :] + merged_edge_features * 0.01

        # add embed_scale, drop, and layer norm
        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if not last_state_only:
            self.inner_states = x[None, ...]

        attn_bias = (
            attn_bias.contiguous()
            .view(n_graph, self.args.encoder_layers + 1, -1, n_node + 1, n_node + 1)
            .contiguous()
        ).to(
            x.dtype
        )  # B x (nlayer+1) x nhead_pre_tprank x T x T

        assert (
            self.args.encoder_attention_heads % self.config.tensor_model_parallel_size
            == 0
        ), f"Force TP size be divisible by nhead to avoid low efficiency, but got {self.args.encoder_attention_heads} and {self.config.tensor_model_parallel_size}"
        nhead_per_TP_rank = (
            self.args.encoder_attention_heads // self.config.tensor_model_parallel_size
        )

        assert list(attn_bias.size()) == [
            Bs,
            self.graphormer_config.num_encoder_layers + 1,
            nhead_per_TP_rank,
            Seq_len,
            Seq_len,
        ], f"attn_bias size is {attn_bias.size()}, but expected to be [{Bs}, {self.graphormer_config.num_encoder_layers+1}, {nhead_per_TP_rank}, {Seq_len}, {Seq_len}]"

        return x, padding_mask, attn_bias, input_ids, llm_mask

        # patition the attn_bias to each TP rank
        # tp_rank = parallel_state.get_tensor_model_parallel_rank()
        # attn_bias_tp = attn_bias[
        #     :, :, tp_rank * nhead_per_TP_rank : (tp_rank + 1) * nhead_per_TP_rank, :, :
        # ]
        #
        # return x, padding_mask, attn_bias_tp, input_ids, llm_mask
