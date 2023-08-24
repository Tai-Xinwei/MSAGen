# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sfm.models.graphormer.modules.graphormer_layers import NodeTaskHead
from sfm.modules.FairseqDropout import FairseqDropout
from sfm.modules.layer_norm import LayerNorm
from sfm.modules.multihead_attention import MultiheadAttention
from sfm.modules.quant_noise import quant_noise as apply_quant_noise_
from sfm.utils import LayerDropModuleList

from .pfm_embedding import PFMEmbedding
from .pfm_encoder_layer import PFMEncoderLayer


def init_params(module):
    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class PFMEncoder(nn.Module):
    def __init__(
        self,
        pfm_config,
        init_bias: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        # num_pred_attn_layer: int = 4,
    ) -> None:
        super().__init__()
        args = pfm_config.args
        self.pfm_config = pfm_config
        self.dropout_module = FairseqDropout(
            pfm_config.dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = pfm_config.layerdrop
        self.max_seq_len = pfm_config.max_seq_len
        self.embedding_dim = pfm_config.embedding_dim
        self.ffn_embedding_dim = pfm_config.ffn_embedding_dim
        self.num_attention_heads = pfm_config.num_attention_heads
        self.num_segments = pfm_config.num_segments
        self.use_position_embeddings = pfm_config.use_position_embeddings
        self.apply_bert_init = pfm_config.apply_bert_init
        self.learned_pos_embedding = pfm_config.learned_pos_embedding

        if init_bias:
            self.pfm_emb = PFMEmbedding(pfm_config)

        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        if pfm_config.encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        droppath_probs = [
            x.item()
            for x in torch.linspace(
                0, pfm_config.droppath_prob, pfm_config.num_encoder_layers
            )
        ]

        for nl in range(pfm_config.num_encoder_layers):
            self.layers.extend(
                [
                    self.build_transformer_sentence_encoder_layer(
                        embedding_dim=pfm_config.embedding_dim,
                        ffn_embedding_dim=pfm_config.ffn_embedding_dim,
                        num_attention_heads=pfm_config.num_attention_heads,
                        dropout=self.dropout_module.p,
                        attention_dropout=pfm_config.attention_dropout,
                        activation_dropout=pfm_config.activation_dropout,
                        activation_fn=pfm_config.activation_fn,
                        export=export,
                        q_noise=q_noise,
                        qn_block_size=qn_block_size,
                        sandwich_ln=pfm_config.sandwich_ln,
                        droppath_prob=droppath_probs[nl],
                        nl=nl,
                        args=args,
                        pfm_config=pfm_config,
                    )
                ]
            )

        init_scale = math.sqrt(math.log(pfm_config.num_encoder_layers))
        for name, p in self.named_parameters():
            if "fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name:
                p.data.mul_(init_scale)

        self.args = args

        if pfm_config.transformer_m_pretrain:
            try:
                mode_prob = [float(item) for item in pfm_config.mode_prob.split(",")]
                assert len(mode_prob) == 2
                assert sum(mode_prob) == 1.0
            except:
                mode_prob = [0.5, 0.5]
            self.mode_prob = mode_prob

    def build_transformer_sentence_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        sandwich_ln,
        droppath_prob,
        nl,
        args,
        pfm_config=None,
    ):
        return PFMEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            sandwich_ln=sandwich_ln,
            droppath_prob=droppath_prob,
            nl=nl,
            args=args,
            pfm_config=pfm_config,
        )

    def forward(
        self,
        batched_data,
        perturb=None,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention

        residue_seq = batched_data["x"]
        n_graph, n_node = residue_seq.size()[:2]

        mask_aa = batched_data["masked_aa"]
        mask_pos = batched_data["mask_pos"]
        if self.pfm_config.transformer_m_pretrain:
            # 0: independent mask_aa and mask_pos
            # 1: mask_aa and mask_pos are the same
            mask_choice = np.random.choice(np.arange(2), n_graph, p=[1.0 / 2, 1.0 / 2])
            mask = torch.tensor([i for i in mask_choice]).to(batched_data["pos"])
            mask = (
                mask.unsqueeze(1).unsqueeze(-1).repeat(1, n_node, 1)
            )  # [ngraph, nnode+1]
            mask_pos = torch.where(mask == 1, mask_aa, mask_pos)

        # add noise to pos with mask_pos==true
        ori_pos = batched_data["pos"]
        noise = (
            torch.randn(ori_pos.shape, device=ori_pos.device) * self.args.noise_scale
        )
        noise = noise.masked_fill_(mask_pos.bool(), 0.0)
        pos = ori_pos + noise

        padding_mask = (residue_seq[:, :]).eq(0)  # B x T x 1

        x, edge_feature, delta_pos = self.pfm_emb(
            batched_data,
            padding_mask,
            pos=pos,
            mask_aa=mask_aa,
            mask_pos=mask_pos,
        )

        if perturb is not None:
            x[:, :, :] = x[:, :, :] + perturb

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for _, layer in enumerate(self.layers):
            x, attn_bias = layer(
                x,
                edge_feature=edge_feature,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                mask_pos=mask_pos,
            )
            if not last_state_only:
                inner_states.append(x)

        if attn_bias is not None:
            attn_bias = (
                attn_bias.contiguous()
                .view(n_graph, self.num_attention_heads, n_node, n_node)
                .contiguous()
            )

        return x, attn_bias, delta_pos, pos, inner_states, padding_mask, mask_pos


class NodeDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        num_attention_heads: int = 8,
        last_state_only: bool = True,
        args=None,
    ):
        super().__init__()
        if not args.ft:
            self.node_proc = NodeTaskHead(embedding_dim, num_attention_heads)
        self.args = args
        self.last_state_only = last_state_only

    def forward(self, x, attn_bias, delta_pos, inner_states):
        sentence_rep = x[0, :, :]

        node_output = None
        if delta_pos is not None and not self.args.ft:
            node_output = self.node_proc(
                x[1:, :, :], attn_bias[:, :, 1:, 1:], delta_pos
            )

        if self.last_state_only:
            inner_states = [x]

        if not self.last_state_only:
            return torch.stack(inner_states), node_output, sentence_rep
        else:
            return inner_states, node_output, sentence_rep
