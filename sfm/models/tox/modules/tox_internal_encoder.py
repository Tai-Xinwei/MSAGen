# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import torch
import torch.nn as nn

from sfm.data.prot_data.vocalubary import Alphabet
from sfm.models.tox.modules.tox_internal_encoder_layer import ToxInternalEncoderLayer
from sfm.modules.FairseqDropout import FairseqDropout
from sfm.modules.layer_norm import LayerNorm
from sfm.modules.multihead_attention import MultiheadAttention
from sfm.modules.quant_noise import quant_noise as apply_quant_noise_
from sfm.utils import LayerDropModuleList

VOCAB = Alphabet()


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


class ToxInternalResidueEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = self.check_args(args)
        self.embedding_dim = args.embedding_dim
        self.num_residues = args.num_residues
        self.restype_emb = nn.Embedding(
            self.num_residues, self.embedding_dim, padding_idx=VOCAB.padding_idx
        )

    def check_args(self, args):
        required_lst = ["num_residues", "embedding_dim"]
        for k in required_lst:
            assert hasattr(
                args, k
            ), f"args should have {k} attribute in {self.__class__.__name__}"

    def forward(self, batched_data: dict):
        x = batched_data["input"]["aa"]
        x = self.restype_emb(x)
        return x


class ToxInternalEmbedding(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = self.check_args(args)
        self.embedding_dim = args.embedding_dim
        self.num_residues = args.num_residues
        self.res_type_emb = nn.Embedding(
            self.num_residues, self.embedding_dim, padding_idx=VOCAB.padding_idx
        )
        self.bl_emb = nn.Linear(3, self.embedding_dim, bias=False)
        self.ba_emb = nn.Linear(2 * 3, self.embedding_dim, bias=False)
        self.da_emb = nn.Linear(2 * 3, self.embedding_dim, bias=False)

        self.seq_mask_emb = nn.Embedding(1, self.embedding_dim, padding_idx=None)
        self.str_mask_emb = nn.Embedding(1, self.embedding_dim, padding_idx=None)

    def check_args(self, args):
        required_lst = ["num_residues", "embedding_dim"]
        for k in required_lst:
            assert hasattr(
                args, k
            ), f"args should have {k} attribute in {self.__class__.__name__}"
        return args

    def angle2unit_circle(self, angles: torch.Tensor) -> torch.Tensor:
        # angles: [B, L] --> [B, L, 2]
        angles = self.inf2zero(angles)
        return torch.cat(
            [torch.cos(angles).unsqueeze(-1), torch.sin(angles).unsqueeze(-1)], dim=-1
        )

    def inf2zero(self, x: torch.Tensor) -> torch.Tensor:
        return x.masked_fill(torch.isinf(x), 0.0)

    def forward(self, batched_data: dict):
        seq = batched_data["crab"]["R"]
        bl_feat = torch.cat(
            [
                self.inf2zero(
                    batched_data["internal"]["bl_N_CA"].unsqueeze(-1)
                ),  # N_res
                self.inf2zero(
                    batched_data["internal"]["bl_CA_C"].unsqueeze(-1)
                ),  # N_res
                self.inf2zero(
                    batched_data["internal"]["bl_C_N"].unsqueeze(-1)
                ),  # N_res - 1
            ],
            dim=-1,
        )  # [B, L, 3]

        ba_feat = torch.cat(
            [
                self.angle2unit_circle(
                    batched_data["internal"]["ba_C_N_CA"]
                ),  # N_res - 1
                self.angle2unit_circle(batched_data["internal"]["ba_N_CA_C"]),  # N_res
                self.angle2unit_circle(
                    batched_data["internal"]["ba_CA_C_N"]
                ),  # N_res - 1
            ],
            dim=-1,
        )  # [B, L, 6]

        da_feat = torch.cat(
            [
                self.angle2unit_circle(
                    batched_data["internal"]["da_CA_C_N_CA"]
                ),  # N_res - 1
                self.angle2unit_circle(
                    batched_data["internal"]["da_C_N_CA_C"]
                ),  # N_res - 1
                self.angle2unit_circle(
                    batched_data["internal"]["da_N_CA_C_N"]
                ),  # N_res - 1
            ],
            dim=-1,
        )  # [B, L, 6]

        mask_seq, mask_str = (
            batched_data["mask"]["mask_seq"],
            batched_data["mask"]["mask_str"],
        )

        seq_feat = self.res_type_emb(seq)  # [B, L, D]
        str_feat = (
            self.bl_emb(bl_feat) + self.ba_emb(ba_feat) + self.da_emb(da_feat)
        ) / 3.0  # [B, L, D]
        seq_feat[mask_seq] = self.seq_mask_emb.weight.squeeze(0)
        str_feat[mask_str] = self.str_mask_emb.weight.squeeze(0)
        return seq_feat + str_feat
        # return seq_feat


class ToxInternalEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = self.check_args(args)
        self.dropout_layer = self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.emb_layer = ToxInternalEmbedding(args)

        if args.q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                args.q_noise,
                args.qn_block_size,
            )
        else:
            self.quant_noise = None

        if args.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=args.layerdrop)
        else:
            self.layers = nn.ModuleList()

        droppath_probs = [
            x.item()
            for x in torch.linspace(0, args.droppath_prob, args.num_encoder_layers)
        ]

        for nl in range(args.num_encoder_layers):
            self.layers.append(
                ToxInternalEncoderLayer(args=args, droppath_prob=droppath_probs[nl])
            )

    def check_args(self, args):
        required_lst = [
            "dropout",
            "q_noise",
            "qn_block_size",
            "layerdrop",
            "droppath_prob",
            "num_encoder_layers",
            "embedding_dim",
        ]
        for k in required_lst:
            assert hasattr(
                args, k
            ), f"args should have {k} attribute in {self.__class__.__name__} class."
        return args

    def forward(
        self,
        batched_data,
        padding_mask: torch.Tensor = None,
        attn_mask: Optional[torch.Tensor] = None,
        last_state_only: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention
        x = self.emb_layer(batched_data)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        x = self.dropout_module(x)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        # B x L x C -> L x B x C
        x = x.transpose(0, 1).contiguous()
        for layer in self.layers:
            x = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask)
            if not last_state_only:
                inner_states.append(x.transpose(0, 1).contiguous())
        return x.transpose(0, 1).contiguous(), inner_states
