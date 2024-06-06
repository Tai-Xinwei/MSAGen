# -*- coding: utf-8 -*-
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from sfm.models.psm.invariant.plain_encoder_layer import PSMPlainEncoderLayer
from sfm.models.psm.psm_config import PSMConfig
from sfm.modules.droppath import DropPath
from sfm.modules.FairseqDropout import FairseqDropout

# from fairseq import utils
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.mem_eff_attn import MemEffSelfAttn


class PSMPlainEncoder(nn.Module):
    """
    Implements a Transformer-M Encoder Layer.
    """

    def __init__(self, args, psm_config: PSMConfig):
        super().__init__()

        self.layers = nn.ModuleList([])

        for nl in range(psm_config.num_encoder_layers):
            self.layers.extend([PSMPlainEncoderLayer(args, psm_config)])

        # dummy param for lora, do not remove
        self.dummy = nn.Linear(1, 1, bias=False)

    # @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        batched_data: Dict,
        pbc_expand_batched: Optional[Dict] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        for layer_index, layer in enumerate(self.layers):
            x, _ = layer(
                x,
                padding_mask,
                batched_data,
                pbc_expand_batched=pbc_expand_batched,
            )
        return x
