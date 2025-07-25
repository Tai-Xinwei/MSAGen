# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import torch

from sfm.models.pfm.modules.pfm_encoder import PFMEncoder
from sfm.models.pfm.pfm_config import PFMConfig


class PFMEncoderPP(PFMEncoder):
    """
    Pipeline parallel mode for PFMEncoder

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        pfm_config: PFMConfig,
    ) -> None:
        # inheret from PFMEncoder
        super().__init__(pfm_config)

    @classmethod
    def config(cls):
        return cls.pfm_config

    def forward(self, input_batchdata: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention
        input_ids, labels, llm_mask, residue_seq = input_batchdata

        batched_data = {}
        batched_data["x"] = residue_seq
        masked_aa = torch.zeros_like(residue_seq).unsqueeze(-1).bool()

        masked_pos = masked_aa
        batched_data["masked_aa"] = masked_aa
        batched_data["mask_pos"] = masked_pos

        if residue_seq.shape[1] > 2:
            (
                x,
                attn_bias,
                delta_pos,
                pos,
                inner_states,
                padding_mask,
                mask_pos,
                mask_aa,
            ) = super().forward(batched_data)
        else:
            x = None
            padding_mask = None

        return (
            x,
            padding_mask,
            llm_mask,
            input_ids,
        )
