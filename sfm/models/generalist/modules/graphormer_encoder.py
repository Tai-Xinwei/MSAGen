# -*- coding: utf-8 -*-
from typing import Optional, Tuple

from sfm.models.graphormer.modules.graphormer_sentence_encoder import (
    GraphormerSentenceEncoder,
)


class GraphormerSentenceEncoderPP(GraphormerSentenceEncoder):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

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
        graphormer_config,
    ) -> None:
        # inheret from GraphormerSentenceEncoder
        super().__init__(graphormer_config)
        self.graphormer_config = graphormer_config

    @classmethod
    def config(cls):
        return cls.graphormer_config

    def forward(self, input_batchdata: Tuple):
        (
            input_ids,
            llm_mask,
            _,
            x,
            in_degree,
            out_degree,
            attn_bias,
            spatial_pos,
            edge_input,
            num_atoms,
            pos,
            mask3d,
            node_type_edge,
        ) = input_batchdata

        # create dict for batched data
        batched_data = {}
        batched_data["attn_bias"] = attn_bias
        batched_data["spatial_pos"] = spatial_pos
        batched_data["in_degree"] = in_degree
        batched_data["out_degree"] = out_degree
        batched_data["x"] = x
        batched_data["edge_input"] = edge_input
        batched_data["attn_edge_type"] = None
        batched_data["pos"] = pos
        batched_data["mask3d"] = mask3d
        batched_data["node_type_edge"] = node_type_edge

        x, _, _, _, _, padding_mask = super().forward(batched_data)

        return (x, padding_mask, llm_mask, input_ids)
