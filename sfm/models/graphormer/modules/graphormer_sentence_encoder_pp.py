# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn

from sfm.utils.pretrained_layer_spec import PretrainedLayerSpec

from .graphormer_embedding import GraphormerEmbeddingPP
from .graphormer_sentence_encoder_layer import GraphormerSentenceEncoderLayer_PP


class GraphormerEncoderPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=True
        )

        self.pipe_layer = []

    @classmethod
    def to_layers(
        cls,
        args,
        graphormer_config,
        layer_id=0,
        load_ckpt=False,
        ckp_list=[],
    ):
        cls.pipe_layer = []
        cls.pipe_layer.append(
            PretrainedLayerSpec(
                GraphormerEmbeddingPP,
                graphormer_config,
                args,
                load_ckpt=load_ckpt,
                pretrained_ckpt_path=os.path.join(
                    args.loadmfmcheck_path,
                    ckp_list[layer_id]
                    if len(ckp_list) > 0
                    else "layer_{}-model_states.pt".format(str(layer_id).zfill(2)),
                ),
                lora_mode="freeze",
            )
        )
        layer_id += 1
        for i in range(graphormer_config.encoder_layers):
            cls.pipe_layer.append(
                PretrainedLayerSpec(
                    GraphormerSentenceEncoderLayer_PP,
                    graphormer_config,
                    args,
                    i,
                    load_ckpt=load_ckpt,
                    pretrained_ckpt_path=os.path.join(
                        args.loadmfmcheck_path,
                        ckp_list[layer_id]
                        if len(ckp_list) > 0
                        else "layer_{}-model_states.pt".format(str(layer_id).zfill(2)),
                    ),
                    lora_mode="freeze",
                )
            )
            layer_id += 1

        return cls.pipe_layer
