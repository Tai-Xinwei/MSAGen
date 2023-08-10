# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn

from megatron.core import parallel_state
from sfm.utils.pretrained_layer_spec import PretrainedLayerSpec

from .graphormer_embedding import GraphormerEmbeddingMP
from .graphormer_sentence_encoder_layer_MP import GraphormerSentenceEncoderLayerMP


class GraphormerEncoderMP(nn.Module):
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
        mp_config,
        layer_id=0,
        load_ckpt=False,
        ckp_list=[],
    ):
        cls.pipe_layer = []
        cls.pipe_layer.append(
            PretrainedLayerSpec(
                GraphormerEmbeddingMP,
                graphormer_config,
                args,
                mp_config,
                load_ckpt=load_ckpt,
                pretrained_ckpt_path=os.path.join(
                    args.loadmfmcheck_path, ckp_list[layer_id]
                ),
                lora_mode="freeze",
                tp_model_size=args.tensor_model_parallel_size,
                tp_rank=parallel_state.get_tensor_model_parallel_rank(),
            )
        )
        layer_id += 1
        for i in range(graphormer_config.encoder_layers):
            cls.pipe_layer.append(
                PretrainedLayerSpec(
                    GraphormerSentenceEncoderLayerMP,
                    graphormer_config,
                    mp_config,
                    args,
                    i,
                    load_ckpt=load_ckpt,
                    pretrained_ckpt_path=os.path.join(
                        args.loadmfmcheck_path, ckp_list[layer_id]
                    ),
                    lora_mode="freeze",
                    tp_model_size=args.tensor_model_parallel_size,
                    tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                )
            )
            layer_id += 1

        return cls.pipe_layer
