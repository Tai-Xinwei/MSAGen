# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Optional

from transformers.models.mixtral.configuration_mixtral import MixtralConfig

from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig


@dataclass
class ScigptMoeConfig(MixtralConfig, DistributedTrainConfig):
    model_type: str = "scigpt_moe"

    learnable_cutoff: int = 0

    pretrained_ckpt_path: Optional[str] = None


def scigptmoe_tiny_config(config: ScigptMoeConfig):
    # just for debug
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_hidden_layers = 2
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    return config
