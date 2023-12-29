# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Optional

from transformers.models.llama.configuration_llama import LlamaConfig

from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig


@dataclass
class PfmMlmConfig(LlamaConfig, DistributedTrainConfig):
    model_type: str = "pfm_mlm_tiny"
    mask_token_id: int = 16384
    mask_prob: float = 0.15
    leave_unmasked_prob: float = 0.1
    random_token_prob: float = 0.1
    vocab_size: int = 16385  # 16384 + 1 for the mask token
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 3
    max_position_embeddings: int = 1024

    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, str]] = None
    attention_bias: bool = False
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    pretraining_tp: int = 1

    train_data_path: str = ""
    valid_data_path: str = ""


def pfm_mlm_tiny_config(config: PfmMlmConfig):
    config.hidden_size = 768
    config.intermediate_size = 2048
    config.num_hidden_layers = 12
    config.num_attention_heads = 12
    config.num_key_value_heads = 12
    return config
