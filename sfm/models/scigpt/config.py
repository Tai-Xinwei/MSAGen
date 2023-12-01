# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Optional

from transformers.models.llama.configuration_llama import LlamaConfig

from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig


@dataclass
class ScigptConfig(LlamaConfig, DistributedTrainConfig):
    model_type: str = "scigpt"

    vocab_size: int = 34177
    learnable_cutoff: int = 0
    hidden_size: int = 768
    intermediate_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    tokens_per_sample: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 32000
    bos_token_id: int = 1
    eos_token_id: int = 2
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, str]] = None
    attention_bias: bool = False

    dict_path: str = ""
    train_data_path: str = ""
    valid_data_path: str = ""
    pretrained_ckpt_path: str = ""
    load_ckpt: bool = False

    ft: bool = False
    infer: bool = False


def scigpt_tiny_config(config: ScigptConfig):
    # just for debug
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_hidden_layers = 2
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    return config


def scigpt_shallow_config(config: ScigptConfig):
    # just for debug
    config.hidden_size = 4096
    config.intermediate_size = 11008
    config.num_hidden_layers = 2
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    return config


def scigpt_350m_config(config: ScigptConfig):
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_hidden_layers = 24
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    return config


def scigpt_7b_config(config: ScigptConfig):
    config.hidden_size = 4096
    config.intermediate_size = 11008
    config.num_hidden_layers = 32
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    return config


def scigpt_7b_1k_config(config: ScigptConfig):
    config.hidden_size = 4096
    config.intermediate_size = 11008
    config.num_hidden_layers = 32
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    return config


def scigpt_7b_512_config(config: ScigptConfig):
    config.hidden_size = 4096
    config.intermediate_size = 11008
    config.num_hidden_layers = 32
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    return config
