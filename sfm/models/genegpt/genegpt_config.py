# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Optional

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mixtral.configuration_mixtral import MixtralConfig

from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig


@dataclass
class GenegptConfig(MixtralConfig, DistributedTrainConfig):
    model_type: str = "genegpt"

    vocab_size: int = 4100
    learnable_cutoff: int = 0
    hidden_size: int = 768
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 16348
    tokens_per_sample: int = 16348
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 4099
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 0
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, str]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_unified_batch_sampler: bool = False
    dict_path: str = ""
    train_data_path: str = ""
    valid_data_path: str = ""
    pretrained_ckpt_path: str = ""
    load_ckpt: bool = False
    pruned_heads: bool = False
    output_attentions: bool = False
    ft: bool = False
    infer: bool = False
    return_dict: bool = True
    torchscript: bool = False


def genegpt_tiny_config(config: GenegptConfig):
    # just for debug
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_hidden_layers = 2
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    return config


def genegpt_shallow_config(config: GenegptConfig):
    # just for debug
    config.hidden_size = 4096
    config.intermediate_size = 11008
    config.num_hidden_layers = 2
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    return config


def genegpt_100m_config(config: GenegptConfig):
    config.hidden_size = 768
    config.intermediate_size = 1536
    config.num_hidden_layers = 12
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    return config


def genegpt_1b_config(config: GenegptConfig):
    config.hidden_size = 2048
    config.intermediate_size = 4096
    config.num_hidden_layers = 24
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    return config


@dataclass
class GenegptConfig3D(LlamaConfig, DistributedTrainConfig):
    model_type: str = "genegpt"

    vocab_size: int = 4100
    learnable_cutoff: int = 0
    hidden_size: int = 768
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 16348
    tokens_per_sample: int = 16348
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = 4099
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 0
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 500000.0
    rope_scaling: Optional[Dict[str, str]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_unified_batch_sampler: bool = False
    dict_path: str = ""
    train_data_path: str = ""
    valid_data_path: str = ""
    pretrained_ckpt_path: str = ""
    load_ckpt: bool = False

    ft: bool = False
    infer: bool = False


def genegpt3D_100m_config(config: GenegptConfig3D):
    config.hidden_size = 768
    config.intermediate_size = 1536
    config.num_hidden_layers = 16
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    return config


def genegpt3D_1b_config(config: GenegptConfig3D):
    config.hidden_size = 2048
    config.intermediate_size = 5504
    config.num_hidden_layers = 16
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    return config
