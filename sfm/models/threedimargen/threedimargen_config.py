# -*- coding: utf-8 -*-
import os
from dataclasses import asdict, dataclass

from transformers.models.llama.configuration_llama import LlamaConfig

from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig


@dataclass
class ThreeDimARGenConfig(LlamaConfig, DistributedTrainConfig):
    model_type: str = "threedimargen"
    tokenizer: str = "num"

    vocab_size: int = 100
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    tokens_per_sample: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    mask_token_id: int = None
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling = None
    attention_bias: bool = False
    attention_dropout: float = 0.0

    max_sites: int = None
    scale_coords: float = None
    scale_energy: float = None
    reorder: bool = False
    niggli_reduced: bool = False

    dict_path: str = os.path.join(
        os.path.dirname(__file__), "../../data/threedimargen_data/dict.txt"
    )
    train_data_path: str = "/hai1/SFM/threedimargen/data/materials_data/mp.jsonl"
    valid_data_path: str = None
    loadcheck_path: str = None

    ft: bool = False
    infer: bool = False

    # for diffusion
    num_timesteps_stepsize: int = -250
    ddpm_schedule: str = "sigmoid"
    num_timesteps: int = 5000
    ddpm_beta_start: float = 1e-7
    ddpm_beta_end: float = 2e-3
    diffusion_noise_std: float = 1.0

    # only for dpm solver
    algorithm_type: str = "dpmsolver++"
    solver_order: int = 2
    solver_type: str = "midpoint"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


@dataclass
class ThreeDimARGenInferenceConfig:
    input_file: str = None
    output_file: str = None
    infer_batch_size: int = 128
    max_length: int = None
    max_new_tokens: int = None
    verbose: bool = False
    space_group: bool = True
    sample: bool = False


def threedimargen_tiny_config(config: ThreeDimARGenConfig):
    # just for debug
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_hidden_layers = 2
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    return config


def threedimargen_base_config(config: ThreeDimARGenConfig):
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_hidden_layers = 24
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    return config


def threedimargen_200m_config(config: ThreeDimARGenConfig):
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_hidden_layers = 12
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    return config


def threedimargen_100m_config(config: ThreeDimARGenConfig):
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_hidden_layers = 6
    config.num_attention_heads = 16
    config.num_key_value_heads = 16
    return config


def threedimargen_1_6_b_config(config: ThreeDimARGenConfig):
    config.hidden_size = 2048
    config.intermediate_size = 8192
    config.num_hidden_layers = 24
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    return config


def threedimargen_3_3_b_config(config: ThreeDimARGenConfig):
    config.hidden_size = 2560
    config.intermediate_size = 10240
    config.num_hidden_layers = 32
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    return config
