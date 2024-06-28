# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Optional

from transformers.models.mixtral.configuration_mixtral import MixtralConfig

from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig


@dataclass
class MoeModelConfig(MixtralConfig, DistributedTrainConfig):
    model_type: str = "scigpt_moe"

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
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, str]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    num_experts_per_tok: int = 2
    num_local_experts: int = 8
    output_router_logits: bool = True
    router_aux_loss_coef: float = 0.02
    router_jitter_noise: float = 0.01  # From SwitchTransformer
    moe_impl: str = "vanilla"  # grouped, sparse, vanilla
    moe_memory_optimized_mlp: bool = False

    dict_path: str = ""
    prot_spm_path: str = "/blob/shufxi/data/scigpt/ur50bpe/bpe"
    dna_spm_path: str = "/blob/shufxi/data/scigpt/dnabpe/bpe"
    rna_spm_path: str = "/blob/shufxi/data/scigpt/rnabpe/bpe"
    data_dir: str = ""
    data_raito: str = ""
    train_data_path: str = ""
    train_data_ratio: str = ""
    valid_data_path: str = ""
    pretrained_ckpt_path: str = ""
    load_ckpt: bool = False

    ft: bool = False
    infer: bool = False
    weighted_dataset: bool = False


def sfm_nlm_moe_tiny_config(config: MoeModelConfig):
    # just for debug
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_hidden_layers = 16
    config.num_attention_heads = 8
    config.num_key_value_heads = 8
    return config


def sfm_nlm_moe_8x7b_config(config: MoeModelConfig):
    # see https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
    config.hidden_size = 4096
    config.intermediate_size = 14336
    config.num_hidden_layers = 32
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    return config


def sfm_nlm_1b_base_config(config: MoeModelConfig):
    config.hidden_size = 2048
    config.intermediate_size = 5504
    config.num_hidden_layers = 16
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    config.max_position_embeddings = 8192
    config.tokens_per_sample = 8192
    return config
