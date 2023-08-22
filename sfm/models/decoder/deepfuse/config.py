# -*- coding: utf-8 -*-
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig


class SciDeocerType(str, Enum):
    BioGPT = "biogpt"
    LLaMA = "llama"


class LayerUsage(str, Enum):
    Mixing = "M"
    Seperate = "S"
    NotUsed = "N"

    @staticmethod
    def from_str(s: str) -> List["LayerUsage"]:
        return [LayerUsage(c) for c in s]


@dataclass
class DecDeepFuseConfig(DistributedTrainConfig):
    freeze_text_encoder: bool = True

    llama_model: str = "/blob/shufxi/llama/7B"
    entity_decoder_model: str = "/blob/shufxi/molxpt"
    entity_decoder_model_type: SciDeocerType = SciDeocerType.BioGPT
    max_txt_len_llama: int = 500
    max_txt_len_smiles: int = 2048
    end_sym: str = "\n"

    train_data_path: str = ""
    val_data_path: str = ""

    hidden_size: int = 1024
    intermediate_size: int = 4096
    hidden_act: str = "gelu"

    entity_hidden_act: str = "gelu"
    entity_hidden_size: int = 1024
    entity_intermediate_size: int = 1024

    num_attention_heads: int = 16
    num_key_value_heads: int = 1  # KV grouping
    entity_num_attention_heads: int = (
        16  # For now, assume to be less than or equal to num_attention_heads
    )
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[Dict[str, float]] = None

    num_adapter_layers: int = 2
    adapter_hidden_size: int = 1024
    adapter_activation: str = "gelu"

    vocab_size: int = 0  # total vocab size

    # This is a string representing how the science decoder layers are used.
    # For example, "MSNMSN" means that:
    # the first and fourth layers are used for Mixing,
    # the second and fifth layers are used Separately, i.e., no attention between them,
    # and the third and sixth layers are Not used, i.e., only text layers are used.
    # The string lenth should be equal to the number of text layers.
    layer_usage: str = ""

    text_loss_weight: float = 0.0
    entity_loss_weight: float = 1.0

    pretraining_tp: int = 0

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def entity_head_dim(self):
        return self.entity_hidden_size // self.entity_num_attention_heads

    @property
    def max_position_embeddings(self):
        return self.max_txt_len_llama + self.max_txt_len_smiles

    # The following are used by BioGptDecoderLayer
    attention_probs_dropout_prob: float = 0.0
    hidden_dropout_prob: float = 0.0
    activation_dropout: float = 0.0
