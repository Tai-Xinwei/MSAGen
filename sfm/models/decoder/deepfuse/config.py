# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from sfm.data.dec_data.datasets import TokenType
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig


class EntityDecoderType(str, Enum):
    BioGPT = "BioGPT"
    LLaMA = "LLaMA"


class TextDecoderType(str, Enum):
    LLaMA2_7B = "LLaMA2_7B"


class LayerUsage(str, Enum):
    Mixing = "M"
    Seperate = "S"
    NotUsed = "N"

    @staticmethod
    def from_str(s: str) -> List["LayerUsage"]:
        return [LayerUsage(c) for c in s if c in "MSN"]


@dataclass
class DataConfig:
    data_type: str = "text2mol"  # text2mol, mixed,

    # For text2mol
    train_mol_path: str = "/blob/shufxi/data/tamgent/chebi/train.textmol.smi"
    train_text_path: str = "/blob/shufxi/data/tamgent/chebi/train.textmol.desc"
    val_mol_path: str = "/blob/shufxi/data/tamgent/chebi/val.textmol.smi"
    val_text_path: str = "/blob/shufxi/data/tamgent/chebi/val.textmol.desc"


@dataclass
class DecDeepFuseInferenceConfig:
    ckpt_folder: str = ""
    input_file: str = ""
    output_file: str = ""
    decoder_batch_size: int = 1
    max_length: int = 64
    max_new_tokens: int = 64


@dataclass
class DecDeepFuseConfig(DistributedTrainConfig):
    load_from_pretrained: bool = True
    freeze_text_model: bool = True
    finetune_text_extra_emb: bool = True
    freeze_entity_model: bool = True

    llama_model: str = "/hai1/ds_dataset/llama2/llama-2-7b"
    entity_decoder_model: str = "/home/shufxi/mixgpt/mixgpt_new/ckpt"
    llama_model_type: TextDecoderType = TextDecoderType.LLaMA2_7B
    entity_decoder_model_type: EntityDecoderType = EntityDecoderType.BioGPT
    max_text_len: int = 1024
    max_entity_len: int = 1024

    hidden_size: int = 0
    intermediate_size: int = 0
    hidden_act: str = ""
    num_hidden_layers: int = 0

    entity_hidden_act: str = ""
    entity_hidden_size: int = 0
    entity_intermediate_size: int = 0
    entity_num_hidden_layers: int = 0

    num_attention_heads: int = 0
    num_key_value_heads: int = 0  # KV grouping
    entity_num_attention_heads: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, float]] = None

    num_adapter_layers: int = 2
    adapter_hidden_size: int = 64
    adapter_activation: str = "gelu"

    vocab_size: int = 0  # total vocab size
    entity_vocab_size: int = 0  # total entity vocab size

    # This is a string representing how the science decoder layers are used.
    # For example, "MSNMSN" means that:
    # the first and fourth layers are used for *M*ixing,
    # the second and fifth layers are used *S*eparately, i.e., no attention between them,
    # and the third and sixth layers are *N*ot used, i.e., only text layers are used.
    # The string lenth should be equal to the number of text layers.
    # You can also use any separator, e.g., "M-S-N-M-S-N".
    layer_usage: str = "NNNN-SSSS-MSSS-SSSS-SSSS-MSSS-SSSS-NNNN"

    loss_weight: Dict[str, float] = field(
        default_factory=lambda: {
            TokenType.Text.name: 1.0,
            TokenType.Entity.name: 1.0,
        }
    )

    pretraining_tp: int = 0

    attention_bias: bool = False

    new_token_count: int = 0

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def entity_head_dim(self):
        return self.entity_hidden_size // self.entity_num_attention_heads

    @property
    def max_position_embeddings(self):
        return max(self.max_text_len, self.max_entity_len)

    # The following are used by BioGptDecoderLayer
    attention_probs_dropout_prob: float = 0.0
    hidden_dropout_prob: float = 0.0
    activation_dropout: float = 0.0

    iters_per_epoch: int = 0

    # For generation
    is_encoder_decoder: bool = False


def llama2_7b_default_config():
    return {
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "max_entity_len": 2048,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "rms_norm_eps": 1e-05,
        "vocab_size": 32000,
    }


def mix_gpt_default_config():
    return {
        "entity_hidden_act": "gelu",
        "entity_hidden_size": 1024,
        "entity_intermediate_size": 4096,
        "entity_num_attention_heads": 16,
        "entity_num_hidden_layers": 24,
        "entity_vocab_size": 1488,
        "new_token_count": len(
            [
                "[M]",
                "[/M]",
                "[P]",
                "[/P]",
                "[A]",
                "[/A]",
                "[T]",
                "[/T]",
                "[R]",
                "[PAD]",
            ]
        ),
    }
