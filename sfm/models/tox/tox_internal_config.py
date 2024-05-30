# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass


@dataclass
class ToxInternalConfig:
    # ft: bool = False
    # infer: bool = False

    # ToxInternalLMDBDataset dataset args:
    data_path: str
    # seed: int = 666  # also defined in trainer
    max_length: int
    min_length: int
    transform_str: str

    # ToxInternalModel args:
    load_ckpt_from: str

    num_residues: int

    # ToxInternalEncoder args
    layerdrop: float
    num_encoder_layers: int

    # ToxInternalEncoderLayer args
    dropout: float
    droppath_prob: float
    activation_dropout: float
    attntion_dropout: float
    activation_fn: str

    embedding_dim: int
    ffn_embedding_dim: int
    num_attention_heads: int
    q_noise: float
    qn_block_size: int
    d_tilde: float
    add_rope: bool

    # InitialLoss args
    seq_type_loss_weight: float
    disto_loss_weight: float
    bl_loss_weight: float
    ba_loss_weight: float
    ba_norm_loss_weight: float
    da_loss_weight: float
    da_norm_loss_weight: float
    eps: float
