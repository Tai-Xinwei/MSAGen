# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass

from transformers.models.llama.configuration_llama import LlamaConfig


@dataclass
class ScigptConfig(LlamaConfig):
    model_type = "scigpt"

    vocab_size: int = 32000
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
    pad_token_id: int = None
    bos_token_id: int = 1
    eos_token_id: int = 2
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling = None

    dict_path: str = ""
    train_data_path: str = ""
    valid_data_path: str = ""
    loadcheck_path: str = ""

    ft: bool = False
    infer: bool = False

    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(kwargs)
        scigpt_base_architecture(args)

        # set attributes of args to self
        for k, v in asdict(self).items():
            if hasattr(args, k):
                setattr(self, k, getattr(args, k))

        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.intermediate_size = args.intermediate_size
        self.num_hidden_layers = args.num_hidden_layers
        self.num_attention_heads = args.num_attention_heads

        self.num_key_value_heads = args.num_key_value_heads
        self.hidden_act = args.hidden_act
        self.max_position_embeddings = args.max_position_embeddings
        self.initializer_range = args.initializer_range
        self.rms_norm_eps = args.rms_norm_eps
        self.use_cache = args.use_cache
        self.pad_token_id = args.pad_token_id
        self.bos_token_id = args.bos_token_id
        self.eos_token_id = args.eos_token_id

        self.pretraining_tp = args.pretraining_tp
        self.tie_word_embeddings = args.tie_word_embeddings
        self.rope_theta = args.rope_theta
        self.rope_scaling = args.rope_scaling

        self.dict_path = args.dict_path
        self.train_data_path = args.train_data_path
        self.valid_data_path = args.valid_data_path
        self.loadcheck_path = args.loadcheck_path

        self.ft = args.ft
        self.pretraining_tp = args.pretraining_tp

        self.args = args


def scigpt_base_architecture(args):
    args.model_type = getattr(args, "model_type", "scigpt")
    args.vocab_size = getattr(args, "vocab_size", 32000)
    args.hidden_size = getattr(args, "hidden_size", 4096)
    args.intermediate_size = getattr(args, "intermediate_size", args.hidden_size * 4)
    args.num_hidden_layers = getattr(args, "num_hidden_layers", 32)
    args.num_attention_heads = getattr(args, "num_attention_heads", 32)
    args.num_key_value_heads = getattr(args, "num_key_value_heads", 32)
    args.hidden_act = getattr(args, "hidden_act", "silu")
    args.max_position_embeddings = getattr(args, "max_position_embeddings", "2048")
    args.tokens_per_sample = getattr(args, "tokens_per_sample", "2048")
    args.initializer_range = getattr(args, "initializer_range", 0.02)
    args.rms_norm_eps = getattr(args, "rms_norm_eps", 1e-6)
    args.use_cache = getattr(args, "use_cache", True)
    args.pad_token_id = getattr(args, "pad_token_id", None)
    args.bos_token_id = getattr(args, "bos_token_id", 1)
    args.eos_token_id = getattr(args, "eos_token_id", 2)
    args.pretraining_tp = getattr(args, "pretraining_tp", 1)
    args.tie_word_embeddings = getattr(args, "tie_word_embeddings", False)
    args.rope_theta = getattr(args, "rope_theta", 10000.0)
    args.rope_scaling = getattr(args, "rope_scaling", None)

    args.dict_path = getattr(args, "dict_path", "")
    args.train_data_path = getattr(args, "train_data_path", "")
    args.valid_data_path = getattr(args, "valid_data_path", "")
    args.loadcheck_path = getattr(args, "loadcheck_path", "")

    args.ft = getattr(args, "ft", False)
    args.infer = getattr(args, "infer", False)
