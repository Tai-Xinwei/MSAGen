# -*- coding: utf-8 -*-
import dataclasses

import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from megatron.core.transformer import TransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal
from sfm.logging import logger


class MPLlamaConfig(LlamaConfig):
    def __init__(
        self,
        args,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_scaling=None,
        init_method=None,
        output_layer_init_method=None,
        embedding_dropout_prob=0.0,
        seq_length=2048,
        rotary_percent=1.0,
        **kwargs,
    ):
        for f in dataclasses.fields(TransformerConfig):
            if hasattr(args, f.name):
                setattr(self, f.name, getattr(args, f.name))

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            pretraining_tp=pretraining_tp,
            use_cache=use_cache,
            rope_scaling=rope_scaling,
            **kwargs,
        )

        setattr(self, "persist_layer_norm", not args.no_persist_layer_norm)
        setattr(self, "layernorm_zero_centered_gamma", args.apply_layernorm_1p)
        setattr(self, "deallocate_pipeline_outputs", True)
        setattr(self, "pipeline_dtype", args.params_dtype)
        setattr(self, "vocab_size", vocab_size)
        setattr(self, "batch_p2p_comm", not args.overlap_p2p_comm)
        setattr(self, "activation_func", torch.nn.functional.silu)
        setattr(self, "gated_linear_unit", True)
        setattr(self, "bias_gelu_fusion", False)
        setattr(self, "add_bias_linear", False)

        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method
        self.embedding_dropout_prob = embedding_dropout_prob
        self.seq_length = seq_length
        self.rotary_percent = rotary_percent

        assert (
            self.num_attention_heads % self.num_key_value_heads == 0
        ), f"num_attention_heads {self.num_attention_heads} must be divisible by num_key_value_heads {self.num_key_value_heads}"

        if not self.add_bias_linear:
            self.bias_dropout_fusion = False
            logger.info(
                "Setting bias_dropout_fusion to False because add_bias_linear is False"
            )

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std, num_hidden_layers
            )
