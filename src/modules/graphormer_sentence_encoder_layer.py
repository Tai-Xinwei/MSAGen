# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn


from .multihead_attention import MultiheadAttention
from .droppath import DropPath
from .layer_norm import LayerNorm, Fp32LayerNorm
from .quant_noise import quant_noise
from .get_activation_fn import get_activation_fn
from .FairseqDropout import FairseqDropout
from models.transformers.configuration_utils import PretrainedConfig

class GraphormerSentenceEncoderLayer(nn.Module):
    """
    Implements a Graphormer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        sandwich_ln: bool = False,
        droppath_prob: float = 0.0,
        nl: int = 0,
        self_attn_mask: Optional[torch.Tensor] = None,
        args = None,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size

        if droppath_prob > 0.0:
            self.dropout_module = DropPath(droppath_prob)
        else:
            self.dropout_module = FairseqDropout(
                dropout, module_name=self.__class__.__name__
            )

        self.activation_dropout_module = FairseqDropout(
            activation_dropout, module_name=self.__class__.__name__
        )

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            d_tilde=args.d_tilde,
        )

        # @ Roger added:
        self.sandwich_ln = sandwich_ln

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        # layer norm associated with the self attention layer, sandwich
        # self.self_attn_sandwich_layer_norm = LayerNorm(self.embedding_dim, export=export) if self.sandwich_ln else None

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        # self.final_sandwich_layer_norm = LayerNorm(self.embedding_dim, export=export) if self.sandwich_ln else None
        self.final_layer_norm_2 = LayerNorm(ffn_embedding_dim, export=export)

        self.nl = nl
        self.args = args
        self.self_attn_mask = self_attn_mask
        
        # self.dummy = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
        self.dummy = nn.Linear(1, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.self_attn_layer_norm.reset_parameters()
        self.final_layer_norm.reset_parameters()
        self.final_layer_norm_2.reset_parameters()

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
        d_tilde=1,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            d_tilde=d_tilde,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C

        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.final_layer_norm_2(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        return x, attn


class GraphormerSentenceEncoderLayer_PP(GraphormerSentenceEncoderLayer):
    """
    Implements a Graphormer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """
    @classmethod
    def config(cls):
        return GraphormerConfig(
            hidden_size = cls.embedding_dim,
            intermediate_size = cls.ffn_embedding_dim,
            num_attention_heads = cls.num_attention_heads,
            hidden_act="relu",
        )
    
    def tensors_encode(self, x, self_attn_bias, delta_pos):
        shape_tensor = torch.cat([torch.tensor(x.shape), torch.tensor(self_attn_bias.shape), torch.tensor(delta_pos.shape)], dim=-1)
        output = torch.cat([x.contiguous().view(-1), self_attn_bias.contiguous().view(-1), delta_pos.contiguous().view(-1)], dim=-1)

        return output, shape_tensor.to(x.device)

    def tensors_decode(self, output, shape_tensor):
        
        x_len = shape_tensor[0]*shape_tensor[1]*shape_tensor[2]
        self_attn_bias_len = shape_tensor[3]*shape_tensor[4]*shape_tensor[5]*shape_tensor[6]*shape_tensor[7]
        delta_pos_len = shape_tensor[8]*shape_tensor[9]*shape_tensor[10]*shape_tensor[11]

        x = output[:x_len].view(shape_tensor[0], shape_tensor[1], shape_tensor[2])
        self_attn_bias = output[x_len:x_len+self_attn_bias_len].view(shape_tensor[3], shape_tensor[4], shape_tensor[5], shape_tensor[6], shape_tensor[7])
        delta_pos = output[x_len+self_attn_bias_len:].view(shape_tensor[8], shape_tensor[9], shape_tensor[10], shape_tensor[11])
        
        return x, self_attn_bias, delta_pos
    
    def forward(self, input_tuple: tuple):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        if not self.args.infer:
            x, self_attn_padding_mask, self_attn_bias, delta_pos, pos = input_tuple
        else:
            x, self_attn_padding_mask, self_attn_bias, input_ids, llm_mask = input_tuple
        # value_tensor, self_attn_padding_mask, shape_tensor = input_tuple
        # x, self_attn_bias, delta_pos = self.tensors_decode(value_tensor, shape_tensor)
        # x = x.to(torch.float16)
        # print("layer", self.nl)
        assert type(x) == torch.Tensor  
        assert type(self_attn_bias) == torch.Tensor
        assert type(self_attn_padding_mask) == torch.Tensor
        # assert type(delta_pos) == torch.Tensor

        self_attn_mask = None
        attn_bias_temp = self_attn_bias[:, self.nl, :, :, :]
        # x: T x B x C
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=attn_bias_temp,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.final_layer_norm_2(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.args.infer:
            return (x.contiguous(), self_attn_padding_mask.contiguous(), self_attn_bias.contiguous(), delta_pos.contiguous(), pos.contiguous())
        else:
            return (x.contiguous(), self_attn_padding_mask.contiguous(), self_attn_bias.contiguous(), input_ids.contiguous(), llm_mask.contiguous())
        # value_tensor, shape_tensor = self.tensors_encode(x, self_attn_bias, delta_pos)
        # return (value_tensor, self_attn_padding_mask, shape_tensor)


class GraphormerConfig(PretrainedConfig):

    model_type = "graphormer"
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        molrep_dict_path=None,
        mfm_hidden_size=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.molrep_dict_path = molrep_dict_path
        self.mfm_hidden_size = mfm_hidden_size
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )