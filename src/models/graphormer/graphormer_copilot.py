# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import (
    init_bert_params,
    GraphormerSentenceEncoder,
)
from ..modules.graphormer_layers_pp import GraphNodeFeaturePipe, GraphAttnBiasPipe, Graph3DBiasPipe, NodeTaskHeadPipe
from ..modules.graphormer_sentence_encoder import Pre_sentence_encoder_layer, Post_sentence_encoder_layer
from graphormer.modules.graphormer_sentence_encoder_layer import GraphormerSentenceEncoderLayer, GraphormerSentenceEncoderLayer_PP

from ..utils.mypp_module import LayerSpec
from ..utils.layer_norm import LayerNorm
from ..utils.quant_noise import quant_noise
from ..utils.get_activation_fn import get_activation_fn
from ..utils.pretrained_layer_spec import PretrainedLayerSpec, LoraLayerSpec

from transformers.models.llama.modeling_llama import LlamaModel_PP, hybrid_emb, LlamaDecoderLayerPP, LlamaClassifier
from transformers.models.llama.configuration_llama import LlamaConfig
# from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
# from peft.peft_model import PeftModel

logger = logging.getLogger(__name__)

class GraphormerModelCopilotPP(nn.Module):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(self, args, config, checkpoint_list, load_ckp=False):
        super().__init__()
        self.args = args

        # if specified then apply bert initialization on the model. We need
        # to explictly call this to make sure that the output embeddings
        # and projection layers are also correctly initialized
        if getattr(args, "apply_bert_init", False):
            self.apply(init_bert_params)
        self.encoder_embed_dim = args.encoder_embed_dim

        # add architecture parameters
        # base_architecture(args)
        graphormer_base_architecture(args)

        if not hasattr(args, "max_positions"):
            try:
                args.max_positions = args.tokens_per_sample
            except:
                args.max_positions = args.max_nodes

        logger.info(args)
        # self.classifer = nn.Linear(4000,4000)

        self.encoder = GraphormerEncoderCopilotPP(args, config, checkpoint_list, load_ckp=load_ckp)

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)
    
    def to_layers(self):
        # return self.encoder.to_layers()
        return self.encoder.layers

class GraphormerEncoderCopilotPP(nn.Module):
    """
    Encoder for Masked Language Modelling.
    """

    def __init__(self, args, config, checkpoint_list=None, load_ckp=False):
        super().__init__()
        self.max_positions = args.max_positions

        self.layers = []
        
        if load_ckp:
            layer_idx = 0
            
            self.layers.extend([PretrainedLayerSpec(Pre_sentence_encoder_layer, 
                                    num_atoms=args.num_atoms,
                                    num_in_degree=args.num_in_degree,
                                    num_out_degree=args.num_out_degree,
                                    num_edges=args.num_edges,
                                    num_spatial=args.num_spatial,
                                    num_edge_dis=args.num_edge_dis,
                                    edge_type=args.edge_type,
                                    multi_hop_max_dist=args.multi_hop_max_dist,
                                    num_encoder_layers=args.encoder_layers,
                                    embedding_dim=args.encoder_embed_dim,
                                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                                    num_attention_heads=args.encoder_attention_heads,
                                    dropout=args.dropout,
                                    attention_dropout=args.attention_dropout,
                                    activation_dropout=args.act_dropout,
                                    max_seq_len=self.max_positions,
                                    num_segments=args.num_segment,
                                    use_position_embeddings=not args.no_token_positional_embeddings,
                                    encoder_normalize_before=args.encoder_normalize_before,
                                    apply_bert_init=args.apply_bert_init,
                                    activation_fn=args.activation_fn,
                                    learned_pos_embedding=args.encoder_learned_pos,
                                    sandwich_ln=args.sandwich_ln,
                                    droppath_prob=args.droppath_prob,
                                    add_3d=args.add_3d,
                                    num_3d_bias_kernel=args.num_3d_bias_kernel,
                                    no_2d=args.no_2d,
                                    args=args,
                                    pretrained_ckpt_path=checkpoint_list[layer_idx])])
            layer_idx += 1
            for nl in range(args.encoder_layers):
                # if config.mfm_lora:
                self.layers.append(LoraLayerSpec(GraphormerSentenceEncoderLayer_PP,
                                            ifload=True,
                                            lora=config.mfm_lora,
                                            embedding_dim=args.encoder_embed_dim,
                                            ffn_embedding_dim=args.encoder_ffn_embed_dim,
                                            num_attention_heads=args.encoder_attention_heads,
                                            dropout=args.dropout,
                                            attention_dropout=args.attention_dropout,
                                            activation_dropout=args.act_dropout,
                                            activation_fn=args.activation_fn,
                                            sandwich_ln=args.sandwich_ln,
                                            droppath_prob=args.droppath_prob,
                                            nl=nl,
                                            args=args,
                                            pretrained_ckpt_path=checkpoint_list[layer_idx]),
                                            )
                # else:
                #     self.layers.append(PretrainedLayerSpec(GraphormerSentenceEncoderLayer_PP,
                #                 embedding_dim=args.encoder_embed_dim,
                #                 ffn_embedding_dim=args.encoder_ffn_embed_dim,
                #                 num_attention_heads=args.encoder_attention_heads,
                #                 dropout=args.dropout,
                #                 attention_dropout=args.attention_dropout,
                #                 activation_dropout=args.act_dropout,
                #                 activation_fn=args.activation_fn,
                #                 sandwich_ln=args.sandwich_ln,
                #                 droppath_prob=args.droppath_prob,
                #                 nl=nl,
                #                 args=args,
                #                 pretrained_ckpt_path=checkpoint_list[layer_idx]),
                #                 )
                layer_idx += 1
            self.layers.extend([PretrainedLayerSpec(Post_sentence_encoder_layer,
                                            args=args,
                                            embedding_dim=args.encoder_embed_dim, 
                                            num_attention_heads=args.encoder_attention_heads, 
                                            num_pred_attn_layer=args.num_pred_attn_layer, 
                                            num_3d_bias_kernel=args.num_3d_bias_kernel,
                                            pretrained_ckpt_path=checkpoint_list[layer_idx],
                                            )])
            layer_idx += 1
            self.layers.extend([PretrainedLayerSpec(Post_decoder, args, pretrained_ckpt_path=checkpoint_list[layer_idx])])
            layer_idx += 1

            self.layers.append(PretrainedLayerSpec(hybrid_emb, config, pretrained_ckpt_path=checkpoint_list[layer_idx]))
            layer_idx += 1
            for l in range(config.num_hidden_layers):
                self.layers.append(PretrainedLayerSpec(LlamaDecoderLayerPP, config, l, pretrained_ckpt_path=checkpoint_list[layer_idx]))
                layer_idx += 1
            self.layers.append(PretrainedLayerSpec(LlamaClassifier, config, pretrained_ckpt_path=checkpoint_list[layer_idx]))
            layer_idx += 1
            
        else:
            self.layers.extend([LayerSpec(Pre_sentence_encoder_layer, 
                                            num_atoms=args.num_atoms,
                                            num_in_degree=args.num_in_degree,
                                            num_out_degree=args.num_out_degree,
                                            num_edges=args.num_edges,
                                            num_spatial=args.num_spatial,
                                            num_edge_dis=args.num_edge_dis,
                                            edge_type=args.edge_type,
                                            multi_hop_max_dist=args.multi_hop_max_dist,
                                            num_encoder_layers=args.encoder_layers,
                                            embedding_dim=args.encoder_embed_dim,
                                            ffn_embedding_dim=args.encoder_ffn_embed_dim,
                                            num_attention_heads=args.encoder_attention_heads,
                                            dropout=args.dropout,
                                            attention_dropout=args.attention_dropout,
                                            activation_dropout=args.act_dropout,
                                            max_seq_len=self.max_positions,
                                            num_segments=args.num_segment,
                                            use_position_embeddings=not args.no_token_positional_embeddings,
                                            encoder_normalize_before=args.encoder_normalize_before,
                                            apply_bert_init=args.apply_bert_init,
                                            activation_fn=args.activation_fn,
                                            learned_pos_embedding=args.encoder_learned_pos,
                                            sandwich_ln=args.sandwich_ln,
                                            droppath_prob=args.droppath_prob,
                                            add_3d=args.add_3d,
                                            num_3d_bias_kernel=args.num_3d_bias_kernel,
                                            no_2d=args.no_2d,
                                            args=args,)])
            for nl in range(args.encoder_layers):
                # if config.mfm_lora:
                self.layers.append(LoraLayerSpec(GraphormerSentenceEncoderLayer_PP,
                                            lora=config.mfm_lora,
                                            embedding_dim=args.encoder_embed_dim,
                                            ffn_embedding_dim=args.encoder_ffn_embed_dim,
                                            num_attention_heads=args.encoder_attention_heads,
                                            dropout=args.dropout,
                                            attention_dropout=args.attention_dropout,
                                            activation_dropout=args.act_dropout,
                                            activation_fn=args.activation_fn,
                                            sandwich_ln=args.sandwich_ln,
                                            droppath_prob=args.droppath_prob,
                                            nl=nl,
                                            args=args,))
                # else:
                #     self.layers.append(LayerSpec(GraphormerSentenceEncoderLayer_PP, 
                #                                 embedding_dim=args.encoder_embed_dim,
                #                                 ffn_embedding_dim=args.encoder_ffn_embed_dim,
                #                                 num_attention_heads=args.encoder_attention_heads,
                #                                 dropout=args.dropout,
                #                                 attention_dropout=args.attention_dropout,
                #                                 activation_dropout=args.act_dropout,
                #                                 activation_fn=args.activation_fn,
                #                                 sandwich_ln=args.sandwich_ln,
                #                                 droppath_prob=args.droppath_prob,
                #                                 nl=nl,
                #                                 args=args,))
            self.layers.extend([LayerSpec(Post_sentence_encoder_layer,
                                            args=args,
                                            embedding_dim=args.encoder_embed_dim, 
                                            num_attention_heads=args.encoder_attention_heads, 
                                            num_pred_attn_layer=args.num_pred_attn_layer, 
                                            num_3d_bias_kernel=args.num_3d_bias_kernel
                                            )])
            self.layers.extend([LayerSpec(Post_decoder, args)])
            

            self.layers.append(LayerSpec(hybrid_emb, config))
            for l in range(config.num_hidden_layers):
                self.layers.append(LayerSpec(LlamaDecoderLayerPP, config, l))
            self.layers.append(LayerSpec(LlamaClassifier, config))
        
        

    def forward(self, batched_data, perturb=None, segment_labels=None, masked_tokens=None, **unused):
        raise Exception("Forward of GraphormerEncoderPP should not be used")
    

    def to_layers(self):
        g_encoder_layers = []
        g_encoder_layers.extend([self.pre_sentence_encoder_layer])
        g_encoder_layers.extend(self.g_layers)
        g_encoder_layers.extend([self.post_sentence_encoder_layer])
        g_encoder_layers.extend([self.post_decoder_layer])

        return g_encoder_layers

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        tmp_dict = {}
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if (
                    "embed_out.weight" in k
                    or "sentence_projection_layer.weight" in k
                    or "lm_output_learned_bias" in k
                    or "regression_lm_head_list" in k
                    or "regression_ln_list" in k
                    or "regression_embed_out_list" in k
                    or "classification_lm_head_list" in k
                    or "classification_ln_list" in k
                    or "classification_embed_out_list" in k
                ):
                    print("Removing", k, "(because load_softmax is False)")
                    tmp_dict[k] = state_dict[k]
                    del state_dict[k]
            proj_weight = torch.rand(
                self.proj_out.weight.shape
            )
            proj_bias = torch.rand(self.proj_out.bias.shape)

            # lm_head_transform_weight_weight = torch.rand(self.lm_head_transform_weight.weight.shape)
            # lm_head_transform_weight_bias = torch.rand(self.lm_head_transform_weight.bias.shape)
            lm_head_transform_weight_weight = tmp_dict.get("encoder.regression_lm_head_list.0.weight", None)
            lm_head_transform_weight_bias = tmp_dict.get("encoder.regression_lm_head_list.0.bias", None)
            ln_weight = tmp_dict.get("encoder.regression_ln_list.0.weight", None)
            ln_bias = tmp_dict.get("encoder.regression_ln_list.0.bias", None)

            self.init_state_dict_weight(proj_weight, proj_bias)
            # self.init_state_dict_weight(lm_head_transform_weight_weight, lm_head_transform_weight_bias)

            state_dict["encoder.proj_out.weight"] = state_dict.get("encoder.proj_out.weight", proj_weight)
            state_dict["encoder.proj_out.bias"] = state_dict.get("encoder.proj_out.bias", proj_bias)
            state_dict["encoder.lm_head_transform_weight.weight"] = state_dict.get("encoder.lm_head_transform_weight.weight", lm_head_transform_weight_weight)
            state_dict["encoder.lm_head_transform_weight.bias"] = state_dict.get("encoder.lm_head_transform_weight.bias", lm_head_transform_weight_bias)
            state_dict["encoder.layer_norm.weight"] = state_dict.get("encoder.layer_norm.weight", ln_weight)
            state_dict["encoder.layer_norm.bias"] = state_dict.get("encoder.layer_norm.bias", ln_bias)
        return state_dict

    def init_state_dict_weight(self, weight, bias):
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)


class Post_decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.sentence_projection_layer = None
        self.sentence_out_dim = args.sentence_class_num
        self.proj_out = None
        self.args = args
        # Remove head is set to true during fine-tuning
        self.load_softmax = not args.ft #getattr(args, "remove_head", False)
        print("if finetune:", args.ft)
        # self.layer_norm = layer_norm
        self.activation_fn = get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)
        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.pooler_activation = get_activation_fn(args.pooler_activation_fn)

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1, requires_grad=True))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim, args.num_classes, bias=False
                )
                nn.init.uniform_(self.embed_out.weight, -1.0/(math.sqrt(768.) * args.d_tilde), 1.0/(math.sqrt(768.) * args.d_tilde))

            if args.sent_loss:
                self.sentence_projection_layer = nn.Linear(
                    args.encoder_embed_dim, self.sentence_out_dim, bias=False
                )
                nn.init.uniform_(self.sentence_projection_layer.weight, -1.0/(math.sqrt(768.) * args.d_tilde), 1.0/(math.sqrt(768.) * args.d_tilde))
        else:
            if isinstance(args.num_classes, int):
                self.proj_out = nn.Linear(
                    args.encoder_embed_dim, args.num_classes, bias=True
                )
                nn.init.uniform_(self.proj_out.weight, -1.0/(math.sqrt(768.) * args.d_tilde), 1.0/(math.sqrt(768.) * args.d_tilde))
                # self.proj_out = RobertaClassificationHead(
                #     args.encoder_embed_dim, args.encoder_embed_dim, args.num_classes, args.activation_fn
                # )
            else:
                raise NotImplementedError
        
        self.reset_parameters()
            
    def reset_parameters(self):
        self.masked_lm_pooler.reset_parameters()
        self.lm_head_transform_weight.reset_parameters()


    def tensors_decode(self, value_tensor, shape_tensor):
        x_len = shape_tensor[0]*shape_tensor[1]*shape_tensor[2]
        x = value_tensor[:x_len].view(shape_tensor[0], shape_tensor[1], shape_tensor[2])
        node_output = value_tensor[x_len:].view(shape_tensor[3], shape_tensor[4], shape_tensor[5])
        
        return x, node_output

    def tensors_encode(self, x, node_output):
        shape_tensor = torch.cat([torch.tensor(x.contiguous().shape), torch.tensor(node_output.contiguous().shape)], dim=-1)
        output = torch.cat([x.view(-1), node_output.view(-1)], dim=-1)

        return output, shape_tensor.to(x.device) #, torch.tensor(shape_list)

    # def forward(self, inner_states, masked_tokens, sentence_rep, embed_tokens):
    def forward(self, input_tensor: tuple):
        # x, node_output, sentence_rep, inner_states = input
        if self.args.infer or self.args.ft:
            x, padding_mask, input_ids, llm_mask = input_tensor
        else:
            x, node_output, sentence_rep = input_tensor

            assert type(node_output) == torch.Tensor

        # value_tensor, shape_tensor = input_tensor
        # x, node_output = self.tensors_decode(value_tensor, shape_tensor)
        # x = x.to(torch.float16)

        assert type(x) == torch.Tensor
        # assert type(sentence_rep) == torch.Tensor
        # assert type(inner_states) == torch.Tensor

        # x = inner_states[-1, ...].transpose(0, 1)
        x = x.transpose(0, 1)

        if self.args.infer:
            # print("x shape", x.shape)
            # print("Post_decoder:", x.shape, padding_mask.shape, input_ids.shape, llm_mask.shape)
            # print("Post_decoder:", x.dtype, padding_mask.dtype, input_ids.dtype, llm_mask.dtype)
            return (x, padding_mask, input_ids, llm_mask)
            
        # FIXME: not compatible with batched_data

        # project masked tokens only
        # if masked_tokens is not None:
            # x = x[masked_tokens, :]
        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # pooled_output = self.pooler_activation(self.masked_lm_pooler(sentence_rep))

        # project back to size of vocabulary
        # if self.share_input_output_embed and embed_tokens is not None:
            # x = F.linear(x, embed_tokens.weight)
        # elif self.embed_out is not None:
            # x = self.embed_out(x)

        if self.embed_out is not None:
            x = self.embed_out(x)
        
        if self.lm_output_learned_bias is not None and self.load_softmax:
            x = x + self.lm_output_learned_bias
            
        #finetuning
        if self.proj_out is not None:
            x = self.proj_out(x)
        


        # sentence_logits = None
        # if self.sentence_projection_layer:
        #     sentence_logits = self.sentence_projection_layer(pooled_output)
        return (x, node_output)

        # value_tensor, shape_tensor = self.tensors_encode(x, node_output)
        # return(value_tensor, shape_tensor)

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout=0.0):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.ln = LayerNorm(inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.ln(x)
        x = self.out_proj(x)
        return x

def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.0)

    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.num_segment = getattr(args, "num_segment", 2)

    args.sentence_class_num = getattr(args, "sentence_class_num", 2)
    args.sent_loss = getattr(args, "sent_loss", False)

    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)

    args.sandwich_ln = getattr(args, "sandwich_ln", False)
    args.droppath_prob = getattr(args, "droppath_prob", 0.0)

    # add
    args.atom_loss_coeff = getattr(args, "atom_loss_coeff", 1.0)
    args.pos_loss_coeff = getattr(args, "pos_loss_coeff", 1.0)

    args.max_positions = getattr(args, "max_positions", 512)
    args.num_atoms = getattr(args, "num_atoms", 512 * 9)
    args.num_edges = getattr(args, "num_edges", 512 * 3)
    args.num_in_degree = getattr(args, "num_in_degree", 512)
    args.num_out_degree = getattr(args, "num_out_degree", 512)
    args.num_spatial = getattr(args, "num_spatial", 512)
    args.num_edge_dis = getattr(args, "num_edge_dis", 128)
    args.multi_hop_max_dist = getattr(args, "multi_hop_max_dist", 5)
    args.edge_type = getattr(args, "edge_type", "multi_hop")


def bert_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.num_segment = getattr(args, "num_segment", 2)

    args.encoder_layers = getattr(args, "encoder_layers", 12)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)

    args.sentence_class_num = getattr(args, "sentence_class_num", 2)
    args.sent_loss = getattr(args, "sent_loss", False)

    args.apply_bert_init = getattr(args, "apply_bert_init", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.sandwich_ln = getattr(args, "sandwich_ln", False)
    args.droppath_prob = getattr(args, "droppath_prob", 0.0)

    args.add_3d = getattr(args, "add_3d", False)
    args.num_3d_bias_kernel = getattr(args, "num_3d_bias_kernel", 128)
    args.no_2d = getattr(args, "no_2d", False)
    base_architecture(args)


def graphormer_base_architecture(args):
    # if args.pretrained_model_name == "pcqm4mv1_graphormer_base" or \
    #    args.pretrained_model_name == "pcqm4mv2_graphormer_base" or \
    #    args.pretrained_model_name == "pcqm4mv1_graphormer_base_for_molhiv":

    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.1)
    # else:
    #     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    #     args.encoder_layers = getattr(args, "encoder_layers", 12)
    #     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    #     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
    #     args.dropout = getattr(args, "dropout", 0.0)
    #     args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    #     args.act_dropout = getattr(args, "act_dropout", 0.1)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", True)
    args.share_encoder_input_output_embed = getattr(
            args, "share_encoder_input_output_embed", False
        )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.pre_layernorm = getattr(args, "pre_layernorm", False)
    base_architecture(args)