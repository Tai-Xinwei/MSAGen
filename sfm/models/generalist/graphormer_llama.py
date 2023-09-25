# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.models.llama import LlamaForCausalLM, LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig

from megatron.core import parallel_state
from sfm.criterions.copilotloss import CopilotCriterionsNumMP, CopilotCriterionsNumPP
from sfm.data.mol_data.moltext_dataset import batch_collater_for_graphormer
from sfm.logging.loggers import logger
from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.models.graphormer.modules import GraphormerSentenceEncoder
from sfm.models.graphormer.modules.graphormer_sentence_encoder_mp import (
    GraphormerEncoderMP,
)
from sfm.models.llama2.llama2mp_config import MPLlamaConfig
from sfm.models.llama2.llama_modules import LlamaEmbeddingsPP, LlamaModelPP
from sfm.models.llama2.llama_modules_3dmp import LlamaEmbeddingsMP, LlamaModelMP
from sfm.pipeline.accelerator.dataclasses import ModelOutput, TrainStrategy
from sfm.pipeline.accelerator.pipeline_module import SFMPipelineModelMixin
from sfm.utils import PretrainedLayerSpec
from sfm.utils.move_to_device import move_to_device

from .modules.graphormer_encoder import GraphormerSentenceEncoderPP
from .modules.hybrid_emb import AdaptorConfig, HybridEmbeddings, HybridEmbeddingsPP
from .modules.hybrid_emb_3dmp import HybridEmbeddingsMP


class LlamaModelTextMolMasked(LlamaModel):
    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    pass  # TODO: implement this


class LlamaForCausalLMTextMolMasked(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelTextMolMasked(config)


class GraphormerLlamaModel(SFMPipelineModelMixin):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(
        self,
        args,
        vocab_size: int,
        ckp_list: List[str] = [],
    ):
        super().__init__()

        graphormer_config = GraphormerConfig(args)
        self.args = graphormer_config.args
        logger.info(f"Trainer args: {args}")

        llama_config = LlamaConfig.from_pretrained(args.llm_model_name_or_path)
        llama_config.vocab_size = vocab_size
        adaptor_config = self.init_adaptor_config(args, llama_config)

        self.llama_config = llama_config
        if args.strategy == TrainStrategy.ThreeD:
            args.padded_vocab_size = max(args.padded_vocab_size, vocab_size)
            vocab_size = args.padded_vocab_size
            llama_config.vocab_size = vocab_size
            mp_config = self.init_mp_config(args, llama_config)
            self.llama_config = mp_config

        self.pipe_layers = []
        if (
            args.strategy != TrainStrategy.ThreeD
            and args.strategy != TrainStrategy.Pipeline
        ):
            self.graphormer_encoder = GraphormerSentenceEncoder(graphormer_config)
            self.decoder = LlamaForCausalLM(llama_config)
            self.adaptor = HybridEmbeddings(adaptor_config)
        elif args.strategy == TrainStrategy.Pipeline:
            self.pipe_layers.extend(
                [
                    PretrainedLayerSpec(
                        GraphormerSentenceEncoderPP,
                        graphormer_config,
                        load_ckpt=args.load_ckpt,
                        pretrained_ckpt_path=args.loadmfmcheck_path,
                        lora_mode="freeze",
                    )
                ]
            )

            self.pipe_layers.extend(
                [
                    PretrainedLayerSpec(
                        LlamaEmbeddingsPP,
                        llama_config,
                        new_num_tokens=vocab_size,
                        load_ckpt=args.load_ckpt,
                        pretrained_ckpt_path=os.path.join(
                            args.llm_model_name_or_path, "model.hybrid_emb.pt"
                        ),
                        lora_mode="full",
                    )
                ]
            )

            self.pipe_layers.extend(
                [
                    PretrainedLayerSpec(
                        HybridEmbeddingsPP,
                        adaptor_config,
                        new_num_tokens=vocab_size,
                        load_ckpt=args.load_ckpt,
                    )
                ]
            )

            self.pipe_layers.extend(
                LlamaModelPP.to_layers(
                    args,
                    llama_config,
                    load_ckpt=args.load_ckpt,
                    new_num_tokens=vocab_size,
                )
            )

            self.loss = CopilotCriterionsNumPP(
                self.llama_config, self.llama_config.vocab_size
            )
        else:
            layer_id = 0
            self.pipe_layers.extend(
                GraphormerEncoderMP.to_layers(
                    args,
                    graphormer_config,
                    mp_config,
                    layer_id=layer_id,
                    load_ckpt=args.load_ckpt,
                    ckp_list=ckp_list,
                )
            )

            self.pipe_layers.extend(
                [
                    PretrainedLayerSpec(
                        LlamaEmbeddingsMP,
                        mp_config,
                        new_num_tokens=vocab_size,
                        load_ckpt=args.load_ckpt,
                        pretrained_ckpt_path=os.path.join(
                            args.llm_model_name_or_path, "model.hybrid_emb.pt"
                        ),
                        lora_mode="full",
                        tp_model_size=args.tensor_model_parallel_size,
                        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                    )
                ]
            )
            layer_id += 1

            self.pipe_layers.extend(
                [
                    PretrainedLayerSpec(
                        HybridEmbeddingsMP,
                        adaptor_config,
                        mp_config,
                        new_num_tokens=vocab_size,
                        load_ckpt=args.load_ckpt,
                        pretrained_ckpt_path=os.path.join(
                            args.llm_model_name_or_path,
                            ckp_list[layer_id]
                            if len(ckp_list) > 0
                            else "model.hybrid_emb.pt",
                        ),
                        tp_model_size=args.tensor_model_parallel_size,
                        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                    )
                ]
            )
            layer_id += 1

            self.pipe_layers.extend(
                LlamaModelMP.to_layers(
                    args,
                    mp_config,
                    layer_id=layer_id,
                    load_ckpt=args.load_ckpt,
                    ckp_list=ckp_list,
                    new_num_tokens=vocab_size,
                )
            )

            self.loss = CopilotCriterionsNumMP(
                self.llama_config, self.llama_config.vocab_size
            )

    @torch.no_grad()
    def generate_with_smiles(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        smiles: List[str] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        batched_data = batch_collater_for_graphormer(smiles)
        if input_ids is not None:
            batched_data = move_to_device(batched_data, device=input_ids.device)
        batched_data["out_degree"] = batched_data["in_degree"]
        batched_data["attn_edge_type"] = None

        # generate text_emb
        text_embeds = self.decoder.get_input_embeddings()(
            torch.where(input_ids > 0, input_ids, 0)
        )

        if batched_data is not None:
            # generate mol emb
            mol_emb, _, _, _, _, mol_padding_mask = self.graphormer_encoder(
                batched_data,
            )

            # mix embeddings
            inputs_embeds, _ = self.adaptor(
                mol_emb,
                mol_padding_mask,
                text_embeds,
                attention_mask,
                input_ids,
            )

            outputs = self.decoder.generate(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        else:
            # text only generation
            outputs = self.decoder.generate(
                input_ids=input_ids,
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        batched_data: Optional[Dict] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        assert (
            batched_data is not None or input_ids is not None
        ), "You should supply a batched_data or input_ids or both"

        # generate text_emb
        text_embeds = self.decoder.get_input_embeddings()(
            torch.where(input_ids > 0, input_ids, 0)
        )

        if batched_data is not None:
            # generate mol emb
            mol_emb, _, _, _, _, mol_padding_mask = self.graphormer_encoder(
                batched_data,
            )

            # mix embeddings
            inputs_embeds, _ = self.adaptor(
                mol_emb,
                mol_padding_mask,
                text_embeds,
                attention_mask,
                input_ids,
            )

            outputs = self.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        else:
            # text only generation
            outputs = self.decoder.generate(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        return outputs

    def load_encoder_state_dict(
        self, graphormer_ckppth, llama_ckppth=None, strict=False
    ):
        self.graphormer_encoder.load_state_dict(
            torch.load(graphormer_ckppth), strict=strict
        )

    def to_layers(self):
        return self.pipe_layers

    def forward(
        self,
        batched_data: Dict,
        perturb=None,
        segment_labels=None,
    ) -> torch.Tensor:
        # generate mol_emb
        mol_emb, _, _, _, mol_padding_mask = self.graphormer_encoder(
            batched_data,
            segment_labels=segment_labels,
            perturb=perturb,
        )

        # generate text_emb
        text_embeds = self.decoder.get_input_embeddings()(
            torch.where(batched_data["input_ids"] > 0, batched_data["input_ids"], 0)
        )

        # mix embeddings
        inputs_embeds, position_ids = self.adaptor(
            mol_emb,
            mol_padding_mask,
            text_embeds,
            batched_data["llm_mask"],
            batched_data["input_ids"],
        )

        # decode
        logits = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=batched_data["llm_mask"],
            position_ids=position_ids,
        )[0]

        return logits

    def init_adaptor_config(self, args, llama_config):
        config = AdaptorConfig(
            vocab_size=llama_config.vocab_size,
            hidden_size=llama_config.hidden_size,
            intermediate_size=llama_config.intermediate_size,
            num_attention_heads=llama_config.num_attention_heads,
            hidden_act=llama_config.hidden_act,
            rms_norm_eps=llama_config.rms_norm_eps,
            mfm_hidden_size=args.encoder_embed_dim,
            pool_mode=args.pool_mode,
            btn_adaptor=args.btn_adaptor,
            hidden_dropout_prob=args.dropout,
            embedding_length=args.embedding_length,
        )

        return config

    def init_mp_config(self, args, llama_config):
        config = MPLlamaConfig(
            args,
            vocab_size=llama_config.vocab_size,
            hidden_size=llama_config.hidden_size,
            intermediate_size=llama_config.intermediate_size,
            num_hidden_layers=llama_config.num_hidden_layers,
            num_attention_heads=llama_config.num_attention_heads,
            num_key_value_heads=llama_config.num_key_value_heads,
            hidden_act=llama_config.hidden_act,
            max_position_embeddings=llama_config.max_position_embeddings,
            initializer_range=llama_config.initializer_range,
            rms_norm_eps=llama_config.rms_norm_eps,
            use_cache=llama_config.use_cache,
            pad_token_id=llama_config.pad_token_id,
            bos_token_id=llama_config.bos_token_id,
            eos_token_id=llama_config.eos_token_id,
            pretraining_tp=llama_config.pretraining_tp,
            tie_word_embeddings=llama_config.tie_word_embeddings,
            rope_scaling=llama_config.rope_scaling,
            seq_length=args.seq_length,
            rotary_percent=args.rotary_percent,
        )

        return config

    def compute_loss(self, pred, batch) -> ModelOutput:
        loss, loss_log = self.loss(pred, batch)
        return ModelOutput(
            loss=loss,
            num_examples=pred[0].size()[0],
            log_output=loss_log,
        )

    def config_optimizer(self, model=None) -> tuple[Optimizer, LRScheduler]:
        return (None, None)
