# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from typing import Dict, List, Optional

import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from sfm.data.mol_data.moltext_dataset import batch_collater_for_graphormer
from sfm.logging.loggers import logger
from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.models.graphormer.modules import GraphormerSentenceEncoder
from sfm.models.llama2.llama_modules import LlamaEmbeddingsPP, LlamaModelPP
from sfm.models.llama2.llama_modules_3dmp import LlamaEmbeddingsMP, LlamaModelMP
from sfm.utils import PretrainedLayerSpec

from .modules.graphormer_encoder import GraphormerSentenceEncoderPP
from .modules.hybrid_emb import AdaptorConfig, HybridEmbeddings, HybridEmbeddingsPP


class GraphormerLlamaModel(torch.nn.Module):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(
        self,
        args,
        vocab_size: int,
        load_ckp: Optional[bool] = False,
    ):
        super().__init__()

        graphormer_config = GraphormerConfig(args)
        self.args = graphormer_config.args
        logger.info(self.args)

        llama_config = LlamaConfig.from_pretrained(args.llm_model_name_or_path)
        llama_config.vocab_size = vocab_size
        adaptor_config = self.init_adaptor_config(args, llama_config)

        self.pipe_layers = []
        if args.pipeline_model_parallel_size == 0:
            self.graphormer_encoder = GraphormerSentenceEncoder(graphormer_config)
            self.decoder = LlamaForCausalLM(llama_config)
            self.adaptor = HybridEmbeddings(adaptor_config)
        elif args.tensor_model_parallel_size == 1:
            load_ckpt = not args.infer
            self.pipe_layers.extend(
                [
                    PretrainedLayerSpec(
                        GraphormerSentenceEncoderPP,
                        graphormer_config,
                        load_ckpt=load_ckpt,
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
                        load_ckpt=load_ckpt,
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
                        load_ckpt=load_ckpt,
                    )
                ]
            )

            self.pipe_layers.extend(
                LlamaModelPP.to_layers(
                    args,
                    llama_config,
                    load_ckpt=load_ckpt,
                    new_num_tokens=vocab_size,
                )
            )
        else:
            load_ckpt = False  # not args.infer
            self.pipe_layers.extend(
                [
                    PretrainedLayerSpec(
                        GraphormerSentenceEncoderPP,
                        graphormer_config,
                        load_ckpt=load_ckpt,
                        pretrained_ckpt_path=args.loadmfmcheck_path,
                        lora_mode="freeze",
                    )
                ]
            )

            self.pipe_layers.extend(
                [
                    PretrainedLayerSpec(
                        LlamaEmbeddingsMP,
                        llama_config,
                        new_num_tokens=vocab_size,
                        load_ckpt=load_ckpt,
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
                        load_ckpt=load_ckpt,
                    )
                ]
            )

            self.pipe_layers.extend(
                LlamaModelMP.to_layers(
                    args,
                    llama_config,
                    load_ckpt=load_ckpt,
                    new_num_tokens=vocab_size,
                )
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
        batched_data["out_degree"] = batched_data["in_degree"]
        batched_data["attn_edge_type"] = None

        # generate text_emb
        text_embeds = self.decoder.get_input_embeddings()(
            torch.where(input_ids > 0, input_ids, 0)
        )

        if batched_data is not None:
            # generate mol emb
            mol_emb, _, _, _, mol_padding_mask = self.graphormer_encoder(
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
            mol_emb, _, _, _, mol_padding_mask = self.graphormer_encoder(
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
