# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.models.llama import LlamaForCausalLM, LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig

from sfm.logging.loggers import logger

# from sfm.models.llama2.llama2mp_config import MPLlamaConfig
from sfm.models.llama2.llama_modules import LlamaEmbeddingsPP, LlamaModelPP, NumMLP
from sfm.models.pfm.modules.pfm_encoder import PFMEncoder
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.models.progpt.modules.bfm_encoder import PFMEncoderPP
from sfm.models.progpt.modules.copilotloss import (
    CopilotCriterionsNum,
    CopilotCriterionsNumPP,
    CopilotCriterionsPP,
)
from sfm.pipeline.accelerator.dataclasses import ModelOutput, TrainStrategy
from sfm.pipeline.accelerator.pipeline_module import SFMPipelineModelMixin
from sfm.utils import PretrainedLayerSpec

from .modules.hybrid_emb import AdaptorConfig, HybridEmbeddings, HybridEmbeddingsPP


class LlamaModelLora(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        LORA_R = 8
        LORA_ALPHA = 16
        LORA_DROPOUT = 0.1
        TARGET_MODULES = [
            "q_proj",
            "k_proj",
            "v_proj",
        ]

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
        )

        self.layers = nn.ModuleList(
            [get_peft_model(layer, lora_config) for layer in self.layers]
        )


class LlamaForCausalLMLora(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelLora(config)


class ProGPTModel(SFMPipelineModelMixin):
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

        pfm_config = PFMConfig(args)
        self.args = pfm_config.args
        logger.info(f"Trainer args: {args}")

        llama_config = LlamaConfig.from_pretrained(args.llm_model_name_or_path)
        llama_config.vocab_size = vocab_size
        args.load_ckpt = args.load_ckpt and not args.ifresume
        adaptor_config = self.init_adaptor_config(args, llama_config)

        self.llama_config = llama_config
        # if args.strategy == TrainStrategy.ThreeD:
        #     args.padded_vocab_size = max(args.padded_vocab_size, vocab_size)
        #     vocab_size = args.padded_vocab_size
        #     llama_config.vocab_size = vocab_size
        #     mp_config = self.init_mp_config(args, llama_config)
        #     self.llama_config = mp_config

        self.pipe_layers = []
        if (
            args.strategy != TrainStrategy.ThreeD
            and args.strategy != TrainStrategy.Pipeline
        ):
            self.pfm_encoder = PFMEncoder(pfm_config)

            if self.args.llm_lora:
                self.decoder = LlamaForCausalLMLora(llama_config)
            else:
                self.decoder = LlamaForCausalLM(llama_config)

            self.adaptor = HybridEmbeddings(adaptor_config)
            self.decoder.num_head = NumMLP(
                llama_config.hidden_size, 4 * llama_config.hidden_size, 1
            )
            self.loss = CopilotCriterionsNum(
                self.llama_config, self.llama_config.vocab_size
            )
        elif args.strategy == TrainStrategy.Pipeline:
            self.pipe_layers.append(
                PretrainedLayerSpec(
                    PFMEncoderPP,
                    pfm_config,
                    load_ckpt=args.load_ckpt,
                    pretrained_ckpt_path=args.loadbfmckpt_path,
                )
            )
            self.pipe_layers.append(
                PretrainedLayerSpec(
                    LlamaEmbeddingsPP,
                    llama_config,
                    new_num_tokens=vocab_size,
                    load_ckpt=args.load_ckpt,
                    pretrained_ckpt_path=os.path.join(
                        # args.llm_model_name_or_path, "model.hybrid_emb.pt"
                        args.llm_model_name_or_path,
                        "layer_{}-model_states.pt".format(str(0).zfill(2)),
                    ),
                    lora_mode="full",
                )
            )

            self.pipe_layers.append(
                PretrainedLayerSpec(
                    HybridEmbeddingsPP,
                    adaptor_config,
                    new_num_tokens=vocab_size,
                    load_ckpt=args.load_ckpt,
                )
            )

            self.pipe_layers.extend(
                LlamaModelPP.to_layers(
                    args,
                    llama_config,
                    load_ckpt=args.load_ckpt,
                    new_num_tokens=vocab_size,
                )
            )

            self.loss = CopilotCriterionsPP(
                self.llama_config, self.llama_config.vocab_size
            )
        else:
            raise NotImplementedError

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

        residue_seq = batched_data["proteins"]
        if residue_seq.shape[1] > 2:
            batched_data["x"] = residue_seq
            (
                prot_emb,
                _,
                _,
                _,
                _,
                prot_padding_mask,
                _,
                _,
            ) = self.pfm_encoder(batched_data)

            # mix embeddings
            inputs_embeds, _ = self.adaptor(
                prot_emb,
                prot_padding_mask,
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

    def to_layers(self):
        return self.pipe_layers

    def forward(
        self,
        batched_data: Dict,
        perturb=None,
        segment_labels=None,
    ) -> torch.Tensor:
        # generate prot_emb
        residue_seq = batched_data["proteins"]
        if residue_seq.shape[1] > 2:
            batched_data["x"] = residue_seq
            (
                prot_emb,
                _,
                _,
                _,
                _,
                prot_padding_mask,
                _,
                _,
            ) = self.pfm_encoder(batched_data)

            # generate text_emb
            text_embeds = self.decoder.get_input_embeddings()(
                torch.where(batched_data["input_ids"] > 0, batched_data["input_ids"], 0)
            )

            # mix embeddings
            inputs_embeds, position_ids = self.adaptor(
                prot_emb,
                prot_padding_mask,
                text_embeds,
                batched_data.get("llm_mask", batched_data["attention_mask"]),
                batched_data["input_ids"],
            )
        else:
            inputs_embeds = self.decoder.get_input_embeddings()(
                torch.where(batched_data["input_ids"] > 0, batched_data["input_ids"], 0)
            )
            position_ids = None

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=batched_data.get("llm_mask", batched_data["attention_mask"]),
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        logits = outputs["logits"]
        num_logits = self.num_head(outputs["hidden_states"][-1])

        return logits, num_logits

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

    # def init_mp_config(self, args, llama_config):
    #     config = MPLlamaConfig(
    #         args,
    #         vocab_size=llama_config.vocab_size,
    #         hidden_size=llama_config.hidden_size,
    #         intermediate_size=llama_config.intermediate_size,
    #         num_hidden_layers=llama_config.num_hidden_layers,
    #         num_attention_heads=llama_config.num_attention_heads,
    #         num_key_value_heads=llama_config.num_key_value_heads,
    #         hidden_act=llama_config.hidden_act,
    #         max_position_embeddings=llama_config.max_position_embeddings,
    #         initializer_range=llama_config.initializer_range,
    #         rms_norm_eps=llama_config.rms_norm_eps,
    #         use_cache=llama_config.use_cache,
    #         pad_token_id=llama_config.pad_token_id,
    #         bos_token_id=llama_config.bos_token_id,
    #         eos_token_id=llama_config.eos_token_id,
    #         pretraining_tp=llama_config.pretraining_tp,
    #         tie_word_embeddings=llama_config.tie_word_embeddings,
    #         rope_scaling=llama_config.rope_scaling,
    #         seq_length=args.seq_length,
    #         rotary_percent=args.rotary_percent,
    #     )

    #     return config

    def compute_loss(self, pred, batch) -> ModelOutput:
        loss, loss_log = self.loss(pred, batch)
        return ModelOutput(
            loss=loss,
            num_examples=pred[0].size()[0],
            log_output=loss_log,
        )

    def config_optimizer(self, model=None) -> tuple[Optimizer, LRScheduler]:
        return (None, None)
