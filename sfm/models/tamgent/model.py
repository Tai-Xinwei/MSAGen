# -*- coding: utf-8 -*-
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from sfm.data.tamgent2.tokenizer import MolxptTokenizer
from sfm.logging import logger
from sfm.models.tamgent.Qformer import BertConfig, BertLMHeadModel
from sfm.models.tamgent.scheduler import LinearWarmupCosineLRScheduler
from sfm.optim.adam import AdamW
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig, ModelOutput
from sfm.pipeline.accelerator.trainer import Model


@dataclass
class Tamgent2Config(DistributedTrainConfig):
    freeze_text_encoder: bool = True
    text_hidden_size: int = 4096
    llama_model: str = "/blob/shufxi/llama/7B"
    molxpt_model: str = "/blob/shufxi/molxpt"
    max_txt_len_llama: int = 500
    max_txt_len_smiles: int = 2048
    end_sym: str = "\n"
    low_resource: bool = False  # use 8 bit and put vit in cpu

    # Q-Former
    qformer_num_query_token: int = 32
    qformer_num_layer: int = 4
    qformer_cross_attention_freq: int = 2
    qformer_num_attention_heads: int = 8

    # Dataset
    train_mol_path: str = "/blob/shufxi/data/tamgent/chebi/train.textmol.smi"
    train_txt_path: str = "/blob/shufxi/data/tamgent/chebi/train.textmol.desc"
    val_mol_path: str = "/blob/shufxi/data/tamgent/chebi/val.textmol.smi"
    val_text_path: str = "/blob/shufxi/data/tamgent/chebi/val.textmol.smi"

    # TODO: shall we put this in the trainer config?
    iters_per_epoch: int = 1


class Tamgent2(Model):
    def __init__(
        self,
        args: Tamgent2Config,
    ):
        super().__init__()

        self.args = args

        self.mol_tokenizer = self.init_mol_tokenizer(args.molxpt_model)
        self.text_tokenizer = self.init_text_tokenizer(args.llama_model)

        logger.info("Loading text_encoder")
        self.init_llama()

        logger.info("Loading text_encoder Done. Loading Q-Former")
        self.init_Qformer()
        self.init_smi_decoder()

        self.freeze_text_encoder()

    def freeze_text_encoder(self):
        if self.args.freeze_text_encoder:
            for name, param in self.text_encoder.named_parameters():
                param.requires_grad = False

            self.text_encoder.eval()
            logging.info("freeze text encoder")

    def before_batch(self):
        if self.args.freeze_text_encoder:
            self.text_encoder.eval()

    def init_text_tokenizer(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def init_mol_tokenizer(self, path):
        tokenizer = MolxptTokenizer.from_pretrained(path, use_fast=False)
        return tokenizer

    def init_llama(self):
        self.text_encoder = AutoModelForCausalLM.from_pretrained(self.args.llama_model)

    def init_smi_decoder(self):
        self.smi_decoder = AutoModelForCausalLM.from_pretrained(self.args.molxpt_model)
        self.text2mol_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.smi_decoder.config.hidden_size
        )

    def init_Qformer(self):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = self.args.text_hidden_size
        encoder_config.num_hidden_layers = self.args.qformer_num_layer
        encoder_config.num_attention_heads = self.args.qformer_num_attention_heads
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = self.args.qformer_cross_attention_freq
        encoder_config.query_length = self.args.qformer_num_query_token
        # encoder_config.is_decoder = True
        print(encoder_config)
        self.Qformer = BertLMHeadModel(config=encoder_config)
        self.query_tokens = nn.Parameter(
            torch.zeros(
                1, self.args.qformer_num_query_token, encoder_config.hidden_size
            )
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

    def emb_text(self, text):
        device = self.text_encoder.device
        # print(device)
        tokens = self.text_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.args.max_txt_len_llama,
        ).to(device)

        text_embeds = (
            self.text_encoder(**tokens, output_hidden_states=True)
            .hidden_states[-1]
            .to(device)
        )
        image_atts = torch.ones(text_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(text_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=text_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        outputs_llama = self.text2mol_proj(query_output.last_hidden_state)
        atts_llama = torch.ones(outputs_llama.size()[:-1], dtype=torch.long).to(device)
        return outputs_llama, atts_llama

    def forward(self, samples):
        text = samples.text
        text_embeds, atts_text = self.emb_text(text)

        self.mol_tokenizer.padding_side = "right"
        smiles = samples.smiles
        # print(text, smiles)
        to_regress_tokens = self.mol_tokenizer(
            smiles,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.args.max_txt_len_smiles,
            add_special_tokens=True,
        ).to(self.smi_decoder.device)
        # print(self.mol_tokenizer.convert_ids_to_tokens(to_regress_tokens.input_ids[0]))
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.mol_tokenizer.pad_token_id, -100
        )
        # print(targets)
        empty_targets = (
            torch.ones([atts_text.shape[0], atts_text.shape[1] + 1], dtype=torch.long)
            .to(self.smi_decoder.device)
            .fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        # print(targets)
        batch_size = text_embeds.shape[0]
        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=to_regress_tokens.input_ids.dtype,
                device=to_regress_tokens.input_ids.device,
            )
            * self.mol_tokenizer.bos_token_id
        )
        bos_embeds = self.smi_decoder.biogpt.embed_tokens(bos)
        atts_bos = atts_text[:, :1]

        to_regress_embeds = self.smi_decoder.biogpt.embed_tokens(
            to_regress_tokens.input_ids
        )
        inputs_embeds = torch.cat([bos_embeds, text_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat(
            [atts_bos, atts_text, to_regress_tokens.attention_mask], dim=1
        )

        outputs = self.smi_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        return outputs

    def compute_loss(self, pred, batch) -> ModelOutput:
        return ModelOutput(loss=pred["loss"], log_output={})

    def config_optimizer(self):
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        logger.info("number of trainable parameters: %d" % num_parameters)
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(self.args.weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]

        optimizer = AdamW(
            optim_params,
            lr=float(self.args.init_lr),
            weight_decay=float(self.args.weight_decay),
            betas=(self.args.beta1, self.args.beta2),
        )
        max_epoch = self.args.total_num_epochs
        warmup_start_lr = self.args.warmup_lr
        warmup_epochs = self.args.warmup_num_epochs
        iters_per_epoch = self.args.iters_per_epoch
        min_lr = self.args.min_lr
        scheduler = LinearWarmupCosineLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            iters_per_epoch=iters_per_epoch,
            min_lr=min_lr,
            init_lr=self.args.init_lr,
            warmup_steps=warmup_epochs * iters_per_epoch,
            warmup_start_lr=warmup_start_lr,
        )
        return optimizer, scheduler
