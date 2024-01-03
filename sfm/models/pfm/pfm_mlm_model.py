# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sfm.data.prot_data.processed_mlm_dataset import Batch
from sfm.logging import logger
from sfm.models.llama2.llama_modules import LlamaDecoderLayer, LlamaHead, LlamaNorm
from sfm.models.pfm.pfm_mlm_config import PfmMlmConfig
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.model import Model
from sfm.utils.optim.optimizer import myAdam
from sfm.utils.optim.set_lr import DECAY_COSINE_RATE, groupWarmupDecayLR


class BpeToAA(nn.Module):
    def __init__(self, config: PfmMlmConfig):
        super().__init__()
        self.config = config
        bpe2aa_data = np.load(config.bpe2aa_path)
        self.bpe2aa = nn.Parameter(
            torch.from_numpy(bpe2aa_data["bpe_to_aa"]).long(), requires_grad=False
        )
        self.bpe_len = nn.Parameter(
            torch.from_numpy(bpe2aa_data["bpe_len"]).long(), requires_grad=False
        )
        self.aa_vocab = bpe2aa_data["aa_vocab"]
        self.max_bpe_len = self.bpe_len.max().item()

        self.pos_emb = nn.Embedding(self.max_bpe_len, config.hidden_size)

        self.len_pred_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SELU(),
            # +1 as it is 0-indexed
            nn.Linear(config.hidden_size, self.max_bpe_len + 1),
        )

        self.type_pred_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SELU(),
            nn.Linear(config.hidden_size, len(self.aa_vocab)),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(self, x: torch.Tensor):  # N, C
        len_pred = self.len_pred_head(x)  # N, L

        x_t = x.unsqueeze(1).expand(-1, self.max_bpe_len, -1)  # N, max_bpe_len, C
        pos_emb = self.pos_emb.weight.unsqueeze(0)  # 1, max_bpe_len, C
        type_pred = self.type_pred_head(x_t + pos_emb)  # N, max_bpe_len, len(aa_vocab)

        return len_pred, type_pred


class PfmMlmModel(Model):
    def __init__(self, config: PfmMlmConfig):
        super().__init__()
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)

        if config.use_aa_loss:
            self.bpe2aa = BpeToAA(config)
        else:
            self.bpe2aa = None

        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(LlamaDecoderLayer(config))

        self.layers = nn.ModuleList(layers)

        self.lm_head = LlamaHead(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            module.weight.data[self.config.pad_token_id].zero_()

    def forward(self, batch: Batch):
        input_ids = batch.x
        bsz, seq_len = input_ids.shape

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)

        inputs_embeds = self.emb(input_ids)

        # Bsz, 1, Q, KV
        attention_mask = torch.zeros(bsz, 1, seq_len, seq_len, device=input_ids.device)
        padding_mask = (
            batch.pad_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, seq_len, -1)
        )

        attention_mask[~padding_mask] = torch.finfo(inputs_embeds.dtype).min

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        hidden_states = hidden_states[batch.mask]

        lm_logits = self.lm_head((hidden_states,))[0]

        if self.config.use_aa_loss:
            aa_len_pred, aa_type_pred = self.bpe2aa(hidden_states)
            return lm_logits, aa_len_pred, aa_type_pred
        else:
            return lm_logits, None, None

    def compute_loss(self, pred: Tuple[torch.Tensor], batch: Batch):
        selected_logits, aa_len_pred, aa_type_pred = pred
        selected_labels = batch.y[batch.mask]

        mlm_loss = F.cross_entropy(selected_logits.float(), selected_labels)
        ppl = torch.exp(mlm_loss.detach()).cpu().item()
        mlm_acc = (
            (selected_logits.argmax(dim=-1) == selected_labels)
            .float()
            .mean()
            .detach()
            .cpu()
            .item()
        )
        n_tokens = batch.y.numel() * 1.0
        n_selected_tokens = selected_labels.numel() * 1.0

        if aa_len_pred is not None and aa_type_pred is not None:
            aa_len_label = torch.index_select(self.bpe2aa.bpe_len, 0, selected_labels)
            aa_len_loss = F.cross_entropy(aa_len_pred, aa_len_label)
            aa_len_acc = (
                (aa_len_pred.argmax(dim=-1) == aa_len_label)
                .float()
                .mean()
                .detach()
                .cpu()
                .item()
            )

            aa_type_label = torch.index_select(
                self.bpe2aa.bpe2aa, 0, selected_labels
            ).reshape(-1)
            aa_type_pred = aa_type_pred.reshape(-1, len(self.bpe2aa.aa_vocab))

            valid_label = aa_type_label != -1
            aa_type_label = aa_type_label[valid_label]
            aa_type_pred = aa_type_pred[valid_label]

            aa_type_loss = F.cross_entropy(aa_type_pred, aa_type_label)
            aa_type_acc = (
                (aa_type_pred.argmax(dim=-1) == aa_type_label)
                .float()
                .mean()
                .detach()
                .cpu()
                .item()
            )
        else:
            aa_len_loss = torch.zeros_like(mlm_loss)
            aa_len_acc = 0.0
            aa_type_loss = torch.zeros_like(mlm_loss)
            aa_type_acc = 0.0

        loss = mlm_loss + aa_len_loss + aa_type_loss

        return ModelOutput(
            loss,
            num_examples=len(batch.x),
            log_output={
                "acc": mlm_acc,
                "ppl": ppl,
                "tokens": n_tokens,
                "selected_tokens": n_selected_tokens,
                "mlm_loss": mlm_loss.detach().cpu().item(),
                "aa_len_loss": aa_len_loss.detach().cpu().item(),
                "aa_type_loss": aa_type_loss.detach().cpu().item(),
                "aa_len_acc": aa_len_acc,
                "aa_type_acc": aa_type_acc,
            },
        )

    def config_optimizer(self, model: Module = None) -> Tuple[Optimizer, LRScheduler]:
        if model is None:
            model = self

        optimizer, _ = myAdam(
            model,
            lr=self.config.max_lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
            eps=1e-8,
        )

        lr_scheduler = groupWarmupDecayLR(
            optimizer,
            total_num_steps=self.config.total_num_steps,
            warmup_max_lr=self.config.max_lr,
            warmup_num_steps=self.config.warmup_num_steps,
            decay_type=DECAY_COSINE_RATE,
        )
        return (optimizer, lr_scheduler)


class LlamaDecoderLayerRd(LlamaDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        res: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
        rd_scale: Optional[float] = 1.0,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        # Self Attention
        residual = hidden_states
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        res = res + hidden_states * rd_scale
        hidden_states = residual + hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        res = res + hidden_states * rd_scale
        hidden_states = residual + hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        return (hidden_states, res)


class PfmMlmModelRd(PfmMlmModel):
    def __init__(self, config: PfmMlmConfig):
        super().__init__(config)
        for i in range(config.num_hidden_layers):
            self.layers[i] = LlamaDecoderLayerRd(config)

        self.final_norm = LlamaNorm(config)
        self.apply(self._init_weights)

    def forward(self, batch: Batch):
        input_ids = batch.x
        bsz, seq_len = input_ids.shape

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)

        inputs_embeds = self.emb(input_ids)

        # Bsz, 1, Q, KV
        attention_mask = torch.zeros(bsz, 1, seq_len, seq_len, device=input_ids.device)
        padding_mask = (
            batch.pad_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, seq_len, -1)
        )

        attention_mask[~padding_mask] = torch.finfo(inputs_embeds.dtype).min

        hidden_states = inputs_embeds
        res = hidden_states * self.config.rd_scale

        for layer in self.layers:
            hidden_states, res = layer(
                hidden_states=hidden_states,
                res=res,
                rd_scale=self.config.rd_scale,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        res = self.final_norm((res, None, None))[0]
        hidden_states = hidden_states + res

        hidden_states = hidden_states[batch.mask]
        lm_logits = self.lm_head((hidden_states,))[0]

        if self.config.use_aa_loss:
            aa_len_pred, aa_type_pred = self.bpe2aa(hidden_states)
            return lm_logits, aa_len_pred, aa_type_pred
        else:
            return lm_logits, None, None


from sfm.models.pfm.pfm_mlm_config import (
    PfmMlmConfig,
    pfm_mlm_tiny_config,
    pfm_mlm_tiny_h24_config,
)
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli

config_registry = {
    "pfm_mlm_tiny": pfm_mlm_tiny_config,
    "pfm_mlm_tiny_h24": pfm_mlm_tiny_h24_config,
}


class PfmMlmBpeModel(Model):
    def __init__(self, args):
        super().__init__()
        config = arg_utils.from_args(args, PfmMlmConfig)
        config = config_registry.get(config.model_type, pfm_mlm_tiny_config)(config)

        if config.use_rd:
            self.model = PfmMlmModelRd(config)
        else:
            self.model = PfmMlmModel(config)

    def forward(self, batch):
        data = Batch(
            x=batch["x"],
            y=batch["y"],
            # This is to select the output tokens, only logits with True will in output
            mask=batch["mask"],
            # This is to ignore the padding tokens, i.e., (x != self.pad_idx)
            pad_mask=batch["pad_mask"],
        )

        return self.model(data)[0]

    def ft_forward(self, batch):
        data = Batch(
            x=batch["x"],
            y=batch["y"],
            # This is to select the output tokens, only logits with True will in output
            mask=batch["mask"],
            # This is to ignore the padding tokens, i.e., (x != self.pad_idx)
            pad_mask=batch["pad_mask"],
        )

        return self.model(data)[0]

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        pass

    def config_optimizer(self):
        pass

    def load_pretrained_weights(self, args, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.
        """
        if args.ft or args.infer:
            checkpoints_state = torch.load(checkpoint_path, map_location="cpu")
            if "model" in checkpoints_state:
                checkpoints_state = checkpoints_state["model"]
            elif "module" in checkpoints_state:
                checkpoints_state = checkpoints_state["module"]

            IncompatibleKeys = self.model.load_state_dict(
                checkpoints_state, strict=False
            )
            IncompatibleKeys = IncompatibleKeys._asdict()

            missing_keys = []
            for keys in IncompatibleKeys["missing_keys"]:
                if keys.find("dummy") == -1:
                    missing_keys.append(keys)

            unexpected_keys = []
            for keys in IncompatibleKeys["unexpected_keys"]:
                if keys.find("dummy") == -1:
                    unexpected_keys.append(keys)

            if len(missing_keys) > 0:
                logger.info(
                    "Missing keys in {}: {}".format(
                        checkpoint_path,
                        missing_keys,
                    )
                )

            if len(unexpected_keys) > 0:
                logger.info(
                    "Unexpected keys {}: {}".format(
                        checkpoint_path,
                        unexpected_keys,
                    )
                )
