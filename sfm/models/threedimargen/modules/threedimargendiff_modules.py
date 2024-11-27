# -*- coding: utf-8 -*-
import os
import warnings
from dataclasses import dataclass
from math import sqrt
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from pyexpat import model
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers import (
    LlamaConfig,
    LlamaPreTrainedModel,
    LlamaTokenizer,
    LlamaTokenizerFast,
)
from transformers.activations import ACT2FN
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GreedySearchOutput
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
)
from transformers.utils import ModelOutput

from sfm.logging import logger
from sfm.models.llama2.llama_modules import (
    LlamaDecoderLayerPP,
    LlamaEmbeddingsBase,
    LlamaHead,
    LlamaNorm,
)
from sfm.models.psm.modules.diffusion import DPM
from sfm.utils import PretrainedLayerSpec, TiedPretrainedLayerSpec
from sfm.utils.pipelinemode import pipemode


@dataclass
class ThreeDimARGenDiffOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_log: Optional[dict] = None
    logits: Optional[torch.FloatTensor] = None
    coordinates: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    num_examples: Optional[int] = None
    label: Optional[torch.LongTensor] = None
    log_output: Optional[dict] = None


@dataclass
class ThreeDimARGenDiffGreedySearchOutput(ModelOutput):
    sequences: torch.LongTensor = None
    coordinates: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class CrystalCriterions(nn.Module):
    def __init__(self, vocab_size, reduction="mean") -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.word_loss = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.05)
        self.pos_loss = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        model_output,
        batch_data,
    ):
        word_logits = model_output.logits
        bs = word_logits.shape[0]
        y_0 = model_output.coordinates
        # note that y_0 has already been shifted and in the desired shape
        # shift so that tokens < n predict n
        label_ids = batch_data["label_ids"]
        label_coordinates = batch_data["label_coordinates"]
        coordinates_mask = batch_data["coordinates_mask"]

        shift_label_ids = label_ids[..., 1:].contiguous()
        shift_label_coordinates_mask = coordinates_mask[..., 1:].contiguous()
        shift_word_logits = word_logits[:, :-1, :].contiguous()

        shift_word_logits = shift_word_logits[~shift_label_coordinates_mask.bool()]

        # Calculate loss on word tokens
        shift_words_labels = shift_label_ids[~shift_label_coordinates_mask.bool()]
        loss_words = self.word_loss(
            shift_word_logits.view(-1, self.vocab_size),
            shift_words_labels.view(-1),
        )

        # Calculate loss on coordinate tokens
        if label_coordinates.dtype != y_0.dtype:
            label_coordinates = label_coordinates.to(y_0.dtype)
        loss_coord = self.pos_loss(y_0, label_coordinates)
        # Combine losses
        loss = loss_words + loss_coord
        loss_log = {
            "loss": loss.item() if loss is not None else None,
            "loss_words": loss_words.item() if loss_words is not None else None,
            "loss_y_0": loss_coord.item() if loss_coord is not None else None,
        }

        model_output.loss = loss
        model_output.num_examples = bs
        model_output.log_output = loss_log
        return model_output


class ProteinMAEDistCriterions(nn.Module):
    def __init__(self, args, reduction="mean") -> None:
        super().__init__()
        self.loss_type = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.05)
        self.loss_pos = nn.L1Loss(reduction="mean")
        self.loss_angle = nn.MSELoss(reduction="mean")
        self.loss_dist = nn.MSELoss(reduction="mean")
        self.args = args
        self.num_aa_type = args.num_residues
        self.diffmode = args.diffmode

    def forward(
        self,
        batch_data,
        output_dict,
    ):
        # """----------------------type loss----------------------"""
        logits = output_dict["x"]
        mask_aa = output_dict["mask_aa"]
        if mask_aa.any():
            with torch.no_grad():
                aa_seq = batch_data["x"]
                aa_seq = aa_seq[mask_aa.squeeze(-1).bool()]

            logits = logits[:, :, :][mask_aa.squeeze(-1).bool()]

            type_loss = (
                self.loss_type(
                    logits.view(-1, logits.size(-1)).to(torch.float32),
                    aa_seq.view(-1),
                )
                * self.args.atom_loss_coeff
            )

            type_acc = (
                (logits.view(-1, logits.size(-1)).argmax(dim=-1) == aa_seq)
                .to(torch.float32)
                .mean()
            )
        else:
            type_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            type_acc = 0.0

        """----------------------pos loss----------------------"""
        pred_ca_pos = output_dict["pos_output"]
        pos_ca_epsilon = output_dict["pos_epsilon"][:, :, 1, :]
        output_dict["pos_sigma"]
        ori_ca_pos = output_dict["ori_pos"][:, :, 1, :]

        mask_pos = output_dict["mask_pos"]
        pos_mask = output_dict["pos_mask"][:, :, 1, :]
        padding_mask = output_dict["padding_mask"]
        pos_unified_mask = (
            (~padding_mask.unsqueeze(-1)) & pos_mask & mask_pos.squeeze(-1)
        ).squeeze(-1)

        if pos_unified_mask.any():
            if self.diffmode == "x0":
                pos_loss = self.loss_pos(
                    ori_ca_pos[pos_unified_mask.expand(-1, -1, 3)].to(torch.float32),
                    pred_ca_pos[pos_unified_mask.expand(-1, -1, 3)].to(torch.float32),
                )
            elif self.diffmode == "epsilon":
                pos_loss = self.loss_pos(
                    pos_ca_epsilon[pos_unified_mask].to(torch.float32),
                    pred_ca_pos[pos_unified_mask].to(torch.float32),
                )
        else:
            pos_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = pos_loss + type_loss

        return loss, {
            "total_loss": loss,
            "loss_type": type_loss,
            # "loss_dist": dist_loss,
            # "loss_angle": angle_loss,
            "loss_pos": pos_loss,
            "type_acc": type_acc,
        }

    @torch.compile
    def _set_dist_mask(self, residue_seq):
        """
        compute the mask for distance loss in the complete doc mode
        """
        B, L = residue_seq.shape
        pair_mask_aa = torch.zeros(
            (B, L, L), device=residue_seq.device, dtype=torch.int8
        )
        for i in range(B):
            mask_start_idx = (residue_seq[i] == 0).nonzero(as_tuple=True)[0]
            mask_end_idx = (residue_seq[i] == 2).nonzero(as_tuple=True)[0]

            for j in range(len(mask_end_idx)):
                s_idx = mask_start_idx[j]
                e_idx = mask_end_idx[j]
                pair_mask_aa[i, s_idx:e_idx, s_idx:e_idx] = 1.0

            if len(mask_start_idx) > len(mask_end_idx):
                s_idx = mask_start_idx[-1]
                pair_mask_aa[i, s_idx:, s_idx:] = 1.0

        return pair_mask_aa.bool()


class DiffNoise(nn.Module):
    def __init__(self, config):
        super(DiffNoise, self).__init__()
        self.config = config

        assert config.ddpm_schedule in ["linear", "quadratic", "sigmoid", "cosine"]
        (
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
            self.alphas_cumprod,
            self.beta_list,
        ) = self._beta_schedule(
            config.num_timesteps + 1,
            config.ddpm_beta_start,
            config.ddpm_beta_end,
            config.ddpm_schedule,
        )
        self.unit_noise_scale = config.diffusion_noise_std

    def _beta_schedule(
        self, num_timesteps, beta_start, beta_end, schedule_type="sigmoid"
    ):
        if schedule_type == "linear":
            beta_list = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "quadratic":
            beta_list = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
            )
        elif schedule_type == "sigmoid":
            betas = torch.linspace(-6, 6, num_timesteps)
            beta_list = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        elif schedule_type == "cosine":
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            beta_list = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise NotImplementedError("only support linear, quadratic, sigmoid, cosine")

        alphas = 1 - beta_list
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        return (
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            alphas_cumprod,
            beta_list,
        )

    def _extract(self, a, t, x_shape):
        if len(t.shape) == 1:
            batch_size = t.shape[0]
            out = a.gather(-1, t.cpu().long())
            return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        elif len(t.shape) == 2:
            batch_size, L = t.shape
            # a is in shape of [num_timesteps], t is in shape of [batch_size, L],
            out = torch.gather(a.unsqueeze(0).expand(batch_size, -1), 1, t.cpu().long())
            return out.reshape(batch_size, L, *((1,) * (len(x_shape) - 2))).to(t.device)
        else:
            raise Exception(f"t shape: {t.shape} not supported")

    def get_noise(self, pos):
        noise = torch.randn_like(pos, dtype=pos.dtype) * self.unit_noise_scale
        # for cell corner nodes (the noise is centered so that the noised cell is centered at the original point)
        return noise

    def get_sampling_start(self, init_pos):
        noise = self.get_noise(init_pos)
        return init_pos + noise

    def noise_sample(
        self,
        x_start,
        t,
        x_init=None,
        clean_mask: Optional[torch.Tensor] = None,
    ):
        t = (t * self.config.num_timesteps).long()
        noise = self.get_noise(x_start)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod.to(x_start.dtype), t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod.to(x_start.dtype), t, x_start.shape
        )

        if x_init is None:
            x_t = (
                sqrt_alphas_cumprod_t * x_start
                + sqrt_one_minus_alphas_cumprod_t * noise
            )
        else:
            x_t = (
                sqrt_alphas_cumprod_t * (x_start - x_init)
                + sqrt_one_minus_alphas_cumprod_t * noise
                + x_init
            )

        if clean_mask is not None:
            if len(clean_mask.shape) == 1:
                x_t = torch.where(clean_mask.unsqueeze(-1).unsqueeze(-1), x_start, x_t)
            elif len(clean_mask.shape) == 2:
                x_t = torch.where(clean_mask.unsqueeze(-1), x_start, x_t)
            else:
                raise ValueError(
                    f"clean_mask should be [B] or [B, L] tensor, but it's shape is {clean_mask.shape}"
                )

        return x_t, noise, sqrt_one_minus_alphas_cumprod_t


class TimeStepSampler:
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps

    def sample(self, n_graph, device, dtype, clean_sample_ratio: float = 0.0):
        time_step = torch.rand(size=(n_graph // 2 + 1,), device=device)
        time_step = torch.cat([time_step, 1.0 - time_step], dim=0)[:n_graph]
        time_step = time_step.to(dtype=dtype)
        clean_mask = torch.tensor(
            np.random.rand(n_graph) <= clean_sample_ratio,
            dtype=torch.bool,
            device=device,
        )
        return time_step, clean_mask

    def get_continuous_time_step(self, t, n_graph, device, dtype):
        time_step = torch.zeros(
            [
                n_graph,
            ],
            device=device,
            dtype=dtype,
        )
        time_step = time_step.fill_(t * 1.0 / self.num_timesteps)
        return time_step


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DiffusionModel(nn.Module):
    def __init__(
        self,
        y_dim,
        t_dim,
        hidden_features,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(DiffusionModel, self).__init__()
        self.proj_y = nn.Linear(y_dim, hidden_features, bias=False)
        self.proj_t = nn.Linear(t_dim, hidden_features, bias=False)
        self.proj_y_hat = nn.Linear(y_dim, hidden_features, bias=False)
        self.fc1 = nn.Linear(hidden_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, y_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, y, t, y_hat):
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        y = self.proj_y(y)
        t = self.proj_t(t)
        y_hat = self.proj_y_hat(y_hat)
        x = y + t + y_hat
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ThreeDimARGenDiff(LlamaForCausalLM):
    """
    3D Auto-regressive generator with diffusion loss.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.coordinate_encoder = MLP(3, config.hidden_size, config.hidden_size)
        self.coordinate_decoder = MLP(config.hidden_size, config.hidden_size, 3)
        self.diffnoise = DiffNoise(config)
        self.time_sampler = TimeStepSampler(config.num_timesteps)
        self.diffusion_model = DiffusionModel(3, 1, config.hidden_size)
        self.diffusion_process = DPM(self.diffnoise.alphas_cumprod, config)
        # self.lattice_encoder = NumMLP(3, config.hidden_size)
        # self.lattice_decoder = NumMLP(config.hidden_size, 3)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_coordinates: torch.FloatTensor = None,
        coordinates_mask: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ntokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, ThreeDimARGenDiffOutputWithPast]:
        r"""
        Args:
            label_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]`, or -100 (see `input_ids` docstring), or coordinates (x, y, z). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]` and coordinates.

        Returns:

        Example:

        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            words_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds = words_embeds
            if input_coordinates is not None:
                if input_coordinates.dtype != words_embeds.dtype:
                    input_coordinates = input_coordinates.to(words_embeds.dtype)
                coordinates_embeds = self.coordinate_encoder(input_coordinates)
                inputs_embeds[coordinates_mask.bool()] = coordinates_embeds

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # pass the hidden states to lm_head and coordinate_decoder to get logits and coordinates
        # get word logits using lm head
        word_logits = self.lm_head(hidden_states)
        # get coordinates using coordinate decoder
        coordinates = self.coordinate_decoder(hidden_states)

        # difussion process
        if self.training or coordinates.shape[1] != 1:
            # shift so that tokens < n predict n
            y = input_coordinates
            shift_coordinates_mask = coordinates_mask[:, 1:].contiguous()
            y_hat = coordinates[:, :-1, :].contiguous()
            y_hat = y_hat[shift_coordinates_mask.bool()]
            t = self.time_sampler.sample(y.size(0), y.device, y.dtype)[0]
            noisy_y, noise, _ = self.diffnoise.noise_sample(y, t)
            y_0 = self.diffusion_model(noisy_y, t, y_hat)
            # note that noisy_y has been shifted by 1 with the shape of [B, L-1, 3]

        else:
            # noise = self.diffusion.get_noise()
            y_0 = None

        # word_logits = word_logits.float()
        # coordinates = coordinates.float()

        if not return_dict:
            output = (word_logits, coordinates, y_0) + outputs[1:]
            return output

        return ThreeDimARGenDiffOutputWithPast(
            logits=word_logits,
            coordinates=coordinates,
            y_0=y_0,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def sample(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        input_coordinates: Optional[torch.FloatTensor] = None,
        coordinates_mask: Optional[torch.LongTensor] = None,
        output_coordinates_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            words_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds = words_embeds
            if input_coordinates is not None:
                if input_coordinates.dtype != words_embeds.dtype:
                    input_coordinates = input_coordinates.to(words_embeds.dtype)
                #
                coordinates = input_coordinates[coordinates_mask.bool()]
                coordinates_embeds = self.coordinate_encoder(coordinates)
                inputs_embeds[coordinates_mask.bool()] = coordinates_embeds

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # get word logits using lm head
        word_logits = self.lm_head(hidden_states)
        # get coordinates using coordinate decoder
        coordinates = self.coordinate_decoder(hidden_states)

        y_hat = coordinates[:, -1:, :].contiguous()
        # y_hat = y_hat[output_coordinates_mask[:, -1:].contiguous().bool()]
        # y_0 = y_hat.detach()
        y_t = (
            torch.randn(
                y_hat.size(),
                device=y_hat.device,
                dtype=y_hat.dtype,
                generator=torch.Generator(device=y_hat.device),
            )
            * 1.0
        )
        # y_t = input_coordinates[:, -1:, :].contiguous()

        for t in range(
            self.config.num_timesteps - 1, -1, self.config.num_timesteps_stepsize
        ):
            time_step = self.time_sampler.get_continuous_time_step(
                t, y_hat.shape[0], y_hat.device, y_hat.dtype
            )
            time_step = time_step.unsqueeze(-1).repeat(1, y_hat.shape[1])
            # sqrt_one_minus_alphas_cumprod_t = self.diffusion._extract(
            #     self.diffusion.sqrt_one_minus_alphas_cumprod.to(input_ids.dtype),
            #     (time_step * self.config.num_timesteps).long(),
            #     input_ids.shape,
            # )
            # difussion process
            y_theta = self.diffusion_model(y_t, time_step, y_hat.detach())
            epsilon = self.diffnoise.get_noise(y_t)
            y_t = self.diffusion_process.sample_step(
                y_t,
                y_hat,
                y_theta,
                epsilon,
                t,
                -self.config.num_timesteps_stepsize,
            )

        if not return_dict:
            output = (word_logits, coordinates, y_t) + outputs[1:]
            return output

        return ThreeDimARGenDiffOutputWithPast(
            logits=word_logits,
            coordinates=y_t,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        mask_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.generation_config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.generation_config.eos_token_id
        )
        mask_token_id = (
            mask_token_id if mask_token_id is not None else self.config.mask_token_id
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = (
            torch.tensor(eos_token_id).to(input_ids.device)
            if eos_token_id is not None
            else None
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init input coordinates
        input_coordinates = torch.empty(
            (input_ids.shape[0], input_ids.shape[1], 3),
            dtype=torch.float32,
            device=input_ids.device,
        )
        if model_kwargs.get("input_coordinates", None) is not None:
            input_coordinates_mask = model_kwargs["coordinates_mask"][
                :, : input_ids.shape[1]
            ]
            input_coordinates[input_coordinates_mask.bool()] = model_kwargs[
                "input_coordinates"
            ]
            del model_kwargs["input_coordinates"]
        original_coordinates_mask = model_kwargs["coordinates_mask"].clone()

        # init output coordinates
        # output_coordinates = torch.empty(
        #     (input_ids.shape[0], 0, 3), dtype=torch.float32, device=input_ids.device
        # )

        # init attention / hidden states / scores tuples
        scores = None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0
                ).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, input_coordinates, **model_kwargs
            )

            # forward pass to get next token
            outputs = self.sample(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            coordinates_mask = model_inputs["output_coordinates_mask"]
            next_token_logits = outputs.logits[:, -1, :]
            next_coordinates = outputs.coordinates[:, -1, :]
            next_coordinates_mask = coordinates_mask[:, -1]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_word = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_word = next_word * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            next_tokens = (
                next_word * (1 - next_coordinates_mask.bool().long())
                + mask_token_id * next_coordinates_mask.bool().long()
            )

            # output_coordinates = torch.cat(
            #     [output_coordinates, next_coordinates[:, None]], dim=1
            # )
            # next_coordinates = next_coordinates[next_coordinates_mask.bool()]
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            input_coordinates = torch.cat(
                [input_coordinates, next_coordinates[:, None]], dim=1
            )

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            # as stopping criteria checks the last dimension as the length, we pass the first value of the last dimension of the 3-d tensor
            if all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        input_coordinates = input_coordinates[original_coordinates_mask.bool()]

        if return_dict_in_generate:
            return ThreeDimARGenDiffGreedySearchOutput(
                sequences=input_ids,
                coordinates=input_coordinates,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
        else:
            return (input_ids, input_coordinates)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_coordinates=None,
        coordinates_mask=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        seq_length = input_ids.shape[1]
        input_coordinates_mask = coordinates_mask[:, :seq_length]
        output_coordinates_mask = coordinates_mask[:, 1 : seq_length + 1]
        if past_key_values:
            input_ids = input_ids[:, -1:]
            input_coordinates_mask = input_coordinates_mask[:, -1:]
            output_coordinates_mask = output_coordinates_mask[:, -1:]
            input_coordinates = input_coordinates[
                :, -1:
            ]  # [input_coordinates_mask.bool()]
        else:
            input_coordinates = input_coordinates  # [input_coordinates_mask.bool()]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "input_coordinates": input_coordinates,
                "coordinates_mask": input_coordinates_mask,
                "output_coordinates_mask": output_coordinates_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
