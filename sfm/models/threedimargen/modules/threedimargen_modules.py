# -*- coding: utf-8 -*-
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
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
from sfm.utils import PretrainedLayerSpec, TiedPretrainedLayerSpec
from sfm.utils.pipelinemode import pipemode


class ThreeDimARGenEmbeddingsPP(LlamaEmbeddingsBase):
    def __init__(self, config: LlamaConfig, learnable_cutoff: int = 0):
        super().__init__(config, learnable_cutoff=learnable_cutoff)

    def forward(
        self, input_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        input_ids, llm_mask = input_tuple
        assert llm_mask.dtype == torch.bool, "llm_mask must be of type torch.bool"

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            raise ValueError("decoder_input_ids cannot be None")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        device = input_ids.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        mol_idx_mask = input_ids < 0  # B, T

        text_embeds = self.embed_tokens(
            input_ids.masked_fill(mol_idx_mask, 0)
        )  # B, T, hidden_size

        # attention mask
        if llm_mask is None:
            llm_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=text_embeds.device,
            )

        llm_mask = self._prepare_decoder_attention_mask(
            llm_mask,
            (batch_size, seq_length),
            text_embeds,
            past_key_values_length,
            input_ids,
        )

        return text_embeds, llm_mask, position_ids


class ThreeDimARGenModelPP(LlamaPreTrainedModel):
    def __init__(self, args, config: LlamaConfig):
        super().__init__(config)
        self.dummy = nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=True
        )

        self.pipe_layer = []

    @classmethod
    def to_layers(
        cls, args, config, learnable_cutoff=0, new_num_tokens=None, load_ckpt=False
    ):
        cls.pipe_layer = []
        cls.pipe_layer.append(
            PretrainedLayerSpec(
                ThreeDimARGenEmbeddingsPP,
                config,
                learnable_cutoff=0,
                load_ckpt=args.load_ckpt,
            )
        )
        for i in range(config.num_hidden_layers):
            cls.pipe_layer.append(
                PretrainedLayerSpec(
                    LlamaDecoderLayerPP,
                    config,
                    i,
                    load_ckpt=load_ckpt,
                )
            )
        cls.pipe_layer.append(
            PretrainedLayerSpec(
                LlamaNorm,
                config,
                load_ckpt=load_ckpt,
            )
        )
        cls.pipe_layer.append(
            PretrainedLayerSpec(
                LlamaHead,
                config,
                learnable_cutoff=learnable_cutoff,
                load_ckpt=load_ckpt,
            )
        )

        return cls.pipe_layer


@dataclass
class ThreeDimARGenOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_log: Optional[dict] = None
    logits: Optional[torch.FloatTensor] = None
    coordinates: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ThreeDimARGenGreedySearchOutput(ModelOutput):
    sequences: torch.LongTensor = None
    coordinates: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class NumMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(NumMLP, self).__init__()
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


class ThreeDimARGen(LlamaForCausalLM):
    """
    3D Auto-regressive generator.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.coordinate_encoder = NumMLP(3, config.hidden_size, config.hidden_size)
        self.coordinate_decoder = NumMLP(config.hidden_size, config.hidden_size, 3)
        # self.lattice_encoder = nn.Linear(3, config.hidden_size, bias=False)
        # self.lattice_decoder = nn.Linear(config.hidden_size, 3, bias=False)
        # self.tanh = nn.Tanh()
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
        label_ids: Optional[torch.LongTensor] = None,
        label_coordinates: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ntokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, ThreeDimARGenOutputWithPast]:
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

        loss = None
        loss_log = None

        # pass the hidden states to lm_head and coordinate_decoder to get logits and coordinates
        if label_ids is not None:
            # calculate loss if label_ids is given in training / validation
            # get word logits using lm head
            word_logits = self.lm_head(hidden_states)
            # get coordinates using coordinate decoder
            coordinates = self.coordinate_decoder(hidden_states)
            word_logits = word_logits.float()
            coordinates = coordinates.float()

            # shift so that tokens < n predict n
            shift_label_ids = label_ids[..., 1:].contiguous()
            shift_coordinates_mask = coordinates_mask[..., 1:].contiguous()
            shift_word_logits = word_logits[:, :-1, :].contiguous()
            shift_coordinates = coordinates[:, :-1, :].contiguous()
            shift_word_logits = shift_word_logits[~shift_coordinates_mask.bool()]
            shift_coordinates = shift_coordinates[shift_coordinates_mask.bool()]
            # shift_hidden_states = hidden_states[..., :-1, :]
            # get word hidden states
            # word_hidden_states = shift_hidden_states[~shift_coordinates_mask.bool()]

            # get coordinates hidden states
            # coordinates_hidden_states = shift_hidden_states[
            #    shift_coordinates_mask.bool()
            # ]

            # word_logits is [batch_size * seq_length, vocab_size], coordinates is [batch_size * seq_length, 3]
            # Calculate loss on word tokens
            loss_words_fct = CrossEntropyLoss()
            shift_words_labels = shift_label_ids[~shift_coordinates_mask.bool()]
            loss_words = loss_words_fct(
                shift_word_logits.view(-1, self.config.vocab_size),
                shift_words_labels.view(-1),
            )

            # Calculate loss on coordinate tokens
            loss_coord_fct = MSELoss()
            if label_coordinates.dtype != shift_coordinates.dtype:
                label_coordinates = label_coordinates.to(coordinates.dtype)
            loss_coord = loss_coord_fct(shift_coordinates, label_coordinates)

            # Combine losses
            loss = loss_words + loss_coord
            loss_log = {
                "loss": loss.item() if loss is not None else None,
                "loss_words": loss_words.item() if loss_words is not None else None,
                "loss_coord": loss_coord.item() if loss_coord is not None else None,
            }
        else:
            # in inference mode
            word_logits = self.lm_head(hidden_states)
            coordinates = self.coordinate_decoder(hidden_states)
            # word_logits is [batch_size, seq_length, vocab_size], coordinates is [batch_size, seq_length, 3]
            word_logits = word_logits.float()
            coordinates = coordinates.float()

        if not return_dict:
            output = (word_logits, coordinates) + outputs[1:]
            return (loss, loss_log) + output if loss is not None else output

        return ThreeDimARGenOutputWithPast(
            loss=loss,
            loss_log=loss_log,
            logits=word_logits,
            coordinates=coordinates,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def greedy_search(
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
            (0, 3), dtype=torch.float32, device=input_ids.device
        )

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
            outputs = self(
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
                next_word * (1 - next_coordinates_mask)
                + mask_token_id * next_coordinates_mask
            )
            next_coordinates = next_coordinates[next_coordinates_mask.bool()]
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            input_coordinates = torch.cat([input_coordinates, next_coordinates], dim=0)

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
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return ThreeDimARGenGreedySearchOutput(
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
        output_coordinates_mask = coordinates_mask[:, 1:]
        seq_length = input_ids.shape[1]
        coordinates_mask = coordinates_mask[:, :seq_length]
        output_coordinates_mask = output_coordinates_mask[:, :seq_length]
        if past_key_values:
            input_ids = input_ids[:, -1:]
            coordinates_mask = coordinates_mask[:, -1:]
            output_coordinates_mask = output_coordinates_mask[:, -1:]
            coordinates_count = coordinates_mask.sum().item()
            if coordinates_count > 0:
                input_coordinates = input_coordinates[-coordinates_count:]
            else:
                input_coordinates = None
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
                "coordinates_mask": coordinates_mask,
                "output_coordinates_mask": output_coordinates_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
