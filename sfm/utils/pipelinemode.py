# -*- coding: utf-8 -*-
import inspect
import os
import sys
from dataclasses import is_dataclass, replace
from functools import partial, wraps
from typing import List, Optional, Tuple, Union, final

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from sfm.logging import logger


def check_grad_requirements(output, index_or_key=0):
    def check_grad_hook(index_or_key, tensor, grad):
        if grad is None:
            logger.error(
                f"Float tensors must have gradients, the index/key and shape of the tensor are {index_or_key} and {tensor.shape}"
            )
            raise ValueError("Float tensors must have gradients")

    def check_tensor(index_or_key, tensor):
        if tensor.is_floating_point():
            if not tensor.requires_grad:
                pass
                # logger.error(
                # f"Float tensors must have gradients, the shape of the tensor is {tensor.shape}"
                # )
                # raise ValueError("Float tensors must have gradients")
            else:
                hooked_check_grad = partial(check_grad_hook, index_or_key, tensor)
                hook_handle = tensor.register_hook(hooked_check_grad)
                hook_handle.remove()

        elif tensor.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        ):
            if tensor.requires_grad:
                logger.error(
                    f"Int and bool tensors must not have gradients, the index/key and shape of the tensor are {index_or_key} and {tensor.shape}"
                )
                raise ValueError("Int and bool tensors must not have gradients")
        else:
            logger.error(f"Unsupported tensor dtype: {tensor.dtype}")
            raise TypeError(f"Unsupported tensor dtype: {tensor.dtype}")

    if isinstance(output, torch.Tensor):
        check_tensor(index_or_key, output)
        index_or_key += 1
    elif isinstance(output, (list, tuple)):
        for item in output:
            index_or_key = check_grad_requirements(item, index_or_key)
    elif isinstance(output, dict):
        for value in output.values():
            index_or_key = check_grad_requirements(value, index_or_key)
    elif is_dataclass(output):
        for value in output.__dict__.values():
            index_or_key = check_grad_requirements(value, index_or_key)

    return index_or_key


def convert2list(output):
    result = []

    if isinstance(output, (int, bool, float)):
        result.append(torch.tensor(output))
    elif isinstance(output, (list, tuple)):
        for item in output:
            result.extend(convert2list(item))
    elif isinstance(output, dict):
        for value in output.values():
            result.extend(convert2list(value))
    elif is_dataclass(output):
        for value in output.__dict__.values():
            result.extend(convert2list(value))
    elif isinstance(output, torch.Tensor):
        result.append(output)
    else:
        raise TypeError(f"Unsupported type: {type(output)}")

    return result


def tuple2input(input_tuple, param_dict):
    input_data = []
    i = 0
    for name, param in param_dict.items():
        if param == torch.Tensor:
            input_data.append(input_tuple[i])
            i += 1
        elif isinstance(param, dict):
            converted_dict = {}
            for k, v in param.items():
                converted_dict[k] = input_tuple[i]
                i += 1
            input_data.append(converted_dict)
        elif is_dataclass(param):
            dataclass_fields = param.__dataclass_fields__
            field_values = {}
            for field_name, field_type in dataclass_fields.items():
                field_values[field_name] = input_tuple[i]
                i += 1
            input_data.append(param(**field_values))
        else:
            input_data.append(input_tuple[i])
            i += 1
    return input_data


def pipemode(forward_func):
    # Check the output tensor grad requirementså
    # TODO: auto convert tuple input to wrapped function's input
    @wraps(forward_func)
    @final  ## forbid overriding from child class
    def wrapper(*input_tuple, **kwargs):
        # Get the parameter names and types from the original forward function
        signature = inspect.signature(forward_func)
        param_names = list(signature.parameters.keys())
        {name: param.annotation for name, param in signature.parameters.items()}

        if "self" in param_names:
            param_names.remove("self")
            param_self = input_tuple[0]
            param_dict = param_self.param_dict
            input_tuple = input_tuple[1:]
        else:
            raise ValueError("The first parameter must be self")

        # Convert input tuple to input of original forward function by matching parameter types
        input_data = tuple2input(*input_tuple, param_dict)
        # Call original forward function with input dictionary

        output = forward_func(param_self, *input_data, **kwargs)

        # Convert output dictionary to tuple
        output_list = convert2list(output)

        check_grad_requirements(output_list)

        return output_list

    return wrapper


def pipemodegradcheck(forward_func):
    # Check the output tensor grad requirementså
    @wraps(forward_func)
    @final  ## forbid overriding from child class
    def wrapper(*input_tuple, **kwargs):
        output = forward_func(*input_tuple, **kwargs)

        # check grad requirements
        check_grad_requirements(output)

        return output

    return wrapper


class LlamaDecoderLayerTest(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, l: int):
        super().__init__(config)
        self.config = config
        self.l = l
        self.dummy = nn.Linear(1, 1)

        self.param_dict = {
            "batched_data": {
                "hidden_states": torch.Tensor,
                "attention_mask": torch.Tensor,
                "position_ids": torch.Tensor,
            },
        }

        # self.param_dict = {
        #     "hidden_states": torch.Tensor,
        #     "attention_mask": torch.Tensor,
        #     "position_ids": torch.Tensor,
        # }

    @pipemode
    def tensor_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_bool: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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

        attention_mask = torch.zeros_like(
            attention_mask_bool, dtype=hidden_states.dtype, device=hidden_states.device
        )
        attention_mask.masked_fill_(
            ~attention_mask_bool, torch.finfo(hidden_states.dtype).min
        )

        hidden_states = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]

        return hidden_states, attention_mask_bool, position_ids

    @pipemode
    def dict_forward(
        self,
        batched_data: dict,
    ) -> Tuple[torch.FloatTensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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

        hidden_states = batched_data["hidden_states"]
        attention_mask_bool = batched_data["attention_mask"]
        position_ids = batched_data["position_ids"]

        attention_mask = torch.zeros_like(
            attention_mask_bool, dtype=hidden_states.dtype, device=hidden_states.device
        )
        attention_mask.masked_fill_(
            ~attention_mask_bool, torch.finfo(hidden_states.dtype).min
        )

        hidden_states = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]

        return batched_data


if __name__ == "__main__":
    config = LlamaConfig.from_pretrained("sfm/models/llama2/llama-2-7B/config.json")
    decoder = LlamaDecoderLayerTest(config, 0)
    hidden_states = torch.randn((1, 20, config.hidden_size))
    attention_mask = torch.ones(1, 1, 20, 20, dtype=torch.bool)
    position_ids = torch.arange(20, dtype=torch.long).unsqueeze(0)

    tuple_input = (hidden_states, attention_mask, position_ids)

    print(hidden_states.shape, attention_mask.shape, position_ids.shape)

    # ### test Tensor input
    # output = decoder.tensor_forward(tuple_input)
    # print(type(output))
    # for item in output:
    #     print(item.shape)

    ### test dict input
    dict_output = decoder.dict_forward(tuple_input)
    print(type(dict_output))
    for k in dict_output:
        print(k.shape)
