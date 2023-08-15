# -*- coding: utf-8 -*-
import os
from typing import Dict, Optional

import torch
from deepspeed.pipe import LayerSpec, TiedLayerSpec
from deepspeed.utils import logger as ds_logger
from peft import LoraConfig, get_peft_model


class PretrainedLayerSpec(LayerSpec):
    def __init__(self, typename, *module_args, **module_kwargs):
        self.pretrained_ckpt_path = module_kwargs.pop("pretrained_ckpt_path", None)
        self.load_ckpt = module_kwargs.pop("load_ckpt", False)
        self.lora_mode = module_kwargs.pop("lora_mode", "full")
        self.new_num_tokens = module_kwargs.pop("new_num_tokens", None)
        self.tp_model_size = module_kwargs.pop("tp_model_size", None)
        self.tp_rank = module_kwargs.pop("tp_rank", 0)

        super().__init__(typename, *module_args, **module_kwargs)

    def build(self, device="cpu", log=False, load=False):
        layer = super().build(log=log)

        if self.pretrained_ckpt_path is not None and self.load_ckpt and load:
            if os.path.exists(self.pretrained_ckpt_path):
                if self.tp_model_size is None or self.tp_model_size == 0:
                    self.load_pretrained(layer, device=device)
                else:
                    self.partition_load_pretrained(layer, device=device)
            else:
                ds_logger.warn(f"Checkpoint {self.pretrained_ckpt_path} is not found.")

            # # TODO: LORA
            if self.lora_mode == "lora":
                layer = self.create_peft_model(layer, lora=True)
            # elif self.lora_mode == "freeze":
            #     layer = self.create_peft_model(layer, lora=False)

        return layer

    def load_pretrained(self, layer, device="cpu"):
        checkpoints_state = torch.load(self.pretrained_ckpt_path, map_location="cpu")
        if type(checkpoints_state) == dict and "module" in checkpoints_state:
            checkpoints_state = checkpoints_state["module"]
        elif type(checkpoints_state) == dict and "model" in checkpoints_state:
            checkpoints_state = checkpoints_state["model"]

        if self.new_num_tokens is not None:
            checkpoints_state = self.resize_token_embeddings(checkpoints_state)

        IncompatibleKeys = layer.load_state_dict(checkpoints_state, strict=False)
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
            ds_logger.info(
                "{} Missing keys in {}: {}".format(
                    device,
                    self.pretrained_ckpt_path,
                    missing_keys,
                )
            )

        if len(unexpected_keys) > 0:
            ds_logger.info(
                "{} Unexpected keys {}: {}".format(
                    device,
                    self.pretrained_ckpt_path,
                    unexpected_keys,
                )
            )

        del checkpoints_state
        ds_logger.info(f"{device} Loaded from {self.pretrained_ckpt_path}")

    def partition_load_pretrained(self, layer, device="cpu"):
        checkpoints_state = torch.load(self.pretrained_ckpt_path, map_location="cpu")
        if type(checkpoints_state) == dict and "module" in checkpoints_state:
            checkpoints_state = checkpoints_state["module"]
        elif type(checkpoints_state) == dict and "model" in checkpoints_state:
            checkpoints_state = checkpoints_state["model"]

        IncompatibleKeys = layer.auto_partition_load_state_dict(
            state_dict=checkpoints_state,
            tp_model_size=self.tp_model_size,
            tp_rank=self.tp_rank,
            strict=False,
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
            ds_logger.info(
                "{} Missing keys in {}: {}".format(
                    device,
                    self.pretrained_ckpt_path,
                    missing_keys,
                )
            )

        if len(unexpected_keys) > 0:
            ds_logger.info(
                "{} Unexpected keys {}: {}".format(
                    device,
                    self.pretrained_ckpt_path,
                    unexpected_keys,
                )
            )

        del checkpoints_state
        ds_logger.info(f"{device} Loaded from {self.pretrained_ckpt_path}")

    def resize_token_embeddings(self, checkpoints_state: dict) -> dict:
        if "lm_head.weight" in checkpoints_state:
            old_head_size = checkpoints_state["lm_head.weight"].size(0)
            if old_head_size == self.new_num_tokens:
                return checkpoints_state
            elif old_head_size <= self.new_num_tokens:
                old_head_weight = checkpoints_state["lm_head.weight"]
                new_head = torch.nn.Linear(
                    old_head_weight.size(1),
                    self.new_num_tokens,
                    bias=False,
                    dtype=old_head_weight.dtype,
                    device=old_head_weight.device,
                )
                new_head.weight.data[
                    : old_head_weight.size(0), :
                ] = old_head_weight.data
                checkpoints_state["lm_head.weight"] = new_head.weight

                return checkpoints_state
            else:
                raise ValueError(
                    f"new embedding size {self.new_num_tokens} must be larger than the current one {old_head_size}"
                )

        elif "embed_tokens.weight" in checkpoints_state:
            old_embed_size = checkpoints_state["embed_tokens.weight"].size(0)
            if old_embed_size == self.new_num_tokens:
                return checkpoints_state
            elif old_embed_size <= self.new_num_tokens:
                old_embed_weight = checkpoints_state["embed_tokens.weight"]
                new_embed = torch.nn.Embedding(
                    self.new_num_tokens,
                    old_embed_weight.size(1),
                    dtype=old_embed_weight.dtype,
                    device=old_embed_weight.device,
                )
                new_embed.weight.data.normal_(mean=0.0, std=1.0)
                new_embed.weight.data[
                    : old_embed_weight.size(0), :
                ] = old_embed_weight.data
                checkpoints_state["embed_tokens.weight"] = new_embed.weight

                return checkpoints_state
            else:
                raise ValueError(
                    f"new embedding size {self.new_num_tokens} must be larger than the current one {old_embed_size}"
                )

        raise ValueError(
            "lm_head.weight and embed_tokens.weight are not found in checkpoints_state"
        )

    def create_peft_model(self, model, lora=True):
        LORA_R = 8
        LORA_ALPHA = 16
        LORA_DROPOUT = 0.1
        if lora:
            TARGET_MODULES = [
                "q_proj",
                "k_proj",
                "v_proj",
                # "down_proj",
                # "gate_proj",
                # "up_proj",
            ]
        else:
            TARGET_MODULES = ["dummy"]

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            # task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

        return model


class TiedPretrainedLayerSpec(PretrainedLayerSpec):
    def __init__(
        self,
        key,
        typename,
        *module_args,
        forward_fn=None,
        tied_weight_attr="weight",
        **module_kwargs,
    ):
        super().__init__(typename, *module_args, **module_kwargs)
        self.key = key
        self.forward_fn = forward_fn
        self.tied_weight_attr = tied_weight_attr


def load_ckp_tied_modules(net: torch.nn.Module, args, new_num_tokens, device="cpu"):
    for key in net.tied_modules.keys():
        if key == "embed_tokens":
            loadpath = os.path.join(args.llm_model_name_or_path, "model.hybrid_emb.pt")

            checkpoints_state = torch.load(loadpath, map_location=device)
            old_embed_weight = checkpoints_state["embed_tokens.weight"]
            new_embed = torch.nn.Embedding(
                new_num_tokens,
                old_embed_weight.size(1),
                dtype=old_embed_weight.dtype,
                device=old_embed_weight.device,
            )

            new_embed.weight.data.normal_(mean=0.0, std=1.0)
            new_embed.weight.data[: old_embed_weight.size(0), :] = old_embed_weight.data

            checkpoints_state["embed_tokens.weight"] = new_embed.weight

            layer = net.tied_modules[key]
            layer.load_state_dict(checkpoints_state, strict=True)

            ds_logger.info(f"{device} Loaded from {loadpath}")
