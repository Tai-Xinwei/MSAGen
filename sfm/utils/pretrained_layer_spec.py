# -*- coding: utf-8 -*-
from typing import Dict, Optional

import torch
from deepspeed.pipe import LayerSpec
from deepspeed.utils import logger as ds_logger
from peft import LoraConfig, get_peft_model


class PretrainedLayerSpec(LayerSpec):
    def __init__(self, typename, *module_args, **module_kwargs):
        self.pretrained_ckpt_path = module_kwargs.pop("pretrained_ckpt_path", None)
        self.load_ckpt = module_kwargs.pop("load_ckpt", False)
        self.lora_mode = module_kwargs.pop("lora_mode", "full")
        self.new_num_tokens = module_kwargs.pop("new_num_tokens", None)

        super().__init__(typename, *module_args, **module_kwargs)

    def build(self, device="cpu", log=False, load=False):
        self.layer = super().build(log=log)

        if self.pretrained_ckpt_path is not None and self.load_ckpt:
            self.load_pretrained(device=device)

        # self.resize_token_embeddings(self.new_num_tokens)

        # # TODO: LORA
        # if self.lora_mode == "freeze":
        #     self.layer = self.create_peft_model(self.layer, lora=False)
        # elif self.lora_mode == "lora":
        #     self.layer = self.create_peft_model(self.layer, lora=True)

        return self.layer

    def load_pretrained(self, device="cpu"):
        # TODO: each process loads the whole model in cpu part, needs fixing.

        checkpoints_state = torch.load(self.pretrained_ckpt_path, map_location=device)
        if type(checkpoints_state) == dict and "module" in checkpoints_state:
            checkpoints_state = checkpoints_state["module"]
        elif type(checkpoints_state) == dict and "model" in checkpoints_state:
            checkpoints_state = checkpoints_state["model"]

        if self.new_num_tokens is not None:
            checkpoints_state = self.resize_token_embeddings(checkpoints_state)

        IncompatibleKeys = self.layer.load_state_dict(checkpoints_state, strict=False)
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

        ds_logger.info(f"{device} Loaded from {self.pretrained_ckpt_path}")

    # def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> None:
    #     if new_num_tokens is not None:
    #         self.layer.resize_token_embeddings(new_num_tokens)

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
