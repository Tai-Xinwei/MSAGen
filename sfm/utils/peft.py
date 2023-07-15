# -*- coding: utf-8 -*-
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training


def create_peft_model(model):
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        # "down_proj",
        # "gate_proj",
    ]

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
