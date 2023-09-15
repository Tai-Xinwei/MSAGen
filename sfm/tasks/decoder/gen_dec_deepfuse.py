# -*- coding: utf-8 -*-
import math
import sys
from dataclasses import replace

import torch
from transformers.generation.configuration_utils import GenerationConfig

from sfm.data.dec_data.datasets import (
    ENTITY_MARKERS,
    MixedTokenDataset,
    TextSpan,
    TokenType,
)
from sfm.logging import logger
from sfm.models.decoder.deepfuse.config import (
    DecDeepFuseConfig,
    DecDeepFuseInferenceConfig,
    llama2_7b_default_config,
    mix_gpt_default_config,
)
from sfm.models.decoder.deepfuse.model import DecDeepFuseModel
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli
from sfm.utils.move_to_device import move_to_device


def load_state_dict(model: DecDeepFuseModel, path: str):
    total_num_layers = len(model.decoder_layers) + 2
    ret = {}

    # load embedding
    logger.info("Loading embedding")
    state_dict = torch.load(f"{path}/layer_00-model_states.pt", map_location="cpu")
    for k, v in state_dict.items():
        ret[f"embed.{k}"] = v

    # load decoder layers
    has_inv_freq = False

    for k in model.decoder_layers[0].state_dict().keys():
        if "inv_freq" in k:
            has_inv_freq = True
            break

    for i in range(total_num_layers - 2):
        logger.info(f"Loading decoder layer {i}")

        state_dict = torch.load(
            f"{path}/layer_{i+1:02d}-model_states.pt", map_location="cpu"
        )
        for k, v in state_dict.items():
            if not has_inv_freq and "inv_freq" in k:
                continue
            ret[f"decoder_layers.{i}.{k}"] = v

    # load head
    logger.info("Loading head")
    state_dict = torch.load(
        f"{path}/layer_{total_num_layers-1:02d}-model_states.pt", map_location="cpu"
    )
    for k, v in state_dict.items():
        ret[f"head.{k}"] = v

    model.load_state_dict(ret)
    logger.info("Loading done")
    return model


@cli(DecDeepFuseConfig, DecDeepFuseInferenceConfig)
def main(args):
    config = arg_utils.from_args(args, DecDeepFuseConfig)
    config = replace(config, **llama2_7b_default_config())
    config = replace(config, **mix_gpt_default_config())

    inference_config = arg_utils.from_args(args, DecDeepFuseInferenceConfig)

    with open(inference_config.input_file, "r") as f:
        sents = f.readlines()

        # TODO: support other patterns,like text+mol => mol
        sents = [[TextSpan(sent.strip(), TokenType.Text)] for sent in sents]

    model = DecDeepFuseModel(config)
    model.eval()
    model.half()

    logger.info(f"Loading model from {inference_config.ckpt_folder}")

    model = load_state_dict(model, inference_config.ckpt_folder)
    model.cuda()

    dataset = MixedTokenDataset(
        sents=sents,
        text_tokenizer=config.llama_model,
        entity_tokenizer=config.entity_decoder_model,
        max_text_len=config.max_text_len,
        max_entity_len=config.max_entity_len,
        return_tuple=False,
        pad_left=True,
    )

    if inference_config.output_file == "":
        output_file = sys.stdout
    else:
        output_file = open(inference_config.output_file, "w")

    bsz = inference_config.decoder_batch_size
    gen_config = GenerationConfig(
        pad_token_id=dataset.text_tokenizer.pad_token_id,
        eos_token_id=dataset.text_tokenizer.eos_token_id,
        use_cache=True,
        max_length=inference_config.max_length,
        max_new_tokens=inference_config.max_new_tokens,
    )

    special_token_mapping = []
    for token in ENTITY_MARKERS:
        special_token_mapping.append(
            (
                dataset.text_tokenizer.convert_tokens_to_ids([token])[0],
                dataset.entity_tokenizer.convert_tokens_to_ids([token + "</w>"])[0],
            )
        )

    for i in range(math.ceil(len(dataset) // bsz)):
        batch = dataset[i * bsz : (i + 1) * bsz]
        with torch.no_grad():
            batch = move_to_device(batch, "cuda")
            input_ids = batch.token_seq
            ret = model.generate(
                input_ids,
                gen_config,
                batch=batch,
                special_token_mapping=special_token_mapping,
            )

        for sent in ret:
            print(sent, file=output_file)


if __name__ == "__main__":
    main()
