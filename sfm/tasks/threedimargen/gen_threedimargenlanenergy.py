# -*- coding: utf-8 -*-
import json
import math
import os
import sys
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.generation.configuration_utils import GenerationConfig

from sfm.data.threedimargen_data.dataset import MODE, ThreeDimARGenEnergyDataset
from sfm.data.threedimargen_data.tokenizer import ThreeDimARGenEnergyTokenizer
from sfm.logging import logger
from sfm.models.threedimargen.threedimargen_config import (
    ThreeDimARGenConfig,
    ThreeDimARGenInferenceConfig,
)
from sfm.models.threedimargen.threedimargenlan import ThreeDimARGenLanModel
from sfm.tasks.threedimargen.train_threedimargenlanenergy import config_registry
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli
from sfm.utils.move_to_device import move_to_device


def convert(o):
    if isinstance(o, np.float32):
        return float(o)
    raise TypeError


@cli(ThreeDimARGenConfig, ThreeDimARGenInferenceConfig)
def main(args):
    config = arg_utils.from_args(args, ThreeDimARGenConfig)
    inference_config = arg_utils.from_args(args, ThreeDimARGenInferenceConfig)

    checkpoints_state = torch.load(config.loadcheck_path, map_location="cpu")
    saved_args = checkpoints_state["args"]
    saved_config = arg_utils.from_args(saved_args, ThreeDimARGenConfig)
    saved_config.tokenizer = config.tokenizer
    saved_config = config_registry[saved_config.model_type](saved_config)

    saved_config.update(asdict(inference_config))

    tokenizer = ThreeDimARGenEnergyTokenizer.from_file(config.dict_path, saved_config)
    saved_config.vocab_size = len(tokenizer)
    saved_config.pad_token_id = tokenizer.padding_idx
    saved_config.mask_token_id = tokenizer.mask_idx

    model = ThreeDimARGenLanModel(saved_config)
    model.eval()
    # model.half()

    logger.info(f"Loading model from {config.loadcheck_path}")

    model.load_pretrained_weights(config.loadcheck_path)
    model.cuda()

    gen_config = GenerationConfig(
        pad_token_id=tokenizer.padding_idx,
        eos_token_id=tokenizer.eos_idx,
        use_cache=True,
        max_length=saved_config.max_position_embeddings,
        return_dict_in_generate=True,
    )

    if inference_config.input_file is not None:
        logger.info(f"infering from {inference_config.input_file}")
        dataset = ThreeDimARGenEnergyDataset(
            tokenizer,
            inference_config.input_file,
            saved_config,
            shuffle=False,
            mode=MODE.INFER,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=inference_config.infer_batch_size,
            shuffle=False,
            collate_fn=dataset.collate,
            drop_last=False,
        )

        index = 0
        with open(inference_config.output_file, "w") as fw:
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    batch = move_to_device(batch, "cuda")
                    ret = model.net.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        generation_config=gen_config,
                    )
                    sentences = ret.sequences.cpu().numpy()
                    ret = tokenizer.decode_batch(sentences)

                    for i in range(len(ret)):
                        sent, energy = ret[i]
                        if inference_config.verbose:
                            print(sent, energy)
                        dataset.data[index]["prediction"] = {
                            "energy": energy,
                        }
                        fw.write(
                            json.dumps(dataset.data[index], default=convert) + "\n"
                        )
                        index += 1


if __name__ == "__main__":
    main()
