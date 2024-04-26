# -*- coding: utf-8 -*-
import json
import math
import os
import sys
from dataclasses import asdict

import numpy as np
import torch
from ase import Atoms
from ase.io import write
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MaxLengthCriteria, StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig

from sfm.data.threedimargen_data.dataset import MODE, ThreeDimARGenDataset
from sfm.data.threedimargen_data.tokenizer import ThreeDimARGenTokenizer
from sfm.logging import logger
from sfm.models.threedimargen.threedimargen_config import (
    ThreeDimARGenConfig,
    ThreeDimARGenInferenceConfig,
)
from sfm.models.threedimargen.threedimargenlan import ThreeDimARGenLanModel
from sfm.tasks.threedimargen.train_threedimargenlan import config_registry
from sfm.utils import arg_utils
from sfm.utils.cli_utils import cli
from sfm.utils.move_to_device import move_to_device


def get_ele_num(sent):
    ret = {}
    sent_list = sent.split()
    i = 0
    while sent_list[i] != "<gen>":
        i += 1
    while i < len(sent_list):
        if sent_list[i] not in ret:
            ret[sent_list[i]] = 1
        else:
            ret[sent_list[i]] += 1
        i += 1
    return ret


def convert_to_ascii(fname):
    with open(fname, "r") as f:
        content = f.read()
    with open(fname, "w", encoding="ascii") as f:
        f.write(content)


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
    # saved_config = config_registry[saved_config.model_type](saved_config)

    saved_config.update(asdict(inference_config))

    tokenizer = ThreeDimARGenTokenizer.from_file(config.dict_path, saved_config)
    # saved_config.vocab_size = len(tokenizer)
    # saved_config.pad_token_id = tokenizer.padding_idx
    # saved_config.mask_token_id = tokenizer.mask_idx

    logger.info(saved_config)

    scale_coords = getattr(saved_config, "scale_coords", None)

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
        dataset = ThreeDimARGenDataset(
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
                        sent, lattice, atom_coordinates = ret[i]
                        if inference_config.verbose:
                            print(sent)
                        dataset.data[index]["prediction"] = {
                            "lattice": lattice,
                            "coordinates": atom_coordinates,
                        }
                        fw.write(
                            json.dumps(dataset.data[index], default=convert) + "\n"
                        )
                        index += 1
    else:
        if inference_config.output_file is None or inference_config.output_file == "":
            fw = sys.stdout
        else:
            fw = open(inference_config.output_file, "w")
        while True:
            line = input("Enter a formula: ")  # Na1 Cl1
            formula = line.strip()
            if formula == "":
                break

            line = input("Enter the space group (defualt 1):")  # 225
            space_group = line.strip()
            if space_group == "":
                space_group = "1"
            space_group_tok = f"<sgn>{space_group}"

            tokens = [
                tokenizer.get_idx(tok)
                for tok in tokenizer.tokenize(
                    formula, prepend_bos=True, append_gen=False, append_eos=False
                )
            ]
            tokens.append(tokenizer.sp_idx)
            space_group_token = tokenizer.get_idx(space_group_tok)
            tokens.append(space_group_token)
            # tokens.append(tokenizer.gen_idx)

            # 0 for formula
            # coordinates_mask = [0 for _ in range(len(tokens))]
            # 1 for coordinates
            # coordinates_mask.extend([1 for _ in range((len(tokens) - 3 + 3))])
            # for <eos>
            # coordinates_mask.append(0)

            tokens = torch.tensor(tokens).unsqueeze(0)

            with torch.no_grad():
                tokens = move_to_device(tokens, "cuda")
                ret = model.net.generate(
                    input_ids=tokens,
                    generation_config=gen_config,
                )
            sent = ret.sequences[0].cpu().numpy()
            decode_result = tokenizer.decode(
                sent,
                scale_coords,
            )
            sent, lattice, atom_coordinates = decode_result
            print(sent, file=fw)

            # Define a crystal structure using ASE's Atoms object
            # encoded_formula = tokenizer.tokenize(
            #     formula, prepend_bos=False, append_gen=False, append_eos=False
            # )
            # structure = Atoms(
            #     encoded_formula, scaled_positions=atom_coordinates, cell=lattice
            # )

            # Save the structure to a file in the .cif format, which can be read by VESTA
            # write("structure.cif", structure, format="cif")
            # convert_to_ascii("structure.cif")


if __name__ == "__main__":
    main()
