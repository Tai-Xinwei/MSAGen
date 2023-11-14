# -*- coding: utf-8 -*-
import math
import os
import sys

import torch
from ase import Atoms
from ase.io import write
from transformers import MaxLengthCriteria, StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig

from sfm.data.threedimargen_data.tokenizer import ThreeDimTokenizer
from sfm.logging import logger
from sfm.models.threedimargen.threedimargen import ThreeDimARGenModel
from sfm.models.threedimargen.threedimargen_config import (
    ThreeDimARGenConfig,
    ThreeDimARGenInferenceConfig,
)
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
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


@cli(DistributedTrainConfig, ThreeDimARGenConfig, ThreeDimARGenInferenceConfig)
def main(args):
    checkpoints_state = torch.load(args.loadcheck_path, map_location="cpu")
    saved_args = checkpoints_state["args"]

    digit_scale = getattr(saved_args, "scale_digit", None)

    model = ThreeDimARGenModel(saved_args)
    model.eval()
    # model.half()

    logger.info(f"Loading model from {args.loadcheck_path}")

    model.load_pretrained_weights(args, args.loadcheck_path)
    model.cuda()

    tokenizer = ThreeDimTokenizer.from_file(args.dict_path)

    if args.output_file == "":
        output_file = sys.stdout
    else:
        output_file = open(args.output_file, "w")

    gen_config = GenerationConfig(
        pad_token_id=tokenizer.padding_idx,
        eos_token_id=tokenizer.eos_idx,
        use_cache=True,
        max_length=2048,
        return_dict_in_generate=True,
    )
    model.net.config.mask_token_id = tokenizer.mask_idx

    while True:
        line = input("Enter a formula: ")
        formula = line.strip()
        if formula == "":
            break

        line = input("Enter the space group (defualt 1):")
        space_group = line.strip()
        if space_group == "":
            space_group = "1"

        tokens = [
            tokenizer.get_idx(tok)
            for tok in tokenizer.tokenize(
                formula, prepend_bos=True, append_gen=False, append_eos=False
            )
        ]
        space_group_token = tokenizer.get_idx(space_group)
        tokens.append(space_group_token)
        tokens.append(tokenizer.gen_idx)

        # 0 for formula
        coordinates_mask = [0 for _ in range(len(tokens))]
        # 1 for coordinates
        coordinates_mask.extend([1 for _ in range((len(tokens) - 3 + 3))])
        # for <eos>
        coordinates_mask.append(0)

        tokens = torch.tensor(tokens).unsqueeze(0)
        coordinates_mask = torch.tensor(coordinates_mask).unsqueeze(0)

        with torch.no_grad():
            tokens = move_to_device(tokens, "cuda")
            coordinates_mask = move_to_device(coordinates_mask, "cuda")
            ret = model.net.generate(
                input_ids=tokens,
                coordinates_mask=coordinates_mask,
                generation_config=gen_config,
            )
        sent = ret.sequences[0].cpu().numpy()
        coordinates = ret.coordinates.cpu().numpy()
        mask = coordinates_mask[0].cpu().numpy()
        sent = tokenizer.decode(sent, coordinates, mask, digit_scale)
        print(sent, file=output_file)

        lattice = coordinates[:3]
        positions = coordinates[3:]
        if digit_scale is not None:
            positions = positions / digit_scale
        # Define a crystal structure using ASE's Atoms object
        encoded_formula = tokenizer.tokenize(
            formula, prepend_bos=False, append_gen=False, append_eos=False
        )
        structure = Atoms(encoded_formula, scaled_positions=positions, cell=lattice)

        # Save the structure to a file in the .cif format, which can be read by VESTA
        write("structure.cif", structure, format="cif")
        convert_to_ascii("structure.cif")


if __name__ == "__main__":
    main()
