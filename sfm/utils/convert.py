# -*- coding: utf-8 -*-
import json
from argparse import ArgumentParser

import torch
from tqdm import tqdm


def convert(in_dir, out_dir):
    total_size = 0
    index_map = {"weight_map": {}}
    for i in tqdm(range(85)):
        new_model_states = {}
        ckpt_path = f"{in_dir}/layer_{i:02d}-model_states.pt"
        model_states = torch.load(ckpt_path, map_location="cpu")
        all_keys = list(model_states.keys())
        for key in all_keys:
            if key.find("dummy") != -1:
                continue
            weight = model_states[key]
            if i == 0:
                # molecular model
                new_key = "graphormer_encoder." + key
            elif i == 1:
                # embed tokens
                new_key = "decoder.model." + key
            elif i == 2:
                # hybrid embedding
                new_key = "adaptor." + key
            elif i < 83:
                new_key = f"decoder.model.layers.{i - 3}." + key
            elif i == 83:
                new_key = "decoder.model." + key
            else:
                new_key = "decoder." + key
            index_map["weight_map"][new_key] = f"layer_{i:02d}-model_states.bin"
            total_size += weight.nelement() * weight.element_size()
            new_model_states[new_key] = weight
        torch.save(new_model_states, f"{out_dir}/layer_{i:02d}-model_states.bin")

    index_map["total_size"] = total_size

    with open(f"{out_dir}/pytorch_model.bin.index.json", "w") as out_file:
        json.dump(index_map, out_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("in_dir", type=str, help="input model path")
    arg_parser.add_argument("out_dir", type=str, help="output model path")
    args = arg_parser.parse_args()
    convert(args.in_dir, args.out_dir)
