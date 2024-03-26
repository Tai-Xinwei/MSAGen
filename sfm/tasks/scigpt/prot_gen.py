# -*- coding: utf-8 -*-
import os
import random
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from sfm.logging import logger


def show_ckpt(name, ckpt):
    for k, v in ckpt.items():
        if "dummy" not in k:
            print(name, k, v.shape)


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_home", type=str)
    parser.add_argument("--tokenizer_home", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--max_count", type=int, default=250)
    parser.add_argument("--t", type=float)
    parser.add_argument("--p", type=float)

    args = parser.parse_args()

    # ensure output path exists
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # load model
    tokenizer = SFMDecTokenizer.from_pretrained(
        args.tokenizer_home,
        prot_spm_path="/blob/shufxi/data/scigpt/ur50bpe/bpe",
        dna_spm_path="/blob/shufxi/data/scigpt/dnabpe/bpe",
    )

    model = AutoModelForCausalLM.from_pretrained(args.tokenizer_home)
    model_dict = model.state_dict()
    ckpt_dict = {}
    ckpt_home = args.ckpt_home
    layer0 = torch.load(
        os.path.join(ckpt_home, "layer_00-model_states.pt"),
        map_location=torch.device("cpu"),
    )
    ckpt_dict["model.embed_tokens.weight"] = layer0["embed_tokens.weight"]
    show_ckpt("layer0", layer0)

    for l in range(0, 32):
        l_index = str(l + 1).zfill(2)
        layer = torch.load(
            os.path.join(ckpt_home, f"layer_{l_index}-model_states.pt"),
            map_location=torch.device("cpu"),
        )
        show_ckpt(l_index, layer)
        for k in layer:
            if "dummy" in k or "rotary_emb" in k:
                continue
            ckpt_dict[f"model.layers.{l}.{k}"] = layer[k]
    layer = torch.load(
        os.path.join(ckpt_home, "layer_33-model_states.pt"),
        map_location=torch.device("cpu"),
    )
    show_ckpt(33, layer)
    ckpt_dict["model.norm.weight"] = layer["norm.weight"]

    layer = torch.load(
        os.path.join(ckpt_home, "layer_34-model_states.pt"),
        map_location=torch.device("cpu"),
    )
    show_ckpt(33, layer)
    ckpt_dict["lm_head.weight"] = layer["lm_head.weight"]
    model_dict.update(ckpt_dict)

    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(model_dict)

    model = model.cuda()

    logger.info(f"Loaded model from {args.ckpt_home}")
    # generate
    input_ids = tokenizer("<protein>", return_tensors="pt").input_ids.cuda()
    t = args.t
    p = args.p

    with open(output_path / f"t{t}_p{p}.txt", "w") as f:
        for _ in tqdm(range(args.max_count), desc=f"t={t}, p={p}"):
            output = model.generate(
                input_ids,
                do_sample=True,
                top_p=p,
                temperature=t,
                max_length=100,
                num_return_sequences=1,
                repetition_penalty=1.2,
            )
            output = output[0]
            s = tokenizer.decode(output)
            prot = s[len("<s> <protein>") : s.find("</protein>")]
            prot = prot.replace("<a>", "").replace(" ", "")
            f.write(prot + "\n")


if __name__ == "__main__":
    main()
