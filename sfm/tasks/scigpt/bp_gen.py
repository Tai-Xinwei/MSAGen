# -*- coding: utf-8 -*-
import json
import os
import random
import urllib.request
from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from sfm.logging import logger

prompt_template = "The SMILES for a molecule boiling at {:.2f} Celsius is <mol>"

lower_bond = 35
upper_bond = 65


def query_boiling_point(smiles):
    # Request data goes here
    data = {"smiles": smiles}
    body = str.encode(json.dumps(data))
    url = "<endpoint-url>"

    # TODO: Replace the <fill-in-api-key> with the API key for the endpoint
    api_key = "<fill-in-api-key>"

    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # The azureml-model-deployment header will force the request to go to a specific deployment.
    # Remove this header to have the request observe the endpoint traffic rules
    headers = {
        "Content-Type": "application/json",
        "Authorization": ("Bearer " + api_key),
        "azureml-model-deployment": "teal",
    }
    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        json_data = json.loads(result)
        return json_data["predictions"]

    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", "ignore"))
        return []


def show_ckpt(name, ckpt):
    for k, v in ckpt.items():
        if "dummy" not in k:
            print(name, k, v.shape)


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_home", type=str)
    parser.add_argument("--tokenizer_home", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--max_mol", type=int, default=1000000)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--num_return_sequences", type=int, default=10)

    args = parser.parse_args()

    # check if output file exists, if not, create it
    if not os.path.exists(args.output_path):
        open(args.output_path, "w").close()

    # resume from last output
    generated_mols = dict()
    with open(args.output_path, "r") as f:
        for line in f:
            items = line.strip().split(",")
            generated_mols[items[0]] = float(items[1])

    logger.info(f"Loaded {args.output_path} with {len(generated_mols)} lines")
    random.seed(len(generated_mols))

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

    while len(generated_mols) < args.max_mol:
        prompt = prompt_template.format(random.uniform(40, 50))
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        output = model.generate(
            input_ids,
            do_sample=True,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
        )

        new_gen_smiles = []
        for s in output:
            s = tokenizer.decode(s)
            mol = s[s.rfind("<mol>") + len("<mol>") : s.rfind("</mol>")]
            mol = mol.replace("<m>", "").replace(" ", "")

            if mol in generated_mols:
                continue
            else:
                new_gen_smiles.append(mol)
        if len(new_gen_smiles) == 0:
            continue

        bp = query_boiling_point(new_gen_smiles)
        if not bp:
            continue
        cnt = 0
        with open(args.output_path, "a") as f:
            for i in range(len(new_gen_smiles)):
                if lower_bond <= bp[i] <= upper_bond:
                    mol = new_gen_smiles[i]
                    if mol not in generated_mols:
                        generated_mols[new_gen_smiles[i]] = bp[i]
                        f.write(f"{new_gen_smiles[i]},{bp[i]}\n")
                        cnt += 1

        logger.info(
            f"Generated {len(new_gen_smiles)} new molecules, {cnt} in [{lower_bond}, {upper_bond}], total {len(generated_mols)}"
        )


if __name__ == "__main__":
    main()
