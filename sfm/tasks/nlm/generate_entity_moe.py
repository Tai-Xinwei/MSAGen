# -*- coding: utf-8 -*-
import json
import os
import re
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch
from moe_infinity import MoE
from tqdm import tqdm

from sfm.data.sci_data.NlmTokenizer import NlmTokenizer

prefix_re = re.compile(r" ?\<[a-z]\>")


def download_and_convert_ckpt(mixtral_blob_path, nlm_blob_path, local_path):
    os.makedirs(local_path, exist_ok=True)
    bar = tqdm(total=35)

    tensor_index = {"metadata": {"total_size": 0}, "weight_map": {}}

    # input emb
    bar.set_description("input emb")
    ckpt_old = torch.load(
        os.path.join(nlm_blob_path, "layer_00-model_states.pt"), map_location="cpu"
    )
    ckpt_new_name = "model_00.safetensors"
    emb_weight = ckpt_old["embed_tokens.weight"]
    ckpt_new = {"model.embed_tokens.weight": emb_weight}

    tensor_index["metadata"]["total_size"] += emb_weight.numel()
    tensor_index["weight_map"]["model.embed_tokens.weight"] = ckpt_new_name
    st.save_file(ckpt_new, os.path.join(local_path, ckpt_new_name))
    bar.update(1)

    # layer 1 to 32
    for i in range(0, 32):
        bar.set_description(f"layer {i+1}")
        ckpt_old = torch.load(
            os.path.join(nlm_blob_path, f"layer_{i+1:02d}-model_states.pt"),
            map_location="cpu",
        )
        ckpt_new_name = f"model_{i+1:02d}.safetensors"
        ckpt_new = {}

        # Attn QKVO proj
        ckpt_new[f"model.layers.{i}.self_attn.q_proj.weight"] = ckpt_old[
            "self_attn.q_proj.weight"
        ]
        ckpt_new[f"model.layers.{i}.self_attn.k_proj.weight"] = ckpt_old[
            "self_attn.k_proj.weight"
        ]
        ckpt_new[f"model.layers.{i}.self_attn.v_proj.weight"] = ckpt_old[
            "self_attn.v_proj.weight"
        ]
        ckpt_new[f"model.layers.{i}.self_attn.o_proj.weight"] = ckpt_old[
            "self_attn.o_proj.weight"
        ]

        # MoE
        for j in range(8):
            ckpt_new[
                f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight"
            ] = ckpt_old[f"block_sparse_moe.experts.{j}.w1.weight"]
            ckpt_new[
                f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight"
            ] = ckpt_old[f"block_sparse_moe.experts.{j}.w2.weight"]
            ckpt_new[
                f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight"
            ] = ckpt_old[f"block_sparse_moe.experts.{j}.w3.weight"]
        ckpt_new[f"model.layers.{i}.block_sparse_moe.gate.weight"] = ckpt_old[
            "block_sparse_moe.gate.weight"
        ]

        # LN
        ckpt_new[f"model.layers.{i}.input_layernorm.weight"] = ckpt_old[
            "input_layernorm.weight"
        ]
        ckpt_new[f"model.layers.{i}.post_attention_layernorm.weight"] = ckpt_old[
            "post_attention_layernorm.weight"
        ]

        for k, v in ckpt_new.items():
            tensor_index["metadata"]["total_size"] += v.numel()
            tensor_index["weight_map"][k] = ckpt_new_name

        st.save_file(ckpt_new, os.path.join(local_path, ckpt_new_name))
        bar.update(1)

    # Final norm
    bar.set_description("final norm")
    ckpt_old = torch.load(
        os.path.join(nlm_blob_path, "layer_33-model_states.pt"), map_location="cpu"
    )
    ckpt_new_name = "model_33.safetensors"
    emb_weight = ckpt_old["norm.weight"]
    ckpt_new = {"model.norm.weight": emb_weight}

    tensor_index["metadata"]["total_size"] += emb_weight.numel()
    tensor_index["weight_map"]["model.norm.weight"] = ckpt_new_name
    st.save_file(ckpt_new, os.path.join(local_path, ckpt_new_name))
    bar.update(1)

    # LM head
    bar.set_description("LM head")
    ckpt_old = torch.load(
        os.path.join(nlm_blob_path, "layer_34-model_states.pt"), map_location="cpu"
    )
    ckpt_new_name = "model_34.safetensors"
    emb_weight = ckpt_old["lm_head.weight"]
    ckpt_new = {"lm_head.weight": emb_weight}

    tensor_index["metadata"]["total_size"] += emb_weight.numel()
    tensor_index["weight_map"]["lm_head.weight"] = ckpt_new_name
    st.save_file(ckpt_new, os.path.join(local_path, ckpt_new_name))
    bar.update(1)

    with open(os.path.join(local_path, "model.safetensors.index.json"), "w") as f:
        json.dump(tensor_index, f, indent=2)

    print(f"Maped {tensor_index['metadata']['total_size']} tensors")

    # Other config files
    config = json.load(open(os.path.join(mixtral_blob_path, "config.json")))
    config["vocab_size"] = 33982
    with open(os.path.join(local_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for file in [
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ]:
        shutil.copyfile(
            os.path.join(mixtral_blob_path, file), os.path.join(local_path, file)
        )

    # show file list in local_path
    print("Files in local_path:")
    for root, dirs, files in os.walk(local_path):
        for file in files:
            print(os.path.relpath(os.path.join(root, file), local_path))
    print("Done")
    bar.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("--mixtral_path", type=str, required=True)
    parser.add_argument("--nlm_path", type=str, required=True)
    parser.add_argument("--local_path", type=str, default="/tmp/nlm")
    parser.add_argument("--offload_path", type=str, default="/tmp/moe-infinity")
    parser.add_argument("--output_path", type=str, default="/tmp/moe-infinity")
    parser.add_argument("--device_memory_ratio", type=float, default=0.75)
    parser.add_argument("--n_seq", type=int, default=125)
    parser.add_argument("--entity", type=str, default="protein")

    args = parser.parse_args()
    print(args)

    tokenizer = NlmTokenizer.from_pretrained(args.mixtral_path)
    print("vocab size", len(tokenizer))

    download_and_convert_ckpt(args.mixtral_path, args.nlm_path, args.local_path)

    config = {
        "offload_path": args.offload_path,
        "device_memory_ratio": args.device_memory_ratio,
    }

    model = MoE(args.local_path, config)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_text = f"<{args.entity}>"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()
    output_file = open(output_path, "w")
    printed = False
    for _ in tqdm(range(args.n_seq), mininterval=10):
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=300,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1,
            )
            output_text = tokenizer.decode(outputs[0])
            if not printed:
                print(f"raw text: {output_text}")
            begin_idx = output_text.find(f"<{args.entity}>") + len(f"<{args.entity}>")
            end_idx = output_text.find(f"</{args.entity}>")

            entity_text = output_text[begin_idx:end_idx]
            entity_text = prefix_re.sub("", entity_text).strip()
            if not printed:
                print(f"entity text: {entity_text}")
            output_file.write(entity_text + "\n")
            printed = True

    output_file.close()
    print(f"Output file is saved at {output_path}")


if __name__ == "__main__":
    main()
