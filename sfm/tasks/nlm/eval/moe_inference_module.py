# -*- coding: utf-8 -*-
import json
import os
import re
import shutil
from argparse import ArgumentParser
from pathlib import Path

import click
import safetensors.torch as st
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from sfm.data.sci_data.NlmTokenizer import NlmTokenizer

prefix_re = re.compile(r" ?\<[a-z]\>")


def download_and_convert_ckpt(
    mixtral_blob_path: str, nlm_blob_path: str, local_path: str
) -> None:
    os.makedirs(local_path, exist_ok=True)
    bar = tqdm(total=35)

    metadata = {"format": "pt"}
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
    st.save_file(ckpt_new, os.path.join(local_path, ckpt_new_name), metadata=metadata)
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

        st.save_file(
            ckpt_new, os.path.join(local_path, ckpt_new_name), metadata=metadata
        )
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
    st.save_file(ckpt_new, os.path.join(local_path, ckpt_new_name), metadata=metadata)
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
    st.save_file(ckpt_new, os.path.join(local_path, ckpt_new_name), metadata=metadata)
    bar.update(1)

    with open(os.path.join(local_path, "model.safetensors.index.json"), "w") as f:
        json.dump(tensor_index, f, indent=2)

    print(f"Maped {tensor_index['metadata']['total_size']} tensors")

    # Other config files
    for file in [
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ]:
        shutil.copyfile(
            os.path.join(mixtral_blob_path, file), os.path.join(local_path, file)
        )

    config = json.load(open(os.path.join(mixtral_blob_path, "config.json")))
    config["vocab_size"] = 38078
    with open(os.path.join(local_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # show file list in local_path
    print("Files in local_path:")
    for root, dirs, files in os.walk(local_path):
        for file in files:
            print(os.path.relpath(os.path.join(root, file), local_path))
    print("Done")
    bar.close()


class SFMMoEGenerator:
    def __init__(
        self, mixtral_path: str, mixtral_blob_path: str, nlm_path: str, local_path: str
    ) -> None:
        """
        SFMGenerator class is used to generate responses for the given input string.
        """
        tokenizer = NlmTokenizer.from_pretrained(mixtral_path)
        print("vocab size", len(tokenizer))
        if (
            os.path.exists(local_path)
            and os.path.isdir(local_path)
            and len(os.listdir(local_path)) > 0
        ):
            CKPT_READY = True
        else:
            CKPT_READY = False
        if not CKPT_READY:
            download_and_convert_ckpt(mixtral_blob_path, nlm_path, local_path)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0
        )
        model = AutoModelForCausalLM.from_pretrained(
            local_path, quantization_config=quantization_config
        )
        model.eval()

        self.model = model
        self.tokenizer = tokenizer

    def chat(self, input_str, response_only=True, do_sample=False, **kwargs):
        prompt = f"Instruction: {input_str.strip()}\n\n\nResponse:"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        if "max_new_tokens" in kwargs:
            max_new_tokens = kwargs.pop("max_new_tokens")
        else:
            max_new_tokens = 100
        pad_token_id = self.tokenizer.pad_token_id
        if do_sample:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                num_return_sequences=2,
                do_sample=True,
                temperature=0.75,
                top_p=0.95,
                pad_token_id=pad_token_id,
                **kwargs,
            )
        else:
            outputs = self.model.generate(
                input_ids,
                num_beams=4,
                max_new_tokens=max_new_tokens,
                num_return_sequences=4,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                **kwargs,
            )

        out_list = []
        for out in outputs:
            s = self.tokenizer.decode(out)
            if response_only:
                segs = s.split("Response:")
                s = segs[1].strip()
            segs = s.split("</s>")
            out_list.append(segs[0].strip())
        return out_list

    def extract_first_token_prob(self, input_str):
        prompt = f"Instruction: {input_str.strip()}\n\n\nResponse:"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        model_inputs = self.model.prepare_inputs_for_generation(input_ids)
        outputs = self.model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        yes_id = self.tokenizer("Yes")["input_ids"][-1]
        no_id = self.tokenizer("No")["input_ids"][-1]
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = F.softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)
        yes_prob = next_token_scores[0, yes_id].item()
        no_prob = next_token_scores[0, no_id].item()
        confidence = yes_prob / (yes_prob + no_prob)
        return confidence


@click.command()
@click.option(
    "--mixtral_path",
    type=str,
    default="mistralai/Mixtral-8x7B-Instruct-v0.1",
    help="mixtral path",
)
@click.option(
    "--mixtral_blob_path",
    type=str,
    default="/nlm/Mixtral-8x7B-v0.1",
    help="mixtral blob path",
)
@click.option(
    "--nlm_path",
    type=str,
    default="/nlm/shufxi/nlm/8x7b/inst/20240903153854/global_step2585",
    help="nlm path",
)
@click.option("--local_path", type=str, default="./cache/nlm_moe", help="local path")
def unit_test(
    mixtral_path: str, mixtral_blob_path: str, nlm_path: str, local_path: str
) -> None:
    generator = SFMMoEGenerator(mixtral_path, mixtral_blob_path, nlm_path, local_path)
    # No
    prompt1 = (
        "Can <mol>N1(C)CC(N)=[NH+][C@](C)(c2cccc(NC(=O)c3ccc(Cl)cn3)c2)C1=O</mol>"
        + " restrain beta-secretase 1?"
    )
    # Yes
    prompt2 = "Can the molecule <mol>c1(-c2ccc3nc(N)ccc3c2)ccsc1</mol> impede beta-secretase 1?"

    prompt3 = "What can you tell me about <mol>CCCCCCCC/C=C\\CCCCCCCCC(=O)O</mol>?"

    prompt4 = "Describe <mol>O=P(O)(O)O[C@H]1[C@H](OP(=O)(O)O)[C@@H](OP(=O)(O)O)[C@@H](O)[C@@H](O)[C@@H]1OP(=O)(O)O</mol>."

    prompts = [prompt1, prompt2, prompt3, prompt4]

    print("###" * 10)
    print("Showing the prompts")
    for prompt in prompts:
        print(prompt)
    print("###" * 10)
    print("Testing out the chat function of SFMMoEGenerator")
    for prompt in prompts:
        output_list = generator.chat(prompt, do_sample=True)
        print(output_list)
    print("###" * 10)
    print("Testing out the extract_first_token_prob function of SFMMoEGenerator")
    for prompt in prompts:
        confidence = generator.extract_first_token_prob(prompt)
        print(confidence)


if __name__ == "__main__":
    unit_test()
