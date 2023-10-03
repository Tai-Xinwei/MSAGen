# -*- coding: utf-8 -*-
import datetime
import os

import lmdb
import pandas as pd
import torch
from tqdm import tqdm

# import pickle as pkl
# from transformers import AutoTokenizer


def convert():
    ckpdir = "/home/peiran/FMproj/llama2/llama-2-70b/"

    files = [file for file in os.listdir(ckpdir) if file.endswith(".pth")]
    files.sort()

    print(files)
    layerckp = {}
    embckp = {}
    normckp = {}
    lmheadckp = {}

    import json

    params = json.load(open(ckpdir + "params.json"))

    encoder_dim = params["dim"]
    n_heads = params["n_heads"]

    # base = 10000.0
    # inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    if "n_kv_heads" in params:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        key_value_dim = encoder_dim // num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        key_value_dim = encoder_dim

    vocab_size = 32000

    for file in files:
        if not file.endswith(".pth"):
            continue

        ckp = torch.load(ckpdir + file)
        print(file, ckp.keys())
        # print(ckp['rope.freqs'].shape)
        if "rope.freqs" in ckp.keys():
            rope_freqs = ckp["rope.freqs"]

        # # print(ckp.keys())
        for key in ckp.keys():
            if key == "tok_embeddings.weight":
                if "embed_tokens.weight" not in embckp.keys():
                    embckp["embed_tokens.weight"] = ckp[key]
                else:
                    cur_tensor = embckp["embed_tokens.weight"]
                    w, h = ckp[key].shape
                    if w == vocab_size:
                        new_tensor = torch.cat((cur_tensor, ckp[key]), dim=1)
                    elif h == vocab_size:
                        new_tensor = torch.cat((cur_tensor, ckp[key]), dim=0)
                    else:
                        raise ValueError(
                            "Unexpected shape {}, {} in embed_tokens".format(w, h)
                        )
                    embckp["embed_tokens.weight"] = new_tensor
            elif key == "norm.weight":
                if "norm.weight" not in normckp.keys():
                    normckp["norm.weight"] = ckp[key]
                    torch.save(normckp, ckpdir + "/model.norm.pt")
            elif key == "output.weight":
                if "lm_head.weight" not in lmheadckp.keys():
                    lmheadckp["lm_head.weight"] = ckp[key]
                else:
                    cur_tensor = lmheadckp["lm_head.weight"]
                    w, h = ckp[key].shape
                    if w == encoder_dim:
                        new_tensor = torch.cat((cur_tensor, ckp[key]), dim=1)
                    elif h == encoder_dim:
                        new_tensor = torch.cat((cur_tensor, ckp[key]), dim=0)
                    else:
                        raise ValueError(
                            "Unexpected shape {}, {} in lm_head".format(w, h)
                        )
                    lmheadckp["lm_head.weight"] = new_tensor
            elif key == "rope.freqs":
                pass
            else:
                nl = key.split(".")[1]
                ckpname = "/model.layers." + nl + ".pt"
                if ckpname not in layerckp.keys():
                    layerckp[ckpname] = {}

                name_list = key.split(".")
                name = ""
                for n in name_list[2:]:
                    name += n + "."
                name = name[:-1]

                name = name.replace("attention", "self_attn")
                name = name.replace("wq", "q_proj")
                name = name.replace("wk", "k_proj")
                name = name.replace("wv", "v_proj")
                name = name.replace("wo", "o_proj")

                name = name.replace("feed_forward", "mlp")
                name = name.replace("w1", "gate_proj")
                name = name.replace("w2", "down_proj")
                name = name.replace("w3", "up_proj")

                name = name.replace("self_attn_norm", "input_layernorm")
                name = name.replace("ffn_norm", "post_attention_layernorm")

                if name not in layerckp[ckpname].keys() or len(ckp[key].shape) == 1:
                    # if name.find("q_proj") != -1 or name.find("k_proj") != -1:
                    #     ckp[key] = permute(ckp[key])
                    layerckp[ckpname][name] = ckp[key]
                else:
                    cur_tensor = layerckp[ckpname][name]
                    w, h = ckp[key].shape
                    if w == encoder_dim:
                        new_tensor = torch.cat((cur_tensor, ckp[key]), dim=1)
                    elif h == encoder_dim:
                        new_tensor = torch.cat((cur_tensor, ckp[key]), dim=0)
                    else:
                        raise ValueError("Unexpected shape {}, {}".format(w, h))
                    layerckp[ckpname][name] = new_tensor

    def permute(w, n_heads=n_heads, dim1=encoder_dim, dim2=encoder_dim):
        return (
            w.view(n_heads, dim1 // n_heads // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    for layer in layerckp.keys():
        layerckp[layer]["self_attn.rotary_emb.inv_freq"] = rope_freqs
        for name in layerckp[layer].keys():
            if name.find("q_proj") != -1 or name.find("k_proj") != -1:
                if layerckp[layer][name].shape[0] == layerckp[layer][name].shape[1]:
                    layerckp[layer][name] = permute(layerckp[layer][name])
                else:
                    layerckp[layer][name] = permute(
                        layerckp[layer][name],
                        num_key_value_heads,
                        key_value_dim,
                        encoder_dim,
                    )

        torch.save(layerckp[layer], ckpdir + layer)

    torch.save(embckp, ckpdir + "/model.hybrid_emb.pt")
    torch.save(lmheadckp, ckpdir + "/model.lm_head.pt")


if __name__ == "__main__":
    pass
