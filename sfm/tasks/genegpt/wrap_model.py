# -*- coding: utf-8 -*-
import os

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sfm.data.gene_data.GeneTokenizer import GeneKMerTokenizer
from sfm.models.genegpt.genegpt_config import (  # genegpt_100m_config,
    GenegptConfig,
    GenegptConfig3D,
    genegpt3D_100m_config,
    genegpt_1b_config,
)

config_registry = {
    "genegpt_100m": genegpt3D_100m_config,
    "genegpt_1b": genegpt_1b_config,
}


def save_config1b(model_type, save_path):
    config = GenegptConfig()

    config = config_registry.get(model_type, genegpt_1b_config)(config)

    config.save_pretrained(save_path)


def save_config100m(model_type, save_path):
    config = GenegptConfig3D()

    config = config_registry.get(model_type, genegpt3D_100m_config)(config)

    config.save_pretrained(save_path)


def load_model1b(config_path, ckpt_path):
    config = LlamaConfig.from_json_file(config_path)
    model = LlamaForCausalLM(config)
    model_dict = model.state_dict()
    flag = ""
    if not os.path.exists(os.path.join(ckpt_path, "layer_00-model_states.pt")):
        flag = "_00-model"
    # print(model_dict.keys())
    ckpt_dict = {}
    layer0 = torch.load(
        os.path.join(ckpt_path, f"layer_00-model{flag}_states.pt"),
        map_location=torch.device("cpu"),
    )
    print(layer0.keys())
    ckpt_dict["model.embed_tokens.weight"] = layer0["embed_tokens.weight"]

    for l in range(0, config.num_hidden_layers):
        l_index = str(l + 1).zfill(2)
        layer = torch.load(
            os.path.join(ckpt_path, f"layer_{l_index}-model{flag}_states.pt"),
            map_location=torch.device("cpu"),
        )
        for k in layer:
            if "dummy" in k or "rotary_emb" in k:
                continue
            ckpt_dict[f"model.layers.{l}.{k}"] = layer[k]
    layer = torch.load(
        os.path.join(
            ckpt_path,
            f"layer_{config.num_hidden_layers+1}-model{flag}_states.pt",
        ),
        map_location=torch.device("cpu"),
    )
    ckpt_dict["model.norm.weight"] = layer["norm.weight"]

    layer = torch.load(
        os.path.join(
            ckpt_path,
            f"layer_{config.num_hidden_layers+2}-model{flag}_states.pt",
        ),
        map_location=torch.device("cpu"),
    )
    ckpt_dict["lm_head.weight"] = layer["lm_head.weight"]
    model_dict.update(ckpt_dict)

    model.load_state_dict(model_dict)
    return model


def load_model100m(config_path, ckpt_path):
    config = LlamaConfig.from_json_file(config_path)
    model = LlamaForCausalLM(config)
    model_dict = model.state_dict()
    print(model_dict.keys())
    flag = ""
    if not os.path.exists(os.path.join(ckpt_path, "layer_00-model_states.pt")):
        flag = "_00-model"
    # print(model_dict.keys())
    ckpt_dict = {}
    layer0 = torch.load(
        os.path.join(ckpt_path, f"layer_00-model{flag}_states.pt"),
        map_location=torch.device("cpu"),
    )
    print(layer0.keys())
    ckpt_dict["model.embed_tokens.weight"] = layer0["word_embeddings.weight"]
    print("##########")
    for l in range(0, config.num_hidden_layers):
        l_index = str(l + 1).zfill(2)
        layer = torch.load(
            os.path.join(ckpt_path, f"layer_{l_index}-model{flag}_states.pt"),
            map_location=torch.device("cpu"),
        )
        for k in layer:
            if "dummy" in k or "rotary_emb" in k:
                continue
            if k == "self_attention.layernorm_qkv.query_weight":
                ckpt_dict[f"model.layers.{l}.self_attn.q_proj.weight"] = layer[k]
            elif k == "self_attention.layernorm_qkv.key_weight":
                ckpt_dict[f"model.layers.{l}.self_attn.k_proj.weight"] = layer[k]
            elif k == "self_attention.layernorm_qkv.value_weight":
                ckpt_dict[f"model.layers.{l}.self_attn.v_proj.weight"] = layer[k]
            elif k == "self_attention.proj.weight":
                ckpt_dict[f"model.layers.{l}.self_attn.o_proj.weight"] = layer[k]
            elif k == "self_attention.layernorm_qkv.layer_norm_weight":
                ckpt_dict[f"model.layers.{l}.input_layernorm.weight"] = layer[k]
            elif k == "layernorm_mlp.layer_norm_weight":
                ckpt_dict[f"model.layers.{l}.post_attention_layernorm.weight"] = layer[
                    k
                ]
            elif k == "layernorm_mlp.fc2_weight":
                ckpt_dict[f"model.layers.{l}.mlp.down_proj.weight"] = layer[k]
            elif k == "layernorm_mlp.fc1_weight":
                splits = torch.split(layer[k], int(layer[k].size(0) / 2))
                ckpt_dict[f"model.layers.{l}.mlp.gate_proj.weight"] = splits[0]
                ckpt_dict[f"model.layers.{l}.mlp.up_proj.weight"] = splits[1]
            else:
                print(k)
    layer = torch.load(
        os.path.join(
            ckpt_path,
            f"layer_{config.num_hidden_layers+1}-model{flag}_states.pt",
        ),
        map_location=torch.device("cpu"),
    )
    ckpt_dict["model.norm.weight"] = layer["norm.weight"]
    layer = torch.load(
        os.path.join(
            ckpt_path,
            f"layer_{config.num_hidden_layers+2}-model{flag}_states.pt",
        ),
        map_location=torch.device("cpu"),
    )
    ckpt_dict["lm_head.weight"] = layer["lm_head.weight"]
    model_dict.update(ckpt_dict)

    model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":
    # save_config1b(
    #     "genegpt_1b",
    #     "/home/v-zekunguo/blobdata/v-zekunguo/gene/checkpoints/config/config_1b",
    # )
    save_config100m(
        "genegpt_1b", "/home/v-zekunguo/nlm/zekun/gene/checkpoints/1b6kmer4k_3d/config"
    )
    exit(0)
    tokenizer = GeneKMerTokenizer()

    DNA = "ggccaggaattcaagaccagcgtggctaacatggcgaaaccccatctctaccaaaaatacaaaaattagctgggcgtggtggtgcacacttgtaattccagctacttgagaggctgaggtgggaggatcgcttgaacctgggaggcagaagtttcagtgagcccagaacgtgcctctgcactccagccaggatgacagagcaagactccatctcaaaaaaaaaaaaaaaaaaaaaggaaaataaccaaaTGACAATTAGTGAGTACTACTTGCAAAACTTGTACGCAATAGAGTATGAAGCAActataaaatgagagagaaatatcTCCAAATACTACTCTAAAGTAATCTACAAGGTATACCttaactgaaaagaaacaaaaaagtgacACCAGAAtgctatttttatgttaaaacagGGATAAATACATTGGATTTACatgcatatataagtatatattttataaatgtttaaataagcatacttaaaatggcaaaaacgtaatacatatataattttcttatggCAGGAGGAGGAAACAGGGCAAGGCACAGGGATAAAAGTTATTCTGAAtacatcttattttatatttttgactttgAAATCCTGTAGCTgttttatgtaatataaaaatgtaattaaattaacagaaaaaaattacaactgcTAAAAATCAAGATCTGGCATTTTAATTAAGTTATAAAACATCGGAGAAAAGAATTGTTTCATGGGACACTAACATACAGACAAATTCATTTGGAACCCAATGAATTAATGGGCCTAAGATAACAACCAATAGAAGCTAAAATGACGAATAactgtttcagaagaaaacatatatgGAATGAATCAGCTGAAAATACCTGAACCTACTGATCAATTTTTATATCACATGAAGTGAATACACATAAAGTATAATATGGAGCACATAGAACCAACTAGAAATGAGCCTAATTGTtaaatattctctattttatgaCAATATACAGGAAATATGTCGAAGAGAGAAACATGCAAGAACACCGTAGGGTTTAATAAGATAATCACAAGGTATGGAATATTCAACAGGATGAGTATCCTGGATTATTCAGCAAATACACAGAGCTAAAAAGCAGGAGAAAggaattcatatatatttttaaaaactaaaaagatataTTAGCTGATGCAACTTTGAAACTTCTTTAGATCCTGATTCAAATAGagcaaatttaacaaatatatttgaaactattaaaataatttaaaaatgaccaAGTATTTGATTATATCAAATATAgacaataataaccttgaatgtacaTGGATTAAATGTCCActtaggggctgggtgtggtggctcatgactataattccagcactttgggaggccaaggcagaaggattgcttgaggtcagaggttcaagtgcagcctggtcaacacagtgaaaccctatctctacaaaaaacaaacaaaaataaaaaattaactaattttaaaaaatatatatttcttctaaattctCCACCTGAAAGATATAGACTGACTGAATGAATTTTAACTATGATCTGACTATGTGCTTCCCTGAACAAATGCACTTTACCTGTAAAACACATATTaactaaaagaaaagagatggaaaaaggtattccatgaacagaaaccaaaatgagtaggagtagctatacttctgtcagacaaaacagactttaagtcaaaactagctttagaaaaaagacaaaaatgcttATTATACAACGATAAAGGAATCAATCCagaaagaggatataacaattttaaatatatatgcagccaacactggagcagccagattcataaagcaaatactaCTAGATCAAAACAGAGAGGTAGACtcaaatataataatagtgaaggacttcaacaccccactttcagcattaaaCAGATCATCTAATAAGAAAACCAATCTCGCAGCCCTCACCCTGGAGAGTCCACAGGTACCAGGGGTTGGTCTGAACCCCCAGCACAGAGCACCTGCCTCACAGAAGAGTGGCTGCATTTTTCTTCCTGCAGTTTTCAGTCCTCACTTCTCCTTACCAAGCAGGGCCACCTGGCCTGGGACTCCGGTACAACTACCCTGCCCCCCACCTGACGACTTCAATAAGAAGTAGCCCAGCATTTCTCCAAGGAGGAAATACCAGAGTCAATTCACAACCACTGCAATTGCAGTGGTACCACCATAACAGCCCTTGGGCTGCAGAAGGAACTAAGAGTCTAGTCACTACAGTGGCACCTTCAGCACACCACAGCCACCATACAGAGAGGAATCCAGCCCCCTCCCCTGGGAACCCCCACCACCCACTCCACCAGGCACAGCACCCAGCTCATAACTGCAGATCAGTTGCCCCACCCACAGCTGAGCTTACCTACTGGCAGTGGCCCAGACTTTCCCTAGGGAGAGGCTCCCAGAGGCAAACGGCAGCCTCTCTGCCCGTGTCACAGCAGCAGTTCTATCCATGCTGTCCTCAGGCTTGGAAAGAAACAAAGCGCCTGAAGGCTGCACCTGAACTTACAGCATGCCACAGTTCCCATATGGAGAGGAGACCAGTCTCTCCTCCCAGTGAGCCCTAAACCCCCTGATCCCCAACAAGCAGAGCCCTAACCTCACACCAGCAGTACAGCTGCCCCATCCCCCAGGCTGAACATTCCCAGTAATAGCAGCTCCACCTGGAGATGGAACCCCCAGGGTCAACTAAaagcccctctgccactgcctctaCAGTGGTACTACCCCTGCTACCCTTGAACTAACAAAGGAGCAAAGACCCCAGTGCTTTATCCACACCTCCAACAAGCTGCAGTCGACCACAAAGAAGAAACACGTCTGTCTCCCATGGGTCCTACCCACACCCCCTGCTGTTCACCATGGATGATAGAGTCAACAGTGTGAAAacgaccatactgccaaaagcaacctacaaattcaatgcaattcc"
    # DNA="TTTTATATTAGGAATAAACCTAACATTAATGGAGACACTGAGAAGCCGAGATAACTGAATTATAAGGCATAGCCAGGG"
    # # DNA='tatgagtgagaacatgcggtgtttggttttctgttcttgtgttagtttgcggagaatgatggtttccagcttcatc'
    # DNA='tatacatttatgcatacatatgatatataaccTTTTTTTGGTAaaccatttgaaaataagttACATAACATCATGAGAGTTAACCCCtagtattttttatatacctCCTGAGAACAAGAATATTCTCATACACAAACGCAATACTATGATTACACTCAAATGTTGTAATGATATAGGAATATTATTTAATACATAGTTCCTAGTTGTTTCTCGTTTGTCCCAGTAATGTcctatataactttaaaaaaaatatctgAACCGGACATGCATTGGATTTAATGTCTCTTTAATCTAACATCGTTTTGCTACCTTTTAtgttttttcatgattttaacatttttgaaaagtccAAGCCACTGTTTTGTAGAAAGTCGTACTATTTGAGCTATTCTGGTTATTGCCTCagtaatatatttgtttaaacaaataatatatttgtttaaacaaattatattatttgtttaaacgaatatattgtttgtttatttgtttaaacaaataatacatttgtttaaacaaataatataatttgtttaaacaaatatatttgcCAAGAATACTGTGTGGCTGAGGAAAGATAATTCTTAATAAATTGTTTTGCAAAAAGAGCACTTCAGAGCAATGAACTATGGAGCAAAAGGCAAAATGACGGAAACAGCAACCTTAAGTTCATATGGTTATCCTGGAGATACTGATTCTGCATTGCATTGTCGTTGCTTGCTTTTCATTTAGCTGGTCTTACATGAAGGAAATGAACTGATATTTAGTATTTATCCAAGGTAAAATTCTATAGCTGTTTTTGAATTGCCAGTCTTTTTGGTTTACATAAGCCCTTGCTGAAAACGAAAAATCTGCTGCCTGGAATCTATATATTATAGTTATTGTACATTTCCTAGGATTCTTTTGAGAAAACTAATTTTGATCAAGATTCTCACTCTTTACTGAGAAGAATCATTGGGCAAAGTATCTCTTCATCTTTTAAGAAGTTTGCTTCTCATAAGTTAAAGTAGAAATTGTTAAATAGTCAATTGAAACATTTTCAGTGGTCATACCCAAAGAGAAAGTAATAACAGTATTATTTCTTGTTTAAGTCATTATTTGCTGTTCCTTTTTCAGTTACCTCATAATTTTCATGACAGTTTTAATAGGTAGAAATAATTACTTTCTAACTTGGACTTTTTCTACaagttgaatattttcttttagttgaGAAGTttaaagtatacttttaaaaacaaatgtatatgGTTGCTTGCAGATACGATCAAATCACAGAATAG'
    # DNA='CCTGAGAACAAGAATATTCTCATACACAAACGCAATACTATGATTACACTCAAATGTTGTAATGATATAGGAATATTATTTAATACATAGTTCCTAGTTGTTTCTCGTTTGTCCCAGTAATGT'
    # DNA='agtaatatatttgtttaaacaaataatatatttgtttaaacaaattatattatttgtttaaacgaatatattgtttgtttatttgtttaaacaaataatacatttgtttaaacaaataatataatttgtttaaacaaatatatttgc'
    # DNA='gtgaataaaaaaaaaagggttgcCCATGCATAGACTAAGTGTCAGATTTTGACTTAAGGCTTGCTACCAGGTGTGGCTtaattgtcacttgccactcactgatTAGGTTTTGATATCAGTCTGTCAGCAGTTaatttattatggtctctgttTGGTtaaacctctctgctaatgtttatCTGTATTTGCAGTTGCTCCCCAGCACTAGCATCACTGCCTTAGCTCTACCTCAtatcatcaggcattagattctcataaagAGCTTTCAATCTAGATTCCTCACTTATGCAGTTCACAGTAGGCTTTGTGCTTCTGTGAGATGCTAGTGCCacagctgatctgacaggaggtggagctcagcaGTGATATGAGCTATAGGGAGCAGCTGTAATAGAATTGAAGCTACCTGCACTAACCTGATGCTtacctactttctttcttttttttcagtactggggatcaaacttaggaccttatgcttgccaggcaggtgtgacaccactgagcaacatctctgGCCCTGTGTGGGCACTTTCTAAGAGGCCATCAACCAATACCAGTTGGAGACCCCAGGGTAGATAACAACTTTGGTGTGCATCATGGTGATAACAGATGCCTATGTGACTTTTCAGGGACTTGAAGCTGCTTGAGTTGAAACTACTGACTCTTCACTTGGCCCTGCACAGGTGTCAGATGGCCTCAAAATCACCCTCACATCATGGTAGTGCCACCATTGTCTGAACATACATCCAGGAAGAGCCCTGGAGTTTGATTACCAAATGGCATACCACAGATTATTTCATTTCTAGTCACCTCTTCTGAGGCCTCAGAGCACATTGCTTCTTTAGCCCACAAGTGTTCTATTAGTGCTCCCAAATCCTATACTCTGTAAGGTAGACATGAAACTTGAGTCTTCTGTCTCCTTGTGCATGGCTGACTCTTGAACAGATGGTTCTATTTTGTGAAATTTGTCATTTCAGCCATTGGCATACTATGCAGTAAGTGTTTGGCAAGCCAGCACAAAGGAATGCCTAAAGCATCTTTTGCCCATGGTTTTTCAGCCCCTCCTGGTTTGAGAAAGTGTTCATGAATCCAAGCATGTGCTCAAGGACCATCCCCAAGGTTTTCCCCTAGTTTTATGCTGCCCACCTTCAGTGCATAGTTTAACTTTTGCAGTAGAGAAATGGTGTCTAGGTCTAGAAGATGTCTTATCATGTAACTTTCGTACAAGAGATCagatttctatttctccttttctttaactttcattGTAAGGATCTTAGATTTGGAAATGTCTGAAATTTCACCTTCCAAAAGAAACTCTGGTTTGCCCCTCACTGTGTGGACCCTGGTTTAGGGA'
    # DNA='tacctactttctttcttttttttcagtactggggatcaaacttaggaccttatgcttgccaggcaggtgtgacaccactgagcaacatctctg'
    # DNA='GCCCTGTGTGGGCACTTTCTAAGAGGCCATCAACCAATACCAGTTGGAGACCCCAGGGTAGATAACAACTTTGGTGTGCATCATGGTGATAACAGATGCCTATGTGACTTTTCAGGGACTTGAAGCTGCTTGAGTTGAAACTACTGACTCTTCACTTGGCCCTGCACAGGTGTCAGATGGCCTCAAAATCACCCTCACATCATGGTAGTGCCACCATTGTCTGAACATACATCCAGGAAGAGCCCTGGAGTTTGATTACCAAATGGCATACCACAGATTATTTCATTTCTAGTCACCTCTTCTGAGGCCTCAGAGCACATTGCTTCTTTAGCCCACAAGTGTTCTAT'
    DNA = "aactaaaagaaaagagatggaaaaaggtattccatgaacagaaaccaaaatgagtaggagtagctatacttctgtcagacaaaacagactttaagtcaaaactagctttagaaaaaagacaaaaatgcttATTATACAACGATAAAGGAATCAATCCagaaagaggatataacaattttaaatatatatgcagccaacactggagcagccagattcataaagcaaatactaCTAGATCAAAACAGAGAGGTAGACtcaaatataataatagtgaaggacttcaacaccccactttcagcattaaaCAGATCATCTAATAAGAAAACCAATCTCGCAGCCCTCACCCTGGAGAGTCCACAGGTACCAGGGGTTGGTCTGAACCCCCAGCACAGAGCACCTGCCTCACAGAAGAGTGGCTGCATTTTTCTTCCTGCAGTTTTCAGTCCTCACTTCTCCTTACCAAGCAGGGCCACCTGGCCTGGGACTCCGGTACAACTACCCTGCCCCCCACCTGACGACTTCAATAAGAAGTAGCCCAGCATTTCTCCAAGGAGGAAATACCAGAGTCAATTCACAACCACTGCAATTGCAGTGGTACCACCATAACAGCCCTTGGGCTGCAGAAGGAACTAAGAGTCTAGTCACTACAGTGGCACCTTCAGCACACCACAGCCACCATACAGAGAGGAATCCAGCCCCCTCCCCTGGGAACCCCCACCACCCACTCCACCAGGCACAGCACCCAGCTCATAACTGCAGATCAGTTGCCCCACCCACAGCTGAGCTTACCTACTGGCAGTGGCCCAGACTTTCCCTAGGGAGAGGCTCCCAGAGGCAAACGGCAGCCTCTCTGCCCGTGTCACAGCAGCAGTTCTATCCATGCTGTCCTCAGGCTTGGAAAGAAACAAAGCGCCTGAAGGCTGCACCTGAACTTACAGCATGCCACAGTTCCCATATGGAGAGGAGACCAGTCTCTCCTCCCAGTGAGCCCTAAACCCCCTGATCCCCAACAAGCAGAGCCCTAACCTCACACCAGCAGTACAGCTGCCCCATCCCCCAGGCTGAACATTCCCAGTAATAGCAGCTCCACCTGGAGATGGAACCCCCAGGGTCAACTAAaagcccctctgccactgcctctaCAGTGGTACTACCCCTGCTACCCTTGAACTAACAAAGGAGCAAAGACCCCAGTGCTTTATCCACACCTCCAACAAGCTGCAGTCGACCACAAAGAAGAAACACGTCTGTCTCCCATGGGTCCTACCCACACCCCCTGCTGTTCACCATGGATGATAGAGTCAACAGTGTGAAAacgaccatactgccaaaagcaacctacaaattcaatgcaattcc"
    input_ids = tokenizer.encode_sequences([DNA], padding=True, truncation=True)[
        "input_ids"
    ]
    print(input_ids)
    input_ids = torch.tensor(input_ids)
    labels = input_ids.clone()
    # model = load_model1b("/home/v-zekunguo/blobdata/v-zekunguo/gene/checkpoints/config/config_1b/config.json",
    #             "/home/v-zekunguo/blobdata/v-zekunguo/gene/checkpoints/real_1b6kmer16k/global_step7000"
    #             )
    model = load_model100m(
        "/home/v-zekunguo/blobdata/v-zekunguo/gene/checkpoints/config/config_100m/config.json",
        "/home/v-zekunguo/blobdata/v-zekunguo/gene/checkpoints/100m6kmer160k/global_step6000",
    )
    model.cuda()
    out = model(input_ids.cuda(), labels=labels.cuda())
    print(out.loss)
    # load_model100m(
    #     "/home/v-zekunguo/blobdata/v-zekunguo/gene/checkpoints/config/config_100m/config.json",
    #     "/home/v-zekunguo/blobdata/v-zekunguo/gene/checkpoints/100m6kmer160k/global_step1500",
    #     # "/home/v-zekunguo/blobdata/v-zekunguo/gene/checkpoints/real_1b6kmer16k/global_step7000",
    # )
