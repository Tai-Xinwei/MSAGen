# -*- coding: utf-8 -*-
import torch
from datetime import timedelta
import os

import json
import re
import shutil
import sys
from pathlib import Path

from abc import ABC, abstractmethod
import numpy as np
import random
import argparse
from argparse import ArgumentParser
# from bitsandbytes import quantize_model

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
from transformers import LlamaConfig, LlamaForCausalLM
import torch.nn.functional as F

import pickle as pkl

from glob import glob
import time

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from sfm.data.sci_data.NlmTokenizer import NlmLlama3Tokenizer,NlmTokenizer


sys.path.append(os.path.join(Path(__file__).parent.parent.parent, "."))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(local_rank=0):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["WORLD_SIZE"])
        local_gpu = int(os.environ["LOCAL_RANK"])
        print(
            "Get init distributed settings successfully, rank: {}, world_size: {}!".format(
                local_rank, local_world_size
            )
        )
    else:
        print("Error when get init distributed settings!")
        local_rank = local_rank
        local_world_size = torch.cuda.device_count()
        local_gpu = local_rank
        print(
            "Use local setting, rank: {}, world_size: {}!".format(
                local_rank, local_world_size
            )
        )

    print("| distributed init (rank {}): env://".format(local_rank), flush=True)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=local_world_size,
        rank=local_rank,
        timeout=timedelta(seconds=30*60*1000) # 30 minutes as the model can be large,
    )
    torch.cuda.set_device(local_gpu)

    torch.distributed.barrier()
    setup_for_distributed(local_rank == 0)
    return local_rank


class TextFileDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_text_file()

    def load_text_file(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]

        print(f"Input file size: {len(lines)}")

        world_size = int(os.environ["WORLD_SIZE"])
        residual = len(lines) % world_size
        if residual != 0:
            lines.extend(["This is a padding sentence."] * (world_size - residual))

        print(f"Padded file size: {len(lines)}")

        return lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

import csv
class TSVFileDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_text_file()

    def load_text_file(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            lines = [line for line in reader]

        print(f"Input file size: {len(lines)}")

        world_size = int(os.environ["WORLD_SIZE"])
        residual = len(lines) % world_size
        if residual != 0:
            lines.extend(["This is a padding sentence."] * (world_size - residual))

        print(f"Padded file size: {len(lines)}")

        return lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class NLMGenerator(ABC):
    def __init__(self, base_model_home, model_ckpt_home, aux_home):

        self.base_model_home = base_model_home
        self.model_ckpt_home = model_ckpt_home
        self.aux_home = aux_home

        self.tokenizer, self.model = self.prepare()
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["RANK"])
        print(f"local_rank / world_size: {local_rank} / {world_size}")

        self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        self.model.eval()

    @abstractmethod
    def prepare(self):
        pass

    def chat(
        self, input_str, response_only=True, do_sample=False, file_name="", **kwargs
    ):
        prompt = input_str.strip()
        try:
            tokenized_out = self.tokenizer(prompt, return_tensors="pt")
        except Exception as e:
            print('error',e)
            return []
        input_ids = tokenized_out.input_ids.cuda()
        attention_mask = tokenized_out.attention_mask.cuda()
        if len(input_ids[0])>1024 and 'instruct_gene_annot_031824_test_no_predict' in file_name:
            print('to long')
            return []

        if "max_new_tokens" in kwargs:
            max_new_tokens = kwargs.pop("max_new_tokens")
            num_return_sequences=1
        else:
            max_new_tokens = 128
            max_new_tokens=160
            num_return_sequences=4
        if "temperature" in kwargs:
            temperature = kwargs.pop("temperature")
            print(f"temperature:{temperature}")
        else:
            temperature = 0.75
            # temperature = 1
        # max_new_tokens=256
        # num_return_sequences=3
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model

        if do_sample:
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                top_p=0.95,
                **kwargs,
            )
            # print(outputs)
        else:
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        out_list = []
        outputs = outputs.to("cpu")
        # response_only=False
        for out in outputs:
            s = self.tokenizer.decode(out)
            # print("###")
            # print(s)
            if response_only:
                segs = s.split("Response:")
                s = segs[1].strip()
            segs = s.split(self.tokenizer.eos_token)
            resp = segs[0].strip()
            if "<protein>" in resp:
                resp = resp.replace("<a>", "")
            if "<mol>" in resp:
                resp = resp.replace("<m>", "")
            out_list.append(resp)
        return out_list

    def inference(self, input_file, output_dir, **kwargs):
        print(f"Test file path:{input_file}")
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["RANK"])

        text_dataset = TextFileDataset(input_file)
        # text_dataset = TSVFileDataset(input_file)
        if world_size > 1:
            self.sampler_train = torch.utils.data.DistributedSampler(text_dataset)
        else:
            self.sampler_train = torch.utils.data.RandomSampler(text_dataset)
        batch_sampler = torch.utils.data.BatchSampler(
            self.sampler_train, batch_size=1, drop_last=False
        )
        data_loader = torch.utils.data.DataLoader(
            text_dataset, batch_sampler=batch_sampler, pin_memory=False, num_workers=1
        )

        buffer = []

        start_time = time.time()

        with torch.no_grad():
            for idx, test_sample in tqdm(
                enumerate(data_loader), total=len(data_loader)
            ):
                try:
                    q = test_sample[0].split("\t")[0].strip()
                except :
                    print(f"############{test_sample[0]}")
                    continue
                prompt1 = f"Instruction: {q.strip()}\n\n\nResponse:"

                if 'TamGen' in input_file or '/zequn/' in input_file or 'dashi' in input_file or 'material' in input_file:
                    print('process tamgen')
                    all_r0 = []
                    all_r1 = []
                    for i in range(100):
                        cur_seed=random.randint(0,2147483647)
                        set_seed(cur_seed)
                        if 'dashi' in input_file:
                            r1 = self.chat(prompt1, do_sample=True,max_new_tokens=512, **kwargs)
                        else:
                            r1 = self.chat(prompt1, do_sample=True, **kwargs)
                        all_r1.extend(r1)
                    buffer.append((test_sample[0], all_r0, all_r1))
                else:
                    r0 = self.chat(prompt1, do_sample=False,file_name=input_file)
                    r1 = self.chat(prompt1, do_sample=True,file_name=input_file)

                    buffer.append((test_sample[0], r0, r1))

        print(
            f"Local rank #[{local_rank}]: execution {(time.time() - start_time):.2f} seconds",
            force=True,
        )

        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, f"part{local_rank}.pkl"), "wb") as fw:
            print(f"Local rank #[{local_rank}] writing file: {len(buffer)}", force=True)
            pkl.dump(buffer, fw)
            print(f"Local rank #[{local_rank}] finished writing.", force=True)

        # torch.distributed.barrier()

    # def get_score(self,prompt):
    #     # get yes or no score
    #     prompt = prompt.strip()
    #     # print(f"prompt: {prompt}")
    #     try:
    #         tokenized_out = self.tokenizer(prompt, return_tensors="pt")
    #     except Exception as e:
    #         print('error',e)
    #         return []
    #     input_ids = tokenized_out.input_ids.cuda()
    #     attention_mask = tokenized_out.attention_mask.cuda()
    #     outputs = self.model.module(
    #         input_ids,
    #         input_ids,
    #         return_dict=True,
    #         output_attentions=False,
    #         output_hidden_states=False,
    #     )
    #     yes_id = self.tokenizer("Yes",return_tensors="pt").input_ids[-1]
    #     no_id = self.tokenizer("No",return_tensors="pt").input_ids[-1]
    #     next_token_logits = outputs.logits[:, -1, :]
    #     next_token_scores = F.softmax(
    #         next_token_logits, dim=-1
    #     )  # (batch_size * num_beams, vocab_size)
    #     yes_probs = next_token_scores[:, yes_id]
    #     no_probs = next_token_scores[:, no_id]
    #     confidences = yes_probs / (yes_probs + no_probs)
    #     print(confidences)
    #     return confidences

    def generate(self, n_seq, entity, output_dir):

        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["RANK"])

        print(f"local_rank / world_size: {local_rank} / {world_size}")

        self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        self.model.eval()

        output_path = os.path.join(output_dir, f"{local_rank}.generate.txt")

        input_ids = self.tokenizer(f"<{entity}>", return_tensors="pt").input_ids.cuda()
        print(input_ids)
        printed = False

        with torch.no_grad():
            with open(output_path, "wt") as writer:
                for _ in tqdm(range(n_seq), mininterval=10):
                    outputs = self.model.module.generate(
                        input_ids,
                        max_new_tokens=512,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        num_return_sequences=1,
                    )
                    output_text = self.tokenizer.decode(outputs[0])
                    if not printed:
                        print(f"raw text: {output_text}")
                    begin_idx = output_text.find(f"<{entity}>") + len(f"<{entity}>")
                    end_idx = output_text.find(f"</{entity}>")

                    if end_idx == -1:
                        end_idx = len(output_text)

                    entity_text = output_text[begin_idx:end_idx]
                    entity_text = entity_text.replace("<a>", "").strip()
                    entity_text = entity_text.replace(" ", "")
                    if not printed:
                        print(f"ret text: {entity_text}")
                    writer.write(entity_text + "\n")
                    printed = True


class MockGenerator(NLMGenerator):

    def __init__(self, base_model_home, model_ckpt_home, aux_home):
        super().__init__(base_model_home, model_ckpt_home, aux_home)

    def prepare(self):

        import torch.nn as nn
        import torch.nn.functional as F

        class MockNet(nn.Module):
            def __init__(self):
                super(MockNet, self).__init__()
                self.fc1 = nn.Linear(20, 50)
                self.fc2 = nn.Linear(50, 30)
                self.fc3 = nn.Linear(30, 10)

            def forward(self, x):
                x = F.relu(self.fc1(x))  # Activation function for hidden layer
                x = F.relu(self.fc2(x))  # Activation function for hidden layer
                x = self.fc3(x)  # No activation function for output layer
                return x

            def generate(self, input_ids, **kwargs):
                return torch.randint(0, 100, (1, 100))

        model = MockNet().cuda()

        from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer

        tokenizer = SFMDecTokenizer.from_pretrained(
            self.base_model_home,
            prot_spm_path=f"{self.aux_home}/ur50bpe/bpe",
            dna_spm_path=f"{self.aux_home}/dnabpe/bpe",
            rna_spm_path=f"{self.aux_home}/rnabpe/bpe",
        )

        return tokenizer, model


class Llama2Generator(NLMGenerator):

    def __init__(self, base_model_home, model_ckpt_home, aux_home):
        super().__init__(base_model_home, model_ckpt_home, aux_home)
        # self.base_model_home = "/home/lihe/mlla/lihe/hai1/ds_dataset/llama2/llama-2-7b"
        # self.aux_home = "/home/lihe/mlla/shufxi/data/scigpt/"

    def prepare(self):

        if not self.aux_home:
            raise Exception(
                "The scigpt root path should be specified as aux_home for bpe tokenizer: $root$/{ur50,dna,rna}bpe/bpe"
            )

        from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer

        tokenizer = SFMDecTokenizer.from_pretrained(
            self.base_model_home,
            prot_spm_path=f"{self.aux_home}/ur50bpe/bpe",
            dna_spm_path=f"{self.aux_home}/dnabpe/bpe",
            rna_spm_path=f"{self.aux_home}/rnabpe/bpe",
        )

        def show_ckpt(name, ckpt):
            for k, v in ckpt.items():
                if "dummy" not in k:
                    print(name, k, v.shape)

        model = AutoModelForCausalLM.from_pretrained(self.base_model_home)

        model_dict = model.state_dict()
        ckpt_dict = {}
        layer0 = torch.load(
            os.path.join(self.model_ckpt_home, "layer_00-model_states.pt"),
            map_location=torch.device("cpu"),
        )
        ckpt_dict["model.embed_tokens.weight"] = layer0["embed_tokens.weight"]
        show_ckpt("layer0", layer0)

        for l in range(0, 32):
            l_index = str(l + 1).zfill(2)
            layer = torch.load(
                os.path.join(self.model_ckpt_home, f"layer_{l_index}-model_states.pt"),
                map_location=torch.device("cpu"),
            )
            show_ckpt(l_index, layer)
            for k in layer:
                if "dummy" in k or "rotary_emb" in k:
                    continue
                ckpt_dict[f"model.layers.{l}.{k}"] = layer[k]
        layer = torch.load(
            os.path.join(self.model_ckpt_home, "layer_33-model_states.pt"),
            map_location=torch.device("cpu"),
        )
        show_ckpt(33, layer)
        ckpt_dict["model.norm.weight"] = layer["norm.weight"]

        layer = torch.load(
            os.path.join(self.model_ckpt_home, "layer_34-model_states.pt"),
            map_location=torch.device("cpu"),
        )
        show_ckpt(33, layer)
        ckpt_dict["lm_head.weight"] = layer["lm_head.weight"]
        model_dict.update(ckpt_dict)

        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(model_dict)

        model.cuda()

        return tokenizer, model


class Llama3Generator(NLMGenerator):

    def __init__(self, base_model_home, model_ckpt_home, aux_home):
        super().__init__(base_model_home, model_ckpt_home, aux_home)
        # self.base_model_home = "/home/lihe/sfmdataeastus2_nlm/llama/Meta-Llama-3-8B/original/"
        # self.model_ckpt_home = "/home/lihe/sfmdataeastus2_nlm/kaiyuan/results/nlm/inst/inst_tuning_full_bsz128_lr5e-5_0616/"

    @abstractmethod
    def get_args_and_tokenizer(self):
        pass

    def prepare(self):
        config, tokenizer = self.get_args_and_tokenizer()
        model = self.load_model(config, self.model_ckpt_home)
        return tokenizer, model

    def load_model(self, config, ckpt_path):
        model = LlamaForCausalLM(config)
        model_dict = model.state_dict()
        ckpt_dict = {}
        layer0 = torch.load(
            os.path.join(ckpt_path, "layer_00-model_00-model_states.pt"),
            map_location=torch.device("cpu"),
        )
        ckpt_dict["model.embed_tokens.weight"] = layer0["word_embeddings.weight"]
        for l in range(0, config.num_hidden_layers):
            l_index = str(l + 1).zfill(2)
            layer = torch.load(
                os.path.join(ckpt_path, f"layer_{l_index}-model_00-model_states.pt"),
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
                    ckpt_dict[f"model.layers.{l}.post_attention_layernorm.weight"] = (
                        layer[k]
                    )
                elif k == "layernorm_mlp.fc2_weight":
                    ckpt_dict[f"model.layers.{l}.mlp.down_proj.weight"] = layer[k]
                elif k == "layernorm_mlp.fc1_weight":
                    splits = torch.split(layer[k], int(layer[k].size(0) / 2))
                    ckpt_dict[f"model.layers.{l}.mlp.gate_proj.weight"] = splits[0]
                    ckpt_dict[f"model.layers.{l}.mlp.up_proj.weight"] = splits[1]
                else:
                    print(f"unexcept key {k}")
        layer = torch.load(
            os.path.join(
                ckpt_path,
                f"layer_{config.num_hidden_layers+1}-model_00-model_states.pt",
            ),
            map_location=torch.device("cpu"),
        )
        ckpt_dict["model.norm.weight"] = layer["norm.weight"]

        layer = torch.load(
            os.path.join(
                ckpt_path,
                f"layer_{config.num_hidden_layers+2}-model_00-model_states.pt",
            ),
            map_location=torch.device("cpu"),
        )
        ckpt_dict["lm_head.weight"] = layer["lm_head.weight"]

        model_dict.update(ckpt_dict)

        model.load_state_dict(model_dict)
        model.cuda()
        return model




class Base1BGenerator(Llama3Generator):

    def __init__(self, base_model_home, model_ckpt_home, aux_home):
        super().__init__(base_model_home, model_ckpt_home, aux_home)

    def get_args_and_tokenizer(self):
        config = LlamaConfig.from_json_file(self.base_model_home + "/config.json")
        config.hidden_size = 2048
        config.intermediate_size = 5504
        config.num_hidden_layers = 16
        config.num_attention_heads = 32
        config.num_key_value_heads = 8
        config.max_position_embeddings = 8192
        config.tokens_per_sample = 8192
        config.vocab_size = 130305
        # config.rope_theta = 1000000.0
        tokenizer = NlmLlama3Tokenizer.from_pretrained(self.base_model_home)

        return config, tokenizer


class Base8BGenerator(Llama3Generator):

    def __init__(self, base_model_home, model_ckpt_home, aux_home):
        super().__init__(base_model_home, model_ckpt_home, aux_home)

    def get_args_and_tokenizer(self):
        config = LlamaConfig.from_json_file(self.base_model_home + "/config.json")
        config.hidden_size = 4096
        config.intermediate_size = 14336
        config.num_hidden_layers = 32
        config.num_attention_heads = 32
        config.num_key_value_heads = 8
        config.max_position_embeddings = 8192
        config.tokens_per_sample = 8192
        config.vocab_size = 130304
        # pretrain rope_theta
        # config.rope_theta = 1000000.0
        tokenizer = NlmLlama3Tokenizer.from_pretrained(self.base_model_home)

        return config, tokenizer




class MixtralGenerator(NLMGenerator):

    def __init__(self, base_model_home, model_ckpt_home, aux_home):
        super().__init__(base_model_home, model_ckpt_home, aux_home)
        # self.base_model_home = "/scratch/workspace/nlm/Mixtral-8x7B-v0.1/"
        # self.model_ckpt_home = "/scratch/workspace/nlm/sfmdataeastus2_nlm/shufxi/nlm/8x7b/inst/20240611215447/global_step33216/"
        # self.aux_home = "/tmp/nlm_rank0"

    def download_and_convert_ckpt(self, mixtral_blob_path, nlm_blob_path, local_path):
        import safetensors.torch as st

        if os.path.exists(local_path) and os.listdir(local_path):
            print(f"local_path {local_path} exists and not empty, skip")
            return

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
        st.save_file(
            ckpt_new, os.path.join(local_path, ckpt_new_name), metadata=metadata
        )
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
                ckpt_new[f"model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight"] = (
                    ckpt_old[f"block_sparse_moe.experts.{j}.w1.weight"]
                )
                ckpt_new[f"model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight"] = (
                    ckpt_old[f"block_sparse_moe.experts.{j}.w2.weight"]
                )
                ckpt_new[f"model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight"] = (
                    ckpt_old[f"block_sparse_moe.experts.{j}.w3.weight"]
                )
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
        st.save_file(
            ckpt_new, os.path.join(local_path, ckpt_new_name), metadata=metadata
        )
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
        st.save_file(
            ckpt_new, os.path.join(local_path, ckpt_new_name), metadata=metadata
        )
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

    def prepare(self):

        from sfm.data.sci_data.NlmTokenizer import NlmTokenizer

        tokenizer = NlmTokenizer.from_pretrained(self.base_model_home)
        tokenizer.padding_side = "left"
        print("vocab size", len(tokenizer))

        from transformers import BitsAndBytesConfig

        self.download_and_convert_ckpt(
            self.base_model_home, self.model_ckpt_home, self.aux_home
        )
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.aux_home, quantization_config=quantization_config
        )

        return tokenizer, model

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_name", action="store", help="", default="")
    parser.add_argument("--command", action="store", help="", default="inference")
    parser.add_argument("--input_file", action="store", help="", default="")
    parser.add_argument("--output_dir", action="store", help="", default="")
    parser.add_argument("--base_model_root", action="store", help="", default="")
    parser.add_argument("--model_ckpt_home", action="store", help="", default="")
    parser.add_argument("--aux_home", action="store", help="", default="")
    parser.add_argument("--n_seq", type=int, action="store", help="", default=128)
    parser.add_argument("--entity", action="store", help="", default="protein")
    parser.add_argument(
        "--temperature", action="store", type=float, help="", default=0.7
    )

    return parser.parse_args()


def main():
    args = parse_args()

    init_distributed_mode()

    if args.model_name == "llama2_7b":
        g_cls = Llama2Generator
    elif args.model_name == "llama3_8b":
        g_cls = Base8BGenerator
    elif args.model_name == "mixtral_8x7b":
        g_cls = MixtralGenerator
    elif args.model_name == "base1b":
        g_cls = Base1BGenerator

    g = g_cls(args.base_model_root, args.model_ckpt_home, args.aux_home)
    if args.command == "inference":
        if os.path.isdir(args.input_file):
            file_paths = [
                os.path.join(args.input_file, "test.uspto50k.retro.osmi.tsv"),
                os.path.join(args.input_file, "test.desc2mol.tsv"),
                os.path.join(
                    args.input_file, "iupac_smiles_translation/test.new.i2s_i.txt"
                ),
                os.path.join(args.input_file, "test.instruct.predict_bbbp.tsv"),
                os.path.join(args.input_file, "test.instruct.predict_bace.tsv"),
                os.path.join(args.input_file, "test.uspto50k.reaction.osmi.tsv"),
                os.path.join(
                    args.input_file, "test.molinstruct.reagent_prediction.tsv"
                ),
                os.path.join(args.input_file, "test.mol2desc.tsv"),

                os.path.join(
                    args.input_file, "iupac_smiles_translation/test.new.s2i_s.txt"
                ),
                '/nlm/zekun/data/scidata/instruct/raw/llasmol.test.s2i.tsv',
                '/nlm/zekun/data/scidata/instruct/raw/llasmol.test.i2s.tsv',
                '/blob/zequnliu/SFM_instruct/test.mol2text.chebi',
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/BindingAffinity/sample500/test.BindngDB_pIC50_reg.500.tsv",
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/BindingAffinity/sample500/test.BindngDB_pKd_reg.500.tsv",
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/BindingAffinity/sample500/test.BindngDB_pKi_reg.500.tsv",

                # "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/hit_lead_optim/drd2-hi_prediction_test_1.txt",
                # "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/hit_lead_optim/drd2-lo_prediction_test_1.txt",
                # "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/hit_lead_optim/hiv-hi_prediction_test_1.txt",
                # "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/hit_lead_optim/kcnh2-lo_prediction_test_1.txt",
                # "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/hit_lead_optim/kdr-hi_prediction_test_1.txt",
                # "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/hit_lead_optim/kdr-lo_prediction_test_1.txt",
                # "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/hit_lead_optim/sol-hi_prediction_test_1.txt",
                # "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/basic_property/gen_property_above_full.txt",
                # "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/basic_property/gen_property_below_full.txt",
                # "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/basic_property/gen_property_full.txt",
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/ADMET-generation-optimization/test.CYP1A2.normal.osmi.tsv",
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/ADMET-generation-optimization/test.CYP2C19.normal.osmi.tsv",
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/ADMET-generation-optimization/test.CYP2C9.normal.osmi.tsv",
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/ADMET-generation-optimization/test.CYP2D6.normal.osmi.tsv",
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/ADMET-generation-optimization/test.CYP3A4.normal.osmi.tsv",
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/ADMET-generation-optimization/test.instruct.decrease_bbbp.tsv",
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/ADMET-generation-optimization/test.instruct.gen_bbbp.tsv",
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/small_molecule/ADMET-generation-optimization/test.instruct.increase_bbbp.tsv",
                # "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/BindingAffinity/full/test.BindngDB_pKd_reg.tsv.filt",
                # "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/BindingAffinity/full/test.BindngDB_pKi_reg.tsv.filt",
                # "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/BindingAffinity/full/test.BindngDB_pIC50_reg.tsv.filt",
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/DNA-RNA/val.grna.filter",
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/DNA-RNA/val.grna.improve.classification.filter",
                '/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240617/Drug_Assist/all_test/test.drug.donor.tsv',
                '/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240617/Drug_Assist/all_test/test.drug.logp.tsv',
                '/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240617/Drug_Assist/all_test/test.drug.qed.tsv',
                '/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240617/Drug_Assist/all_test/test.drug.solubility.tsv',
                '/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/material/test.bulk.tsv',
                '/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/test.antibody.design.tsv',
                '/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/test.antigen_antibody.design.tsv',
                '/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/test.antigen_to_full_antibody.design.tsv',
                '/nlm/chuancao/DNAinstuct/classfication/Core_Promoter_detection_test.tsv',
                '/nlm/chuancao/DNAinstuct/classfication/Promoter_detection_test.tsv',
                '/nlm/chuancao/DNAinstuct/classfication/Transcription_factor_binding_site_prediction_0_test.tsv',
                '/nlm/chuancao/DNAinstuct/classfication/Transcription_factor_binding_site_prediction_1_test.tsv',
                '/nlm/chuancao/DNAinstuct/classfication/Transcription_factor_binding_site_prediction_2_test.tsv',
                '/nlm/chuancao/DNAinstuct/classfication/Transcription_factor_binding_site_prediction_3_test.tsv',
                '/nlm/chuancao/DNAinstuct/classfication/Transcription_factor_binding_site_prediction_4_test.tsv',

                '/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/test.protein2rna.filt.tsv',
                '/blob/lihe/data/sfm/scigpt_molinst_pro_v3/catalytic_activity_test.tsv',
                '/blob/lihe/data/sfm/scigpt_molinst_pro_v3/domain_motif_test.tsv',
                '/blob/lihe/data/sfm/scigpt_molinst_pro_v3/general_function_test.tsv',
                '/blob/lihe/data/sfm/scigpt_molinst_pro_v3/protein_function_test.tsv',
                '/nlm/chuancao/DNAinstuct/regression/human_enhancer_K562_test.tsv',
                '/nlm/chuancao/DNAinstuct/regression/yeast_prom_complex_test.tsv',

                '/blob/lihe/data/sfm/geneanno/instruct_gene_annot_031824_test_no_predict.tsv',
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/TamGen/test.SFMinstruct.d2t.tsv",
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/TamGen/test.SFMinstruct.t2d.tsv",
                "/blob/yinxia/wu2/shared/SFM/SFM.overall.data/SFMMolInstruct.20240718.test/cross_modality/TamGen/test.SFMinstruct.t2f.tsv",
                '/nlm/zekun/data/zequn/binding_affinity.txt',
                '/nlm/zekun/data/zequn/general_property.txt',

                '/blob/v-weizhan/Material/material_and_bgap_data/val_bgap_only.csv',
                '/blob/v-weizhan/Material/material_and_bgap_data/val_material_and_bgap.csv',
                '/blob/v-weizhan/Material/material_and_bgap_data/val_material_only.csv',
                '/blob/v-weizhan/Material/material_and_bgap_data/val_smi.txt',
                '/blob/v-weizhan/Material/material_and_bgap_data/val_text.txt',
                '/nlm/zekun/data/dashi/solubility_no.txt',
                '/nlm/zekun/data/dashi/solubility_yes.txt',
                '/nlm/zekun/data/dashi/stability_no.txt',
                '/nlm/zekun/data/dashi/stability_yes.txt',
            ]
            # file_paths=[]
            # for file_name in os.listdir(args.input_file):
            #     file_paths.append(os.path.join(args.input_file,file_name))
            for file in file_paths:
                cur_out_dir = os.path.join(args.output_dir, os.path.basename(file))
                os.makedirs(cur_out_dir, exist_ok=True)
                g.inference(file, cur_out_dir)
        else:
            g.inference(args.input_file, args.output_dir, temperature=args.temperature)
    elif args.command == "generate":
        g.generate(args.n_seq, args.entity, args.output_dir)
    else:
        raise Exception(f'Unkonwn command: "{args.command}"')


if __name__ == "__main__":

    main()


# v5_processed_vertebrate_others_sub1_dna_six_train.npy.lmdb
# mv processed_zekun_vertebrate_others_210_250_0_-1.lmdb v5_processed_vertebrate_others_sub2_dna_six_train.npy.lmdb
