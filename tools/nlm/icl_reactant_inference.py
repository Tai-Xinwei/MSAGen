# -*- coding: utf-8 -*-
from datetime import timedelta
import torch
import os
import numpy as np
import json
import re
import shutil
import sys
from pathlib import Path

from abc import ABC, abstractmethod

import argparse

from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import LlamaConfig, LlamaForCausalLM

import pickle as pkl

import time

from torch.utils.data import Dataset

sys.path.append(os.path.join(Path(__file__).parent.parent.parent, "."))

from sfm.data.sci_data.NlmTokenizer import NlmLlama3Tokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from sfm.logging import logger


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
    def __init__(self, file_path, tokenizer, use_template=False, max_len=8192):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.use_template = use_template
        self.max_len = max_len
        self.data = self.load_text_file()

    def load_text_file(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file]

        print(f"Input file size: {len(lines)}")

        world_size = int(os.environ["WORLD_SIZE"])
        residual = len(lines) % world_size
        if residual != 0:
            padding_line = "\t".join(["This is a padding sentence."] * 3)
            lines.extend([padding_line] * (world_size - residual))

        print(f"Padded file size: {len(lines)}")

        return lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        groups = text.split("\t")
        print("groups is: ", groups)

        tokens = []
        labels = []

        print("groups's length is: ", len(groups))

        if len(groups) > 2:
            for i in range(0, len(groups), 2):
                prompt = groups[i]
                target = groups[i + 1]

                prompt = f"Instruction: {prompt}\n\n\nResponse:"

                prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                target_tokens = self.tokenizer.encode(target, add_special_tokens=False)

                prompt_tokens = [token for token in prompt_tokens if token != self.tokenizer.bos_token_id]
                target_tokens = [token for token in target_tokens if token != self.tokenizer.bos_token_id]

                group_tokens = (
                    ([self.tokenizer.bos_token_id] if i == 0 else [])
                    + prompt_tokens
                    + (target_tokens if i < len(groups) - 2 else [])
                    + ([self.tokenizer.eos_token_id] if i < len(groups) - 2 else [])
                )

                group_labels = group_tokens[:]
                group_labels[: len(prompt_tokens) + (1 if i == 0 else 0)] = [-100] * (
                    len(prompt_tokens) + (1 if i == 0 else 0)
                )

                # print("tokens is: ", tokens)
                # print("labels is: ", labels)

        elif len(groups) == 2:
            prompt = f"Instruction: {groups[0]}\n\n\nResponse:"

            print("prompt is: ", prompt)

            tokens = group_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

            group_labels = group_tokens[:]
            group_labels[: len(tokens) + 1] = [-100] * (
                len(tokens) + 1
            )

            # print("tokens is: ", tokens)
            # print("labels is: ", labels)

        # keep the last max_len tokens, or there maybe no labels to pred
        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]

        return torch.tensor(tokens), torch.tensor(labels)

    def collate(self, samples):
        input_ids_list, labels_list = zip(*samples)

        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        # )

        # padding_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # input = tuple([input_ids, padding_mask])
        return input_ids_list



class NLMGenerator(ABC):
    def __init__(self, base_model_home, model_ckpt_home, aux_home):

        self.base_model_home = base_model_home
        self.model_ckpt_home = model_ckpt_home
        self.aux_home = aux_home

        self.tokenizer, self.model = self.prepare()

    @abstractmethod
    def prepare(self):
        pass

    def chat(self, input_str, response_only=True, do_sample=False, **kwargs):

        # prompt = f"Instruction: {input_str.strip()}\n\n\nResponse:"
        # tokenized_out = input_str = self.tokenizer(
        #     prompt, return_tensors="pt"
        # )
        # input_str = tokenized_out.input_ids.cuda()
        # attention_mask = tokenized_out.attention_mask.cuda()

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model

        print("Starting generation...")

        ## Check input_str
        input_tensor = input_str

        input_numpy = input_tensor.cpu().numpy()

        input_ids = input_numpy[0]

        input_ids_list = input_ids.tolist()

        decoded_string = self.tokenizer.decode(input_ids_list)

        print("Decoded input string is: ", decoded_string)

        ## End check
        input_str = input_str.cuda()
        print("input_str is: ", input_str)

        if do_sample:
            outputs = model.generate(
                input_str,
                # attention_mask=attention_mask,
                max_new_tokens=100,
                num_return_sequences=4,  # Changed to 1
                do_sample=True,
                temperature=0.75,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
            print("Sampled outputs is: ", outputs)
        else:
            outputs = model.generate(
                input_str,
                # attention_mask=attention_mask,
                num_beams=4,
                max_new_tokens=100,
                num_return_sequences=4,  # Changed to 1
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
            print("outputs is: ", outputs)

        out_list = []

        for out in outputs:
            s = self.tokenizer.decode(out)
            if response_only:
                segs = s.split("Response:")
                s = segs[1].strip()
            segs = s.split(self.tokenizer.eos_token)
            print("segs is: ", segs)
            print("segs's length is: ", len(segs))
            # if len(segs) > 2:
            #     resp = segs[-2].strip()
            # else:
            #     resp = segs[0].strip()
            resp = segs[0].strip()
            resp = resp.replace(" <a>", "")
            resp = resp.replace(' <m>', '')
            resp = resp.replace('<m>', '')
            out_list.append(resp)
        return out_list



    def inference(self, input_file, output_dir):

        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["RANK"])

        print(f"local_rank / world_size: {local_rank} / {world_size}")

        self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        text_dataset = TextFileDataset(input_file, self.tokenizer, use_template=False, max_len=8192)
        print("text_dataset is: ", text_dataset)
        print("text_dataset's length is: ", len(text_dataset))
        samples = [text_dataset[i] for i in range(len(text_dataset))]
        text_dataset = text_dataset.collate(samples)

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

        self.model.eval()
        buffer = []

        start_time = time.time()

        with torch.no_grad():
            for idx, test_sample in tqdm(
                enumerate(data_loader), total=len(data_loader)
            ):
                # q = test_sample[0].split('\t')[0].strip()
                print("test_sample shape is: ", test_sample.shape)
                print("test_sample is: ", test_sample)

                ##check test_sample

                token_ids = test_sample.squeeze().tolist()

                print(token_ids)

                decoded_sequence = self.tokenizer.decode(token_ids)

                print("Decoded sequence is: ", decoded_sequence)
                decoded_sequence = decoded_sequence.replace("<|begin_of_text|>", "")
                decoded_sequence = decoded_sequence.replace("<m>", "")
                ##end check

                r0 = self.chat(test_sample, do_sample=False)
                r1 = self.chat(test_sample, do_sample=True)

                r0_array = np.array(r0)
                print("r0's shape is: ", r0_array.shape)
                print("original r0 is: ", r0)

                # r0 = r0[0] if isinstance(r0, list) else r0
                # r1 = r1[0] if isinstance(r1, list) else r1

                print("r0 is: ", r0)
                print("r1 is: ", r1)

                # test_sample_str = test_sample[0] if isinstance(test_sample, (list, tuple)) else test_sample
                #解码输入
                # test_sample_str = self.tokenizer.decode(test_sample_str[0])
                buffer.append((decoded_sequence, f"Strategy 1 Response: {r0}", f"Strategy 2 Response: {r1}"))

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

    def generate(self, n_seq, entity, output_dir, max_new_tokens):

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
                        max_new_tokens=max_new_tokens,
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
                    entity_text = entity_text.replace(" ","")
                    entity_text = entity_text.replace("<m>", "").strip()
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
        config.rope_theta = 1000000.0
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
        config.rope_theta = 1000000.0
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
        config["vocab_size"] = 38078
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

class MixtralDistGenerator(MixtralGenerator):
    def __init__(self, base_model_home, model_ckpt_home, aux_home):
        self.model_parallel_size = int(os.environ.get("MODEL_PARALLEL_SIZE", 1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        print(f"model parallel size: {self.model_parallel_size}")

        assert self.world_size % self.model_parallel_size == 0
        self.model_rank = self.rank // self.model_parallel_size
        super().__init__(base_model_home, model_ckpt_home, aux_home)

    def create_device_map(self):
        model_rank = self.local_rank // self.model_parallel_size
        rank_start = model_rank * self.model_parallel_size

        n_layers = 32
        layer_per_rank = n_layers // self.model_parallel_size
        device_map = {}
        device_map["model.embed_tokens.weight"] = rank_start
        for i in range(n_layers):
            device_idx = rank_start + i // layer_per_rank
            device_map[f"model.layers.{i}"] = device_idx

        device_map["model.norm.weight"] = rank_start + (n_layers-1) // layer_per_rank
        device_map["lm_head.weight"] = rank_start + (n_layers-1) // layer_per_rank

        return device_map

    def prepare(self):
        from sfm.data.sci_data.NlmTokenizer import NlmTokenizer
        if self.local_rank % self.model_parallel_size != 0:
            # wait for rank 0 to download and convert ckpt
            torch.distributed.barrier()
            return None, None

        tokenizer = NlmTokenizer.from_pretrained(self.base_model_home)
        print("vocab size", len(tokenizer))
        if self.local_rank == 0:
            self.download_and_convert_ckpt(self.base_model_home, self.model_ckpt_home, self.aux_home)

        if self.world_size > 1:
            torch.distributed.barrier()

        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(self.aux_home)

        model = load_checkpoint_and_dispatch(
            model,
            self.aux_home,
            device_map=self.create_device_map(),
            no_split_module_classes=['MixtralDecoderLayer'],
            dtype=torch.bfloat16,
            offload_folder=None,
            offload_state_dict=True
        )

        # hack to avoid all past_key_values moved to GPU0
        model._hf_hook.skip_keys = ['past_key_values']
        model.eval()

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
    parser.add_argument('--max_new_tokens', type=int, action='store', help='', default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    print('args:', args)

    init_distributed_mode()

    if args.model_name == "llama2_7b":
        g_cls = Llama2Generator
    elif args.model_name == "llama3_1b":
        g_cls = Base1BGenerator
    elif args.model_name == "llama3_8b":
        g_cls = Base8BGenerator
    elif args.model_name == "mixtral_8x7b":
        g_cls = MixtralGenerator
    elif args.model_name == "mixtral_8x7b-dist":
        g_cls = MixtralDistGenerator

    g = g_cls(args.base_model_root, args.model_ckpt_home, args.aux_home)
    if args.command == "inference":
        g.inference(args.input_file, args.output_dir)
    elif args.command == "generate":
        g.generate(args.n_seq, args.entity, args.output_dir, args.max_new_tokens)
    else:
        raise Exception(f'Unkonwn command: "{args.command}"')


if __name__ == "__main__":

    main()
