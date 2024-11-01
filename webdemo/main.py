# -*- coding: utf-8 -*-
import json
import logging
import os
import re
import shutil
from functools import partial
from threading import Thread

import gradio as gr
import safetensors.torch as st
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from examples import dna2rna_tasks, dnapred_tasks, small_mole_tasks
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    TextIteratorStreamer,
)

from sfm.data.sci_data.NlmTokenizer import NlmLlama3Tokenizer, NlmTokenizer

MIXTRAL_BLOB_PATH = os.environ.get(
    "MIXTRAL_BLOB_PATH", "/home/shufxi/nlm/Mixtral-8x7B-v0.1"
)

LLAMA_BLOB_PATH = os.environ.get(
    "LLAMA_BLOB_PATH", "/home/shufxi/nlm/llama1b/inst/20240705132320/global_step44960"
)

CKPT = os.environ.get(
    "CKPT", "/home/shufxi/nlm/shufxi/nlm/8x7b/inst/20240705132320/global_step44960"
)
LOCAL_PATH = os.environ.get("LOCAL_PATH", "/dev/shm/nlmoe")
DEMO_MODE = os.environ.get("DEMO_MODE", "chat").lower()
DEMO_NAME = os.environ.get("DEMO_NAME", "sfm_seq")
SERVER_PORT = int(os.environ.get("SERVER_PORT", 8234))

N_MODEL_PARALLEL = int(os.environ.get("N_MODEL_PARALLEL", 2))
MODEL_START_DEVICE = int(os.environ.get("MODEL_START_DEVICE", 0))

MODEL_TYPE = os.environ.get("MODEL_TYPE", "mixtral_8x7b")


def get_logger():
    # Set up the logger
    logger = logging.getLogger("ChatLogger")
    logger.setLevel(logging.INFO)
    # Create handlers
    console_handler = logging.StreamHandler()  # Logs to the console
    file_handler = logging.FileHandler(
        "./webdemo/webdemo_logs/{}.txt".format(DEMO_NAME)
    )  # Logs to a file

    # Create formatter and add it to handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def download_and_convert_ckpt(mixtral_blob_path, nlm_blob_path, local_path):
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


def create_device_map(n_parallel=2, start_device=0):
    n_layers = 32
    layer_per_rank = n_layers // n_parallel
    device_map = {}
    device_map["model.embed_tokens.weight"] = 0 + start_device
    for i in range(n_layers):
        device_idx = i // layer_per_rank + start_device
        device_map[f"model.layers.{i}"] = device_idx

    device_map["model.norm.weight"] = (n_layers - 1) // layer_per_rank + start_device
    device_map["lm_head.weight"] = (n_layers - 1) // layer_per_rank + start_device

    return device_map


def load_llama3_model(config, ckpt_path):
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


if MODEL_TYPE == "mixtral_8x7b":
    tokenizer = NlmTokenizer.from_pretrained(MIXTRAL_BLOB_PATH)
    download_and_convert_ckpt(MIXTRAL_BLOB_PATH, CKPT, LOCAL_PATH)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(LOCAL_PATH)

    model = load_checkpoint_and_dispatch(
        model,
        LOCAL_PATH,
        device_map=create_device_map(
            n_parallel=N_MODEL_PARALLEL, start_device=MODEL_START_DEVICE
        ),
        no_split_module_classes=["MixtralDecoderLayer"],
        dtype=torch.bfloat16,
        offload_folder=None,
        offload_state_dict=True,
    )
    model._hf_hook.skip_keys = ["past_key_values"]
    model.eval()
elif MODEL_TYPE == "llama1b":
    config = LlamaConfig.from_json_file(LLAMA_BLOB_PATH + "/config.json")
    config.hidden_size = 2048
    config.intermediate_size = 5504
    config.num_hidden_layers = 16
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    config.max_position_embeddings = 8192
    config.tokens_per_sample = 8192
    config.vocab_size = 130305
    config.rope_theta = 1000000.0
    tokenizer = NlmLlama3Tokenizer.from_pretrained(LLAMA_BLOB_PATH)

    model = load_llama3_model(config, CKPT)
    model.eval()
elif MODEL_TYPE == "llama8b":
    config = LlamaConfig.from_json_file(LLAMA_BLOB_PATH + "/config.json")
    config.vocab_size = 130304
    tokenizer = NlmLlama3Tokenizer.from_pretrained(LLAMA_BLOB_PATH)

    model = load_llama3_model(config, CKPT)
    model.eval()
elif MODEL_TYPE == "ph3_mini":
    # Load the model and tokenizer from Hugging Face for Ph3-mini
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/yeqi/cache/hf_models/Phi-3.5-mini-instruct/"
    )
    model = AutoModelForCausalLM.from_pretrained(
        "/data/yeqi/cache/hf_models/Phi-3.5-mini-instruct/"
    )
    model.eval()  # Set to evaluation mode
    model.cuda()  # Move model to GPU if available
else:
    raise ValueError(f"Unsupported model type {MODEL_TYPE}")


def close_last_tag(xml_string):
    opening_tags = re.findall(r"<(\w{2,})>", xml_string)
    closing_tags = re.findall(r"</(\w{2,})>", xml_string)

    if len(opening_tags) > len(closing_tags):
        last_tag = opening_tags[-1]
        xml_string += f"</{last_tag}>"

    return xml_string


def compress_material(text):
    def replace_material(match):
        content = match.group(1)
        elements = content.split()
        if len(elements) == 0:
            return match.group(0)

        # Keep the space group tag
        if elements[-1].startswith("<sg"):
            sg_tag = elements[-1]
            elements = elements[:-1]
        else:
            sg_tag = ""

        compressed = []
        current_element = None
        count = 0

        for element in elements:
            if element == current_element:
                count += 1
            else:
                if current_element:
                    compressed.append(f"{current_element} {count if count > 1 else ''}")
                current_element = element
                count = 1

        if current_element:
            compressed.append(f"{current_element} {count if count > 1 else ''}")

        # Add the space group tag back
        compressed.append(sg_tag)

        return f"<material> {' '.join(compressed)} </material>"

    pattern = r"<material>(.*?)</material>"
    return re.sub(pattern, replace_material, text)


def wrap_tags_in_code_blocks(text):
    def replace_match(match):
        original = match.group(0)
        backslash = match.group(1)
        tag_name = match.group(2)
        if tag_name.startswith("sg"):
            return original
        if len(tag_name) < 2:
            return original
        if backslash:
            return f"</{tag_name}>`"
        else:
            return f"`<{tag_name}>"

    pattern = r"<([/]?)(\w+)>"
    return re.sub(pattern, replace_match, text)


def post_process_response(text, is_chat=True):
    if is_chat:
        if "Response:" not in text:
            return ""
        text = text.split("Response:")[-1].strip()
    else:
        if text.startswith("<s>"):
            text = text[3:].strip()

    if text.endswith("</s>"):
        text = text[:-4].strip()
    elif text.endswith("<|end_of_text|>"):
        text = text[:-15].strip()

    for prefix in ["<a>", "<m>", "<d>", "r"]:
        text = text.replace(f" {prefix}", "")
    text = text.replace("<i>", "")  # no remove space before <i>

    text = compress_material(text)

    if is_chat:
        text = wrap_tags_in_code_blocks(text)
    return text


def chat(
    message,
    history,
    n_history: int,
    do_sample: bool,
    max_new_tokens: int,
    temperature: float,
    beam_size: int,
    top_p: float,
    logger,
):
    prompt_list = []
    if n_history != 0:
        for inst, resp in history[-n_history:]:
            # resp = resp.replace("&lt;", "<").replace("&gt;", ">")
            resp = close_last_tag(resp)
            prompt_list.append(f"Instruction: {inst}\n\n\nResponse: {resp}")

    prompt_list.append(f"Instruction: {message}\n\n\nResponse:")

    logger.info(f"Prompt: {prompt_list}")

    input_ids = [tokenizer.bos_token_id]
    for i, prompt in enumerate(prompt_list):
        tokens = tokenizer.tokenize(prompt)
        input_ids.extend(tokenizer.convert_tokens_to_ids(tokens))
        if i != len(prompt_list) - 1:
            input_ids.append(tokenizer.eos_token_id)
            input_ids.append(tokenizer.bos_token_id)
    input_ids = torch.Tensor(input_ids).long().unsqueeze(0)
    if MODEL_TYPE != "mixtral_8x7b":
        input_ids = input_ids.cuda()

    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=float(temperature),
        num_beams=int(beam_size),
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    if beam_size == 1:
        streamer = TextIteratorStreamer(tokenizer)
        generation_kwargs["streamer"] = streamer

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for text in streamer:
            generated_text += text
            yield post_process_response(generated_text)
        yield post_process_response(generated_text)
    else:  # beam search is not supported by TextIteratorStreamer
        outputs = model.generate(**generation_kwargs)[0]
        text = tokenizer.decode(outputs)
        yield post_process_response(text)

    logger.info(
        "Response: {}".format(
            [
                post_process_response(generated_text).split("\n\n\nResponse:")[-1],
            ]
        )
    )


def complete(
    message,
    do_sample: bool,
    max_new_tokens: int,
    temperature: float,
    beam_size: int,
    top_p: float,
):
    tokens = tokenizer.tokenize(message)
    input_ids = [tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.Tensor(input_ids).long().unsqueeze(0)

    if MODEL_TYPE != "mixtral_8x7b":
        input_ids = input_ids.cuda()

    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=float(temperature),
        num_beams=int(beam_size),
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    if beam_size == 1:
        streamer = TextIteratorStreamer(tokenizer)
        generation_kwargs["streamer"] = streamer

        def safe_generate(*args, **kwargs):
            try:
                return model.generate(*args, **kwargs)
            except Exception as e:
                print(e)
                return ""

        thread = Thread(target=safe_generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for text in streamer:
            generated_text += text
            yield post_process_response(generated_text, is_chat=False)
        yield post_process_response(generated_text, is_chat=False)
    else:  # beam search is not supported by TextIteratorStreamer
        try:
            outputs = model.generate(**generation_kwargs)[0]
            text = tokenizer.decode(outputs)
            yield post_process_response(text, is_chat=False)
        except Exception as e:
            print(e)
            yield ""


if DEMO_MODE == "chat":
    logger = get_logger()
    demo = gr.ChatInterface(
        partial(chat, logger=logger),
        additional_inputs=[
            gr.Number(label="n_history", value=0),
            gr.Checkbox(label="do_sample", value=True),
            gr.Number(label="max_new_tokens", value=300),
            gr.Slider(
                label="temperature", value=1.0, minimum=0.01, maximum=5.0, step=0.01
            ),
            gr.Number(label="beam_size", value=1),
            gr.Number(label="top_p", value=0.9),
        ],
        chatbot=gr.Chatbot(
            bubble_full_width=False,
            height=580,
        ),
        additional_inputs_accordion=gr.Accordion(
            label="Chat Settings", visible=True, open=True
        ),
    )
elif DEMO_MODE == "chat_dev":
    tasks = dna2rna_tasks + dnapred_tasks + small_mole_tasks
    examples = [[task_str, 0, True, 300, 1.0, 1, 0.9] for task_str in tasks]
    logger = get_logger()
    demo = gr.ChatInterface(
        partial(chat, logger=logger),
        examples=examples,
        additional_inputs=[
            gr.Number(label="n_history", value=0),
            gr.Checkbox(label="do_sample", value=True),
            gr.Number(label="max_new_tokens", value=300),
            gr.Slider(
                label="temperature", value=1.0, minimum=0.01, maximum=5.0, step=0.01
            ),
            gr.Number(label="beam_size", value=1),
            gr.Number(label="top_p", value=0.9),
        ],
        chatbot=gr.Chatbot(
            bubble_full_width=False,
            height=580,
        ),
        additional_inputs_accordion=gr.Accordion(
            label="Chat Settings", visible=True, open=True
        ),
    )
else:  # DEMO_MODE == "COMPLETE"
    demo = gr.Interface(
        complete,
        inputs=[
            gr.Textbox(label="prompt", lines=8),
            gr.Checkbox(label="do_sample", value=True),
            gr.Number(label="max_new_tokens", value=300),
            gr.Slider(
                label="temperature", value=1.0, minimum=0.01, maximum=5.0, step=0.01
            ),
            gr.Number(label="beam_size", value=1),
            gr.Number(label="top_p", value=0.9),
        ],
        outputs=gr.Textbox(label="response", lines=31),
        allow_flagging="auto",
        flagging_mode="auto",
        flagging_dir="./webdemo/webdemo_logs/{}".format(DEMO_NAME),
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=SERVER_PORT,
        auth=("demo_user", "InternalDemo@2024"),
    )
