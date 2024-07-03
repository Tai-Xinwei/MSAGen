# -*- coding: utf-8 -*-
import os
import re
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from sfm.logging import logger
from sfm.models.scigpt.config import ScigptConfig
from sfm.models.scigpt.scigpt import ScigptModel
from sfm.utils import arg_utils
from sfm.utils.science_tokens import SCIENCE_TAG_TOKENS, SCIENCE_TOKENS


def init_tokenizer(tokenizer_path):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.tag_re = re.compile(f'{"|".join(SCIENCE_TAG_TOKENS)}')
    tokenizer.smiles_re = re.compile(
        "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )

    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "unk_token": "<unk>",
        },
    )

    tokenizer.add_tokens(SCIENCE_TAG_TOKENS)
    tokenizer.add_tokens(SCIENCE_TOKENS)
    extra_tokens = []
    # protein
    for i in range(26):
        extra_tokens.append(f"<a>{chr(65 + i)}")

    # DNA, RNA, including ambiguous bases
    for c in "ACTGURYSWKMBDHVN":
        extra_tokens.append(f"<d>{c}")
        extra_tokens.append(f"<r>{c}")

    # materials, non-elements
    for c in "0123456789()+-":
        extra_tokens.append(f"<i>{c}")
    for i in range(26):
        extra_tokens.append(f"<i>{chr(65 + i)}")
        extra_tokens.append(f"<i>{chr(97 + i)}")

    tokenizer.add_tokens(extra_tokens)
    tokenizer.split_special_tokens = (
        True  # Ensure _tokenize() can access special tokens
    )

    logger.info(f"Tokenizer has {len(tokenizer)} tokens")

    # tokenizer.save_pretrained("/home/yinxia/zekuntmp/SFM_framework/Mixtral-llama3-8B-v0.1/tokenizer.model")


smiles_re = re.compile(
    "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def _tokenize_entity(tokenizer, text: str, prefix: str, tok: str):
    if tok == "smiles":
        tokens = smiles_re.findall(text)
    elif tok == "space":
        tokens = text.split(" ")
    else:
        tokens = list(text)

    ret = []
    for t in tokens:
        if t == "":
            continue

        if t.startswith("<sg") and t.endswith(">"):
            # No <i> tag for subgroups
            ret.append(t)
        else:
            ret.append(f"<{prefix}>{t}")
    if len(ret) > 0:
        if tokenizer.convert_tokens_to_ids(ret).count(128257):
            logger.info("unk:{}".format(str(ret)))
    return ret


def _tokenize_by_tag(tokenizer, span, tag, **kwargs):
    if tag in ["mol", "product", "reactants", "fragA", "fragB"]:
        tokens = _tokenize_entity(tokenizer, span, "m", tok="smiles")
    elif tag in ["protein", "antibody"]:
        tokens = _tokenize_entity(tokenizer, span, "a", tok="list")
    elif tag == "material":
        tokens = _tokenize_entity(tokenizer, span, "i", tok="space")
    elif tag == "dna":
        tokens = _tokenize_entity(tokenizer, span, "d", tok="list")
    elif tag == "rna":
        tokens = _tokenize_entity(tokenizer, span, "r", tok="list")
    else:
        tokens = tokenizer.tokenize(span, **kwargs)

    return tokens


def _tokenize(tokenizer, text, **kwargs):
    result = []
    cur_tag = None
    last_idx = 0

    known_tags = [
        "mol",
        "product",
        "reactants",
        "protein",
        "antibody",
        "material",
        "dna",
        "fragA",
        "fragB",
        "rna",
    ]
    tag_re = re.compile(f'{"|".join(SCIENCE_TAG_TOKENS)}')
    for match in tag_re.finditer(text):
        start, end = match.span()
        match_str = match.group()

        if match_str.startswith("</"):
            tag = match_str[2:-1]
            if tag not in known_tags:
                continue

            if tag != cur_tag:
                raise ValueError(f"Tag mismatch: {tag} != {cur_tag} in '{text}'")

            span = text[last_idx:start].strip()
            tokens = _tokenize_by_tag(tokenizer, span, tag, **kwargs)

            result.extend([t for t in tokens if t] + [f"</{tag}>"])
            cur_tag = None
        else:
            tag = match_str[1:-1]
            if tag not in known_tags:
                continue

            if cur_tag is not None:
                raise ValueError(f"Nested tag: {tag} in '{text}'")

            cur_tag = tag
            span = text[last_idx:start].strip()
            tokens = tokenizer.tokenize(span, **kwargs)

            result.extend([t for t in tokens if t] + [f"<{tag}>"])

        last_idx = end

    if last_idx < len(text):
        span = text[last_idx:].strip()
        tokens = _tokenize_by_tag(tokenizer, span, cur_tag, **kwargs)
        result.extend(tokens)

    return result


def convert_tokens_to_string(tokenizer, tokens):
    """Converts a sequence of tokens (string) in a single string."""
    for i in range(len(tokens)):
        for tag in ["<m>", "<a>", "<i>"]:
            tokens[i] = tokens[i].replace(tag, "")

    return tokenizer.convert_tokens_to_string(tokens)


def tokenize(line):
    global tokenizer
    try:
        tokens = _tokenize(tokenizer, line)
        # print("tokens:", tokens)
        tokens = (
            [tokenizer.bos_token_id]
            + tokenizer.convert_tokens_to_ids(tokens)
            + [tokenizer.eos_token_id]
        )
        return tokens

    except:
        # some lines have weird tags that can't be tokenized
        return [], None


class NLMGenerator:
    def __init__(self, ckpt_home, tokenizer_home):
        init_tokenizer(tokenizer_home)

        def get_args():
            parser = ArgumentParser()
            cfg_classes = [ScigptConfig]
            parser = arg_utils.add_dataclass_to_parser(cfg_classes, parser)
            args = parser.parse_args(args=[])
            args.load_ckpt = False
            args.strategy = "DDP"
            args.encoder_layers = 24
            args.encoder_embed_dim = 8192
            args.encoder_ffn_embed_dim = 5120
            args.encoder_attention_heads = 32
            args.infer = True
            args.bf16 = True
            args.llm_model_name_or_path = tokenizer_home

            args.vocab_size = 130304
            return args

        args = get_args()

        # Loading the extended trained model
        ckpt_dict = {}

        model = ScigptModel(args)
        model.decoder.resize_token_embeddings(args.vocab_size)
        model_dict = model.state_dict()
        # print(f"model_dict: {model_dict.keys()}")
        # print(model_dict['decoder.model.layers.0.mlp.gate_proj.weight'].shape)
        # print(model_dict['decoder.model.layers.0.mlp.up_proj.weight'].shape)
        weight1_size = model_dict["decoder.model.layers.0.mlp.gate_proj.weight"].size(0)
        weight2_size = model_dict["decoder.model.layers.0.mlp.up_proj.weight"].size(0)
        layer0 = torch.load(
            os.path.join(ckpt_home, "layer_00-model_00-model_states.pt"),
            map_location=torch.device("cpu"),
        )
        for k, v in layer0.items():
            if k == "word_embeddings.weight":
                ckpt_dict["decoder.model.embed_tokens.weight"] = v

        for l in range(0, 32):
            l_index = str(l + 1).zfill(2)
            layer = torch.load(
                os.path.join(ckpt_home, f"layer_{l_index}-model_00-model_states.pt"),
                map_location=torch.device("cpu"),
            )
            for k in layer:
                if "dummy" in k or "rotary_emb" in k:
                    continue
                if k == "self_attention.layernorm_qkv.layer_norm_weight":
                    ckpt_dict[
                        f"decoder.model.layers.{l}.input_layernorm.weight"
                    ] = layer[k]
                elif k == "self_attention.layernorm_qkv.query_weight":
                    ckpt_dict[
                        f"decoder.model.layers.{l}.self_attn.q_proj.weight"
                    ] = layer[k]
                elif k == "self_attention.layernorm_qkv.key_weight":
                    ckpt_dict[
                        f"decoder.model.layers.{l}.self_attn.k_proj.weight"
                    ] = layer[k]
                elif k == "self_attention.layernorm_qkv.value_weight":
                    ckpt_dict[
                        f"decoder.model.layers.{l}.self_attn.v_proj.weight"
                    ] = layer[k]
                elif k == "self_attention.proj.weight":
                    ckpt_dict[
                        f"decoder.model.layers.{l}.self_attn.o_proj.weight"
                    ] = layer[k]
                elif k == "layernorm_mlp.layer_norm_weight":
                    ckpt_dict[
                        f"decoder.model.layers.{l}.post_attention_layernorm.weight"
                    ] = layer[k]
                elif k == "layernorm_mlp.fc1_weight":
                    weight1, weight2 = torch.split(
                        layer[k], [weight1_size, weight2_size], dim=0
                    )
                    ckpt_dict[
                        f"decoder.model.layers.{l}.mlp.gate_proj.weight"
                    ] = weight1
                    ckpt_dict[f"decoder.model.layers.{l}.mlp.up_proj.weight"] = weight2
                elif k == "layernorm_mlp.fc2_weight":
                    ckpt_dict[f"decoder.model.layers.{l}.mlp.down_proj.weight"] = layer[
                        k
                    ]
            del layer

        layer = torch.load(
            os.path.join(ckpt_home, "layer_33-model_00-model_states.pt"),
            map_location=torch.device("cpu"),
        )
        ckpt_dict["decoder.model.norm.weight"] = layer["norm.weight"]

        layer = torch.load(
            os.path.join(ckpt_home, "layer_34-model_00-model_states.pt"),
            map_location=torch.device("cpu"),
        )
        ckpt_dict["decoder.lm_head.weight"] = layer["lm_head.weight"]

        # print(f"ckpt_dict: {ckpt_dict.keys()}")
        model_dict.update(ckpt_dict)
        model.load_state_dict(model_dict)

        device = torch.device("cuda")
        self.model = model.to(torch.bfloat16).to(device)
        self.model.eval()

        global tokenizer
        self.tokenizer = tokenizer
        self.generation_dict = {
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
        }

    def chat(self, input_str, response_only=True, do_sample=False, **kwargs):
        kwargs.update(self.generation_dict)
        prompt = f"Instruction: {input_str.strip()}\n\n\nResponse:"
        input_ids = torch.tensor(tokenize(prompt)[:-1]).cuda().unsqueeze(0)  # rm eos
        if "max_new_tokens" in kwargs:
            max_new_tokens = kwargs.pop("max_new_tokens")
        else:
            max_new_tokens = 100

        if do_sample:
            outputs = self.model.decoder.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                num_return_sequences=4,
                do_sample=True,
                temperature=0.75,
                top_p=0.95,
                **kwargs,
            )
        else:
            outputs = self.model.decoder.generate(
                input_ids,
                num_beams=4,
                max_new_tokens=max_new_tokens,
                num_return_sequences=4,
                do_sample=do_sample,
                **kwargs,
            )

        out_list = []
        for out in outputs:
            s = self.tokenizer.decode(out)
            if response_only:
                segs = s.split("Response:")
                s = (
                    segs[1]
                    .strip()
                    .replace("<m>", "")
                    .replace("<a>", "")
                    .replace("<i>", "")
                )
            segs = s.split("<|end_of_text|>")
            out_list.append(segs[0].strip())
        return out_list

    def extract_first_token_prob(self, file_name):
        from sfm.data.sci_data.dataset import RawTextSciDatasetwithAltTokenizer

        dataset = RawTextSciDatasetwithAltTokenizer(
            file_name,
            self.tokenizer,
            tokenize,
            conditional_generation=True,
            use_template=True,
            max_len=8192,
            only_prompt=True,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=dataset.collate,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        from tqdm import tqdm

        from sfm.utils.move_to_device import move_to_device

        with torch.no_grad():
            buffer = []
            for data in tqdm(data_loader):
                device = torch.device("cuda")
                data = move_to_device(data, device)

                outputs = self.model.decoder(
                    data[0][0],
                    data[0][1],
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
                yes_probs = next_token_scores[:, yes_id]
                no_probs = next_token_scores[:, no_id]
                confidences = yes_probs / (yes_probs + no_probs)
                buffer.extend(confidences.tolist())

        return buffer
