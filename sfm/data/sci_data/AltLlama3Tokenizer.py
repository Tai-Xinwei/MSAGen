# -*- coding: utf-8 -*-
import mmap
import multiprocessing as mp
import os
import re
import struct
from argparse import ArgumentParser
from multiprocessing import Lock

import numpy as np
import tqdm
from transformers import AutoTokenizer

from sfm.data.sci_data.NlmTokenizer import NlmTokenizer
from sfm.logging import logger
from sfm.utils.science_tokens import SCIENCE_TAG_TOKENS, SCIENCE_TOKENS

lock = Lock()


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
