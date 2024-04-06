# -*- coding: utf-8 -*-
import multiprocessing
import sentencepiece as spm
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import struct
import random
import numpy as np
import os, sys
import lmdb
import pickle as pkl
from multiprocessing import Pool
import functools
import pickle

from copy import deepcopy
from joblib import Parallel, delayed
from sfm.utils.science_tokens import SCIENCE_TAG_TOKENS
from transformers import AutoTokenizer
from sfm.data.prot_data.util import bstr2obj, obj2bstr
import torch

def main():
    ckpt_path = "/fastdata/peiran/bfm/checkpoints/bfm650m_data3_maskspan3_ddp4e5d16mask020drop1L1536B2k_bpev2pairv4_bert2_128A100_adam2/checkpoint_E144.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    new_ckpt = {}
    new_ckpt["model"] = {}
    for k in ckpt["model"].keys():
        new_key = k.replace("net.sentence_encoder.", "")
        new_key = new_key.replace("net.", "")

        print(f"change from old key {k} to new key {new_key}")
        new_ckpt["model"][new_key] = ckpt["model"][k]

    torch.save(new_ckpt, ckpt_path.replace(".pt", "_new.pt"))



if __name__ == "__main__":
    main()
