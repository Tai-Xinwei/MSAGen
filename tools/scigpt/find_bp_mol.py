# -*- coding: utf-8 -*-
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
import torch
import os
from sfm.logging import logger

tokenizer_home = '/hai1/ds_dataset/llama2/llama-2-7b'
tokenizer = SFMDecTokenizer.from_pretrained(
    tokenizer_home,
    prot_spm_path='/blob/shufxi/data/scigpt/ur50bpe/bpe',
    dna_spm_path='/blob/shufxi/data/scigpt/dnabpe/bpe'
)

from transformers import AutoTokenizer, AutoModelForCausalLM
ckpt_home = '/hai1/shufxi/scigpt/7bv2/stageB/global_step26999/'

def show_ckpt(name, ckpt):
    for k, v in ckpt.items():
        if 'dummy' not in k:
            print(name, k, v.shape)

model = AutoModelForCausalLM.from_pretrained(tokenizer_home)

model_dict = model.state_dict()
ckpt_dict = {}
layer0 = torch.load(os.path.join(ckpt_home, "layer_00-model_states.pt"), map_location=torch.device("cpu"))
ckpt_dict['model.embed_tokens.weight'] = layer0['embed_tokens.weight']
show_ckpt('layer0', layer0)

for l in range(0, 32):
    l_index = str(l + 1).zfill(2)
    layer = torch.load(os.path.join(ckpt_home, f"layer_{l_index}-model_states.pt"), map_location=torch.device("cpu"))
    show_ckpt(l_index, layer)
    for k in layer:
        if "dummy" in k or 'rotary_emb' in k:
            continue
        ckpt_dict[f"model.layers.{l}.{k}"] = layer[k]
layer = torch.load(os.path.join(ckpt_home, "layer_33-model_states.pt"), map_location=torch.device("cpu"))
show_ckpt(33, layer)
ckpt_dict["model.norm.weight"] = layer["norm.weight"]

layer = torch.load(os.path.join(ckpt_home, "layer_34-model_states.pt"), map_location=torch.device("cpu"))
show_ckpt(33, layer)
ckpt_dict["lm_head.weight"] = layer["lm_head.weight"]
model_dict.update(ckpt_dict)

model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(model_dict)

model = model.cuda()

data = []
known_mols = set()
with open('/home/shufxi/pubchem_bp.csv', 'r') as f:
    for line in f:
        items = line.strip().split(',')
        if '.' in items[0]:
            continue
        if items[1] == 'bp':
            continue
        data.append((items[0], float(items[1])))
        known_mols.add(items[0])

import random
random.seed(42)
random.shuffle(data)

def make_prompt(k=10, new_bp=40):
    ret = ''
    for item in random.choices(data, k=k):
        ret += f'A moleclue with boiling point {item[1]:.4f} is <mol>{item[0]}</mol>. '

    ret += f'A moleclue with boiling point {new_bp:.4f} is <mol>'

    return ret

found = []
with open('pubchem_bp.txt', 'w') as f:
    while len(found) < 5000:
        prompt = make_prompt(k=20, new_bp=40 + 10*random.random())
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        output = model.generate(input_ids, do_sample=True, top_p=0.9, max_new_tokens=50, num_return_sequences=1)
        for s in output:
            s = tokenizer.decode(s)
            mol = s[s.rfind('<mol>') + len('<mol>'):s.rfind('</mol>')]
            mol = mol.replace('<m>', '').replace(' ', '')
            if not mol:
                continue

            if mol in known_mols:
                continue

            if mol in found:
                continue
            found.append(mol)

            logger.info(f'{len(found)} {mol}')
            f.write(f'{mol}\n')
            f.flush()
