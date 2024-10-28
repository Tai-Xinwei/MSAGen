# -*- coding: utf-8 -*-
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
import torch
import os
from tqdm import tqdm
import pickle as pkl
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='/msralaphilly2/ml-la/yinxia/wu2/shared/SFM/SFM.overall.data/instruction_tuning/valid.instruct.txt')
parser.add_argument('--output', type=str, default='/msralaphilly2/ml-la/yinxia/wu2/backup/SFM_for_material.20240430/instruct_mat_beam4_07082024.pkl')
args = parser.parse_args()

tokenizer_home = '/sfmdataeastus2/nlm/llama/llama-2-7b'
tokenizer = SFMDecTokenizer.from_pretrained(
    tokenizer_home,
    prot_spm_path='/msralaphilly2/ml-la/shufxi/data/scigpt/ur50bpe/bpe',
    dna_spm_path='/msralaphilly2/ml-la/shufxi/data/scigpt/dnabpe/bpe',
    rna_spm_path='/msralaphilly2/ml-la/shufxi/data/scigpt/rnabpe/bpe',
)

from transformers import AutoTokenizer, AutoModelForCausalLM

#ckpt_home = '/msralaphilly2/ml-la/yinxia/scigpt/7bv3/unifyall_v3_full_run1/global_step17984'
ckpt_home = '/msralaphilly2/ml-la/yinxia/scigpt/7bv3/unifyall_v3_purematerial_run1/global_step10330'

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

import re
def chat(prompt, beam=4, retsize=1):
    prompt = 'Instruction: ' + prompt + '\n\n\nResponse:'
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    inputs = inputs.cuda()
    outputs = model.generate(
        inputs,
        num_beams=beam,
        max_new_tokens=300,
        num_return_sequences=beam,
        do_sample=False,
    )
    if retsize == 1:
        return tokenizer.decode(outputs[0]).split('Response:')[-1]
    return [tokenizer.decode(outputs[i]).split('Response:')[-1] for i in range(retsize)]



def chat_sample(prompt, beam=4, retsize=1):
    prompt = 'Instruction: ' + prompt + '\n\n\nResponse:'
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    inputs = inputs.cuda()
    outputs = model.generate(
        inputs,
        max_new_tokens=300,
        num_return_sequences=beam,
        do_sample=True,
        temperature=0.75,
        top_p=0.95,
        repetition_penalty=1.2,
    )
    if retsize == 1:
        return tokenizer.decode(outputs[0]).split('Response:')[-1]
    return [tokenizer.decode(outputs[i]).split('Response:')[-1] for i in range(retsize)]




with open(args.input, 'r', encoding='utf8') as fr:
    src = [e.strip() for e in fr]

return_list = {}

for idx in tqdm(range(len(src)),total=len(src)):
    decoding_results = chat(src[idx], beam=4, retsize=4)
    filtered_decoding_results = []
    for e in decoding_results:
        if '</s>' not in e:
            continue
        e2 = e.split('</s>')[0].strip()
        filtered_decoding_results.append(e2)

    return_list[idx] = [src[idx], filtered_decoding_results]


fw = open(args.output, 'wb')
pkl.dump(return_list, fw)
fw.close()
