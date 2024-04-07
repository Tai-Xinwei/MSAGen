# -*- coding: utf-8 -*-
# %%
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
import torch
import os

# %%
tokenizer_home = '/hai1/ds_dataset/llama2/llama-2-7b'
tokenizer = SFMDecTokenizer.from_pretrained(
    tokenizer_home,
    prot_spm_path='/blob/shufxi/data/scigpt/ur50bpe/bpe',
    dna_spm_path='/blob/shufxi/data/scigpt/dnabpe/bpe',
    rna_spm_path='/blob/shufxi/data/scigpt/rnabpe/bpe',
)

# %%
tokenizer.tokenize('<rna>GCCGGCGUAGCUCAGUUGGUAGAGCAAUUGUUUUGUAAACAAAAGGUCGGGGGUUCGAUUCCUCUCGCCGGCU</rna>')

# %%
s = tokenizer.tokenize('The <protein>EMMFEQTFKNID</protein> and the compound <mol>CCC</mol> can interact with each other.')
print(' '.join(s))

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
# ckpt_home = '/hai1/shufxi/scigpt/7bv2/stageB/global_step26999/'

# ckpt_home = '/hai1/shufxi/scigpt/7bv3/stageB/global_step11999'
# ckpt_home = '/blob/shufxi/scigpt/7bv3/inst/20240227121523/global_step3585/'
ckpt_home = '/blob/shufxi/scigpt/7bv3/prot/20240228025826/global_step11715'

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

# %%
model = model.cuda()

# %%
input_ids = tokenizer('<protein>', return_tensors="pt").input_ids.cuda()
output = model.generate(
    input_ids,
    num_beams=4,
    max_new_tokens=300,
    num_return_sequences=4,
    return_dict_in_generate=True,
    output_scores=True,
    do_sample=True,
    top_p=0.95,
)

for i in range(output.sequences.shape[0]):
    s = tokenizer.decode(output.sequences[i])
    s = s.replace(' <a>', '').replace(' <m>', '')
    print(s)
    print("=====================================")

# %%
def gen_seq():
    input_ids = tokenizer('<protein>', return_tensors="pt").input_ids.cuda()
    output = model.generate(
        input_ids,
        num_beams=4,
        max_new_tokens=300,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.5
    )

    s = tokenizer.decode(output.sequences[0])
    start_idx = s.find('<protein>') + len('<protein>')
    end_idx = s.find('</protein>')
    s = s[start_idx:end_idx].replace(' <a>', '').strip()
    return s

gen_seq()

# %%
from tqdm import tqdm

# %%
with open('prot_gen.txt', 'w') as f:
    for i in tqdm(range(3000)):
        s = gen_seq()
        f.write(s + '\n')
