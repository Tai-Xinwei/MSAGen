# -*- coding: utf-8 -*-
#%%
protein_descriptions = [
    "The protein sequence is composed of a unique arrangement of 20 different amino acids.",
    "Its primary structure is defined by the linear sequence of amino acids connected by peptide bonds.",
    "The sequence contains several hydrophobic residues that contribute to its tertiary structure.",
    "There are multiple phosphorylation sites indicative of regulatory functions.",
    "The sequence includes a conserved domain characteristic of a specific family of enzymes.",
    "It exhibits a high degree of sequence similarity to known proteins in other species, suggesting evolutionary conservation.",
    "The presence of a signal peptide at the N-terminus suggests it is secreted from the cell.",
    "A transmembrane region within the sequence indicates it may be a membrane-bound protein.",
    "The protein sequence contains a zinc finger motif, implying a role in DNA binding.",
    "Cysteine residues in close proximity suggest the formation of disulfide bridges, stabilizing the protein structure.",
    "The sequence includes a stretch of glutamine residues, known as a polyQ tract.",
    "It contains several glycosylation sites, which may affect its folding and stability.",
    "The sequence is rich in proline, which could introduce sharp bends and affect the protein's folding.",
    "A leucine zipper motif within the sequence suggests it may be involved in dimerization.",
    "The protein has a low complexity region, often associated with disordered regions.",
    "The sequence contains ATP-binding motifs, indicating it may have enzymatic activity.",
    "There are multiple splice variants of this protein, each with a unique sequence and potential function.",
    "The protein is predicted to undergo post-translational modifications, altering its activity or localization.",
    "Sequence analysis suggests the protein interacts with several other proteins, playing a role in a complex cellular pathway.",
    "The sequence includes a stop codon, indicating the end of the protein coding region."
]
# %%
import random
random.seed(0)
# %%
ur50_input = []
with open('/tmp/ur50.train.seqs.pended.new.txt', 'r') as f:
    for line in f:
        ur50_input.append(line.strip())

prefix_set = dict()
for seq in ur50_input:
    prefix = seq[:10]
    if prefix not in prefix_set:
        prefix_set[prefix] = 0
    prefix_set[prefix] += 1

# %%
prefix_list = []
for k, v in prefix_set.items():
    if v > 5:
        prefix_list.append(k)

# %%
print(len(prefix_list))
print(random.choice(prefix_list))

# %%
prefix_list = []
with open('/blob/yinxia/wu2/shared/SFM/SFM.overall.data/X-text/protein-text.nonwrap-updated.txt', 'r') as f:
    for line in f:
        end = line.find('<protein>')
        if end == -1:
            continue
        prefix_list.append(line[:end])

print(len(prefix_list))
print(random.choice(prefix_list))
# %%
print(prefix_list[0])

# %%
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
# %%
tokenizer_home = '/hai1/ds_dataset/llama2/llama-2-7b'
tokenizer = SFMDecTokenizer.from_pretrained(
    tokenizer_home,
    prot_spm_path='/blob/shufxi/data/scigpt/ur50bpe/bpe',
    dna_spm_path='/blob/shufxi/data/scigpt/dnabpe/bpe'
)
# %%
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

model.cuda()
# %%
def make_prompt():
    # # prompt = 'The protein sequence is composed of a unique arrangement of 20 different amino acids. <protein>' + random.choice(prefix_list)
    # # input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    # # print(prompt, input_ids)
    # # return input_ids

    # # random select two different desc and concat them
    # descs = random.sample(protein_descriptions, 2)
    # desc = ' '.join(descs)
    # seq_len = random.uniform(30, 100)

    # desc += f' The sequence with {seq_len:} amino acids is <protein>'

    # desc = "Construct a protein sequence with the desired structural and functional characteristics. 1. The protein must contain a signal peptide for proper functionality. 2. The protein should be designed to localize specifically to nucleus in order to perform its role in kinetochore assembly more effectively. 3. For general function, the protein need meet that Probable component of a centromeric complex involved in assembly of kinetochore proteins, mitotic progression and chromosome segregation. The protein sequence incorporates <protein>"
    desc = random.choice(prefix_list) + ' <protein>'
    # print(desc)
    input_ids = tokenizer(desc, return_tensors='pt').input_ids.cuda()
    return input_ids

def filter_seq(seq):
    cnt = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            cnt += 1
        else:
            cnt = 1
        if cnt > 5:
            return False
    return True

def generate(input_ids, beam=1, num_ret=1, top_p=0.9, top_k=10, temperature=1):
    output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=100,
        num_beams=beam,
        num_return_sequences=num_ret,
        # length_penalty=0.6,
        # top_p=top_p,
        # top_k=top_k,
        temperature=temperature,
        # repetition_penalty=1.2,
        # penalty_alpha=0.6
    )
    ret = []
    for s in output:
        # print(s)
        if not filter_seq(s):
            continue
        s = tokenizer.decode(s[2:])
        left = s.find('<protein>') + len('<protein>')
        prot = s[left:s.find('</protein>')]
        # print(prot)
        prot = prot.replace('<a>', '').replace(' ', '')

        valid=True
        for c in prot:
            if c not in 'ACDEFGHIKLMNPQRSTVWY':
                valid=False
                break
        if not valid:
            continue

        ret.append(prot)
    return ret

# random.seed(0)

for seq in generate(make_prompt()):
    print(seq)
# %%
from tqdm import tqdm
# %%



# %%
prot_set = set()
total = 5000
bar = tqdm(total=total)
with open('prot.gen.txt', 'w') as f:
    cnt = 0
    while cnt < total:
        input_ids = make_prompt()
        for seq in generate(input_ids):
            if not seq:
                continue
            if seq in prot_set:
                continue
            prot_set.add(seq)
            f.write(seq + '\n')
            cnt += 1
            bar.update(1)
bar.close()

# %%
