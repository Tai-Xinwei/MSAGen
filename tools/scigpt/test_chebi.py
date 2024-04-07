# -*- coding: utf-8 -*-
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from tqdm import tqdm
from argparse import ArgumentParser

import re

cot_examples = \
"""
Description: The molecule is an epoxy(hydroxy)icosatrienoate that is the conjugate base of 11 hydroxy-(14R,15S)-epoxy-(5Z,8Z,12E)-icosatrienoic acid, obtained by deprotonation of the carboxy group; major species at pH 7.3. It is a conjugate base of an 11 hydroxy-(14R,15S)-epoxy-(5Z,8Z,12E)-icosatrienoic acid.
Thoughts:
1. The molecule is an epoxy(hydroxy)icosatrienoate: This indicates the molecule contains an epoxy group (an oxygen atom connected to two carbon atoms) and a hydroxy group (an oxygen atom connected to a hydrogen atom). The "icosatrienoate" part implies a 20-carbon chain with three double bonds.
2. It is the conjugate base of 11 hydroxy-(14R,15S)-epoxy-(5Z,8Z,12E)-icosatrienoic acid: The conjugate base is formed by removing a proton (H+) from the acid. The positions of the groups and the configuration of the double bonds are specified by the numbers and letters.
Answer: the SMILES is <mol>CCCCC[C@@H]1O[C@@H]1/C=C/C(O)C/C=C\C/C=C\CCCC(=O)[O-]</mol>
"""
# Description: The molecule is the stable isotope of tellurium with relative atomic mass 124.904425, 71.4 atom percent natural abundance and nuclear spin 1/2.
# Thoughts:
# The molecule you're referring to is Tellurium-125. Tellurium (Te) is a chemical element with the atomic number 52. It has multiple isotopes, and the number following the element symbol (125 in this case) represents the atomic mass of the isotope. The natural abundance and nuclear spin you mentioned also match with this isotope.
# Answer: the SMILES is <mol>[125Te]</mol>

# Description: The molecule is a tetracyclic triterpenoid that is 4,4,8-trimethylandrosta-1,14-diene substituted by an oxo group at position 3, an acetoxy group at position 7 and a furan-3-yl group at position 17. Isolated from Azadirachta indica, it exhibits antiplasmodial and antineoplastic activities. It has a role as an antineoplastic agent, an antiplasmodial drug and a plant metabolite. It is an acetate ester, a cyclic terpene ketone, a member of furans, a limonoid and a tetracyclic triterpenoid.
# Thoughts:
# 1. The molecule is a tetracyclic triterpenoid: This suggests the molecule has four fused rings and is derived from terpenes, which are a large class of organic compounds.
# 2. It is 4,4,8-trimethylandrosta-1,14-diene substituted by an oxo group at position 3, an acetoxy group at position 7 and a furan-3-yl group at position 17: This describes the specific substitutions on the tetracyclic triterpenoid backbone. An oxo group is a double-bonded oxygen atom (=O), an acetoxy group is a functional group of the form -OC(=O)CH3, and a furan-3-yl group is a five-membered aromatic ring containing an oxygen atom.
# 3. Isolated from Azadirachta indica, it exhibits antiplasmodial and antineoplastic activities: This provides the origin of the molecule and its biological activities. Azadirachtin is one of the most well-known bioactive compounds isolated from Azadirachta indica (Neem tree) and is known for its antiplasmodial and antineoplastic activities.
# 4. It has a role as an antineoplastic agent, an antiplasmodial drug and a plant metabolite. It is an acetate ester, a cyclic terpene ketone, a member of furans, a limonoid and a tetracyclic triterpenoid: This further describes the functional roles and chemical classes of the molecule.
# Answer: the SMILES is <mol>CC(=O)O[C@@H]1C[C@H]2C(C)(C)C(=O)C=C[C@]2(C)[C@H]2CC[C@]3(C)C(=CC[C@H]3c3ccoc3)[C@@]21C</mol>

# Description: The molecule is a member of the class of N-nitrosoureas that is urea in which one of the nitrogens is substituted by methyl and nitroso groups. It has a role as a carcinogenic agent, a mutagen, a teratogenic agent and an alkylating agent.

# Thoughts:
# 1. The molecule is a member of the class of N-nitrosoureas: This indicates that the molecule is a derivative of urea (a compound with two amine groups (-NH2) connected to a carbonyl functional group (C=O)). N-nitrosoureas are a class of molecules where one of the nitrogens in urea is substituted by a nitroso group (-NO).
# 2. Urea in which one of the nitrogens is substituted by methyl and nitroso groups: This further specifies the substitutions on the urea molecule. A methyl group (-CH3) and a nitroso group are attached to one of the nitrogen atoms in urea.
# 3. It has a role as a carcinogenic agent, a mutagen, a teratogenic agent, and an alkylating agent: These are known properties of N-Nitroso-N-methylurea (NMU). NMU is a potent carcinogen and mutagen, it is teratogenic (can cause birth defects), and it acts as an alkylating agent (transfers an alkyl group to other molecules).

# Answer: the SMILES is <mol>CN(N=O)C(N)=O</mol>


# Description: The molecule is an indole phytoalexin that is indole substituted at position 3 by a 1,3-thiazol-2-yl group. It has a role as a metabolite. It is an indole phytoalexin and a member of 1,3-thiazoles.

# Thoughts:
# 1. The molecule is an indole phytoalexin: This indicates that the molecule is a derivative of indole and functions as a phytoalexin, a substance produced by plants that acts as a toxin to invading organisms.
# 2. Indole substituted at position 3 by a 1,3-thiazol-2-yl group: This specifies the substitutions on the indole molecule. Indole is a bicyclic compound consisting of a benzene ring fused to a pyrrole ring. The substitution at position 3 means that a 1,3-thiazol-2-yl group is attached to the carbon at the third position in the indole structure. A 1,3-thiazol-2-yl group is a five-membered ring containing nitrogen and sulfur atoms at the first and third positions, respectively.
# 3. It has a role as a metabolite: This indicates that the molecule is a product of metabolic processes.
# 4. It is an indole phytoalexin and a member of 1,3-thiazoles: This reiterates the structure and function of the molecule.

# Answer: the SMILES is <mol>c1ccc2c(-c3nccs3)c[nH]c2c1</mol>

def main():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_home', type=str, default='/hai1/shufxi/scigpt/7b/stageB/')
    parser.add_argument('--step', type=int, default=4199)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='test')
    parser.add_argument('--use_cot', default=False, action='store_true')

    args = parser.parse_args()
    print("args", args)

    tokenizer_home = '/hai1/ds_dataset/llama2/llama-2-7b'

    # tokenizer = SFMDecTokenizer.from_pretrained(tokenizer_home)

    tokenizer = SFMDecTokenizer.from_pretrained(
        tokenizer_home,
        prot_spm_path='/hai1/shufxi/scigpt/ur50bpe/bpe',
        dna_spm_path= '/hai1/shufxi/scigpt/dnabpe/bpe',
        rna_spm_path='/hai1/shufxi/scigpt/rnabpe/bpe'
    )

    ckpt_home = os.path.join(args.ckpt_home, f'global_step{args.step}')
    model = AutoModelForCausalLM.from_pretrained(tokenizer_home)

    model_dict = model.state_dict()
    ckpt_dict = {}
    layer0 = torch.load(os.path.join(ckpt_home, "layer_00-model_states.pt"), map_location=torch.device("cpu"))
    ckpt_dict['model.embed_tokens.weight'] = layer0['embed_tokens.weight']

    for l in range(0, 32):
        l_index = str(l + 1).zfill(2)
        #print(f"../blob2/checkpoints/MetaLLM-7B-D_NODE4-D_PROC16-instruction-mt/global_step97077/layer_{l_index}-model_states.pt")
        layer = torch.load(os.path.join(ckpt_home, f"layer_{l_index}-model_states.pt"), map_location=torch.device("cpu"))
        for k in layer:
            if "dummy" in k or 'rotary_emb' in k:
                continue
            ckpt_dict[f"model.layers.{l}.{k}"] = layer[k]
    layer = torch.load(os.path.join(ckpt_home, "layer_33-model_states.pt"), map_location=torch.device("cpu"))
    ckpt_dict["model.norm.weight"] = layer["norm.weight"]

    layer = torch.load(os.path.join(ckpt_home, "layer_34-model_states.pt"), map_location=torch.device("cpu"))
    ckpt_dict["lm_head.weight"] = layer["lm_head.weight"]
    model_dict.update(ckpt_dict)

    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(model_dict)

    model.eval()

    model = model.cuda()

    with open('/hai1/shufxi/data/tamgent/chebi/test.textmol.desc', 'r') as f:
        lines = f.readlines()

    test_input = []
    for line in lines:
        if args.use_cot:
            test_input.append(cot_examples.strip() + '\n\nDescription:' + line.strip() + '\nThoughts:\n')
        else:
            test_input.append(line.strip() + ' The SMILES is <mol>')

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(f'{save_dir}/pred_step{args.step}.txt', 'w') as f:
        for i in tqdm(range(0, len(test_input), args.batch_size)):
            batch = []
            max_len = 0
            for sent in test_input[i:i+args.batch_size]:
                sent_ids = tokenizer(sent, return_tensors="pt").input_ids[0].cuda()
                max_len = max(max_len, sent_ids.shape[0])
                batch.append(sent_ids)

            input_ids = torch.ones((len(batch), max_len)).long().cuda() * tokenizer.pad_token_id
            # left pad
            for j, sent in enumerate(batch):
                input_ids[j, -batch[j].shape[0]:] = batch[j]

            # input_ids = tokenizer.batch_encode_plus(batch, padding=True, return_tensors="pt").input_ids.cuda()
            # input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
            output = model.generate(input_ids, do_sample=False, max_new_tokens=200, num_beams=4, num_return_sequences=1)

            for output_i in output:
                detoked = tokenizer.decode(output_i)

                # find mol between <mol> and </mol>
                try:
                    matches = re.findall('<mol>(.*?)</mol>', detoked)
                    smiles = matches[-1] if matches else ''
                    # smiles = re.search('<mol>(.*)</mol>', detoked).group(1)
                    smiles = smiles.replace('<m>', '').replace(' ', '')
                except:
                    smiles = ''

                f.write(smiles + '\n')

if __name__ == '__main__':
    main()
