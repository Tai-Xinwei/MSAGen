# -*- coding: utf-8 -*-
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from tqdm import tqdm
from argparse import ArgumentParser

import re

def main():
    parser = ArgumentParser()
    parser.add_argument('--step', type=int, default=4199)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='test')

    args = parser.parse_args()

    tokenizer_home = '/hai1/ds_dataset/llama2/llama-2-7b'

    tokenizer = SFMDecTokenizer.from_pretrained(tokenizer_home)

    ckpt_home = f'/hai1/shufxi/scigpt/7b/stageB/global_step{args.step}/'
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
                    smiles = re.search('<mol>(.*)</mol>', detoked).group(1)
                    smiles = smiles.replace('<m>', '').replace(' ', '')
                except:
                    smiles = ''

                f.write(smiles + '\n')

if __name__ == '__main__':
    main()
