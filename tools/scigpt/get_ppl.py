# -*- coding: utf-8 -*-
import os
from copy import deepcopy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer


def show_ckpt(name, ckpt):
    for k, v in ckpt.items():
        if 'dummy' not in k:
            print(name, k, v.shape)


def get_llama_ckpt(llama_path):
    tokenizer = AutoTokenizer.from_pretrained(
        llama_path,
    )
    print(len(tokenizer))
    model = AutoModelForCausalLM.from_pretrained(llama_path)
    model.eval()
    model = model.cuda()
    return tokenizer, model


def get_sfm_ckpt(sfm_path, llama_path):
    tokenizer = SFMDecTokenizer.from_pretrained(
        llama_path,
        prot_spm_path='/blob/shufxi/data/scigpt/ur50bpe/bpe',
        dna_spm_path='/blob/shufxi/data/scigpt/dnabpe/bpe',
        rna_spm_path='/blob/shufxi/data/scigpt/rnabpe/bpe',
    )
    print(len(tokenizer))
    model = AutoModelForCausalLM.from_pretrained(llama_path)
    sfm_model = deepcopy(model)
    model_dict = sfm_model.state_dict()
    ckpt_dict = {}

    layer0 = torch.load(os.path.join(sfm_path, "layer_00-model_states.pt"), map_location=torch.device("cpu"))
    ckpt_dict['model.embed_tokens.weight'] = layer0['embed_tokens.weight']#[:32000]
    show_ckpt('layer0', layer0)
    for l in range(0, 32):
        l_index = str(l + 1).zfill(2)
        layer = torch.load(os.path.join(sfm_path, f"layer_{l_index}-model_states.pt"), map_location=torch.device("cpu"))
        show_ckpt(l_index, layer)
        for k in layer:
            if "dummy" in k or 'rotary_emb' in k:
                continue
            ckpt_dict[f"model.layers.{l}.{k}"] = layer[k]

    layer = torch.load(os.path.join(sfm_path, "layer_33-model_states.pt"), map_location=torch.device("cpu"))
    show_ckpt(33, layer)
    ckpt_dict["model.norm.weight"] = layer["norm.weight"]

    layer = torch.load(os.path.join(sfm_path, "layer_34-model_states.pt"), map_location=torch.device("cpu"))
    show_ckpt(34, layer)
    ckpt_dict["lm_head.weight"] = layer["lm_head.weight"]#[:32000]
    model_dict.update(ckpt_dict)

    sfm_model.resize_token_embeddings(len(tokenizer))
    sfm_model.load_state_dict(model_dict)
    sfm_model.eval()
    sfm_model = sfm_model.cuda()
    return tokenizer, sfm_model


def get_ppl(sentence, tokenizer, model):
    tokens = tokenizer._tokenize(sentence)
    tokens = ['<s>'] + tokens + ['</s>']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss
        ppl = torch.exp(neg_log_likelihood)
    return ppl.item(), neg_log_likelihood.item()


def evaluate(tokenizer, model, data):
    ppls = []
    losses = []
    i = 0
    for sentence in tqdm(data):
        ppl, loss = get_ppl(sentence, tokenizer, model)
        ppls.append(ppl)
        losses.append(loss)
        i += 1
        if i % 1000 == 0:
            print(f"step {i} loss: {sum(losses) / len(losses)}\tppl: {sum(ppls) / len(ppls)}")
    ppl = sum(ppls) / len(ppls)
    loss = sum(losses) / len(losses)
    return ppl, loss


def load_data(data_path):
    with open(data_path, 'r') as f:
        data = [line.strip() for line in f.readlines()]
    return data


def main():
    llama_path = '/hai1/mfm/ds_dataset/llama2/llama-2-7b'
    sfm_path = "/hai1/mfm/shufxi/scigpt/7bv3/stageA_prot_e10_bs512_emb_8xG8H100/global_step11562/"
    data_path = "/blob/renqian/data/sfm/ur90/valid.uniref90.shuf.10k"
    data = load_data(data_path)

    # evaluate llama
    # tokenizer, model = get_llama_ckpt(llama_path)
    # ppl = evaluate(tokenizer, model, data)
    # print(f"llama ppl: {ppl}")

    # evaluate bio0
    tokenizer, model = get_sfm_ckpt(sfm_path, llama_path)
    ppl, loss = evaluate(tokenizer, model, data)
    print(f"sfm loss: {loss}\tppl: {ppl}")


if __name__ == "__main__":
    main()
