# -*- coding: utf-8 -*-
import marimo

__generated_with = "0.2.9"
app = marimo.App(layout_file="layouts/prot_gen_demo.grid.json")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
    import torch
    import os
    return SFMDecTokenizer, os, torch


@app.cell
def __(SFMDecTokenizer):
    tokenizer_home = '/hai1/ds_dataset/llama2/llama-2-7b'
    tokenizer = SFMDecTokenizer.from_pretrained(
        tokenizer_home,
        prot_spm_path='/blob/shufxi/data/scigpt/ur50bpe/bpe',
        dna_spm_path='/blob/shufxi/data/scigpt/dnabpe/bpe',
        rna_spm_path='/blob/shufxi/data/scigpt/rnabpe/bpe',
    )
    return tokenizer, tokenizer_home


@app.cell
def __(os, tokenizer, tokenizer_home, torch):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ckpt_home = '/blob/shufxi/scigpt/7bv3/inst/20240227121523/global_step3585/'

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
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        ckpt_dict,
        ckpt_home,
        k,
        l,
        l_index,
        layer,
        layer0,
        model,
        model_dict,
        show_ckpt,
    )


@app.cell
def __(model, tokenizer):
    def generate(s):
        if not s:
            return ""
        prompt = f"Instruction: {s}\n\n\nResponse:"
        print("prompt:", prompt)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        output = model.generate(
            input_ids,
            num_beams=4,
            max_new_tokens=100,
            num_return_sequences=4,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
        )

        output_ids = output.sequences[0]

        out_str = tokenizer.decode(output_ids).replace(' <a>', '').replace(' <m>', '')

        response_start = out_str.find('Response:') + len('Response:')
        response_end = out_str.find('</s>')

        return out_str[response_start:response_end]
    return generate,


@app.cell
def __(mo):
    instruction = mo.ui.text_area(label='instruction', full_width=True)
    instruction
    return instruction,


@app.cell
def __(generate, instruction, mo):
    response = generate(instruction.value)
    mo.md(f'```{response}```')
    return response,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
