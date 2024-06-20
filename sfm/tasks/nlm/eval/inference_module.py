# -*- coding: utf-8 -*-
import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer


class SFMGenerator:
    def __init__(
        self, ckpt_home, tokenizer_home, prot_spm_path, dna_spm_path, rna_spm_path
    ):
        """
        SFMGenerator class is used to generate responses for the given input string.
        """
        tokenizer = SFMDecTokenizer.from_pretrained(
            tokenizer_home,
            prot_spm_path=prot_spm_path,
            dna_spm_path=dna_spm_path,
            rna_spm_path=rna_spm_path,
        )

        def show_ckpt(name, ckpt):
            for k, v in ckpt.items():
                if "dummy" not in k:
                    print(name, k, v.shape)

        model = AutoModelForCausalLM.from_pretrained(tokenizer_home)

        model_dict = model.state_dict()
        ckpt_dict = {}
        layer0 = torch.load(
            os.path.join(ckpt_home, "layer_00-model_states.pt"),
            map_location=torch.device("cpu"),
        )
        ckpt_dict["model.embed_tokens.weight"] = layer0["embed_tokens.weight"]
        show_ckpt("layer0", layer0)

        for l in range(0, 32):
            l_index = str(l + 1).zfill(2)
            layer = torch.load(
                os.path.join(ckpt_home, f"layer_{l_index}-model_states.pt"),
                map_location=torch.device("cpu"),
            )
            show_ckpt(l_index, layer)
            for k in layer:
                if "dummy" in k or "rotary_emb" in k:
                    continue
                ckpt_dict[f"model.layers.{l}.{k}"] = layer[k]
        layer = torch.load(
            os.path.join(ckpt_home, "layer_33-model_states.pt"),
            map_location=torch.device("cpu"),
        )
        show_ckpt(33, layer)
        ckpt_dict["model.norm.weight"] = layer["norm.weight"]

        layer = torch.load(
            os.path.join(ckpt_home, "layer_34-model_states.pt"),
            map_location=torch.device("cpu"),
        )
        show_ckpt(33, layer)
        ckpt_dict["lm_head.weight"] = layer["lm_head.weight"]
        model_dict.update(ckpt_dict)

        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(model_dict)

        self.model = model.cuda()
        self.tokenizer = tokenizer

    def chat(self, input_str, response_only=True, do_sample=False, **kwargs):
        prompt = f"Instruction: {input_str.strip()}\n\n\nResponse:"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        if "max_new_tokens" in kwargs:
            max_new_tokens = kwargs.pop("max_new_tokens")
        else:
            max_new_tokens = 100
        if do_sample:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                num_return_sequences=4,
                do_sample=True,
                temperature=0.75,
                top_p=0.95,
                **kwargs,
            )
        else:
            outputs = self.model.generate(
                input_ids,
                num_beams=4,
                max_new_tokens=max_new_tokens,
                num_return_sequences=4,
                do_sample=do_sample,
                **kwargs,
            )

        out_list = []
        for out in outputs:
            s = self.tokenizer.decode(out)
            if response_only:
                segs = s.split("Response:")
                s = segs[1].strip()
            segs = s.split("</s>")
            out_list.append(segs[0].strip())
        return out_list

    def extract_first_token_prob(self, input_str):
        prompt = f"Instruction: {input_str.strip()}\n\n\nResponse:"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        model_inputs = self.model.prepare_inputs_for_generation(input_ids)
        outputs = self.model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        yes_id = self.tokenizer("Yes")["input_ids"][-1]
        no_id = self.tokenizer("No")["input_ids"][-1]
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = F.softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)
        yes_prob = next_token_scores[0, yes_id].item()
        no_prob = next_token_scores[0, no_id].item()
        confidence = yes_prob / (yes_prob + no_prob)
        return confidence
