# -*- coding: utf-8 -*-
import argparse
from inference_module import NLMGenerator
import pickle as pkl
import os
from tqdm import tqdm
from glob import glob

from sfm.models.tox.modules.torus import p

class NLMInferencer:
    def __init__(self, ckpt_home, input_file, output_file=None):
        self.generator = NLMGenerator(ckpt_home, "/sfmdataeastus2/nlm/llama/Meta-Llama-3-8B")
        self.input_file = input_file
        self.output_file = output_file

    def generate_responses(self):
        print('Checking file: {}'.format(self.input_file))
        if self.output_file is None:
            fn_out = self.input_file + '.response.pkl'
        else:
            fn_out = self.output_file

        with open(self.input_file, 'r', encoding='utf8') as fr:
            all_lines = [e.strip() for e in fr]

        buffer = []
        for idx, test_sample in tqdm(enumerate(all_lines), total=len(all_lines)):
            q = test_sample #.split('\t')[0].strip()
            r = self.generator.chat(q, do_sample=True)
            buffer.append([test_sample, r])
            if idx % 100 == 0:
                print(f"{q}\n{r}")

        with open(fn_out, 'wb') as fw:
            pkl.dump(buffer, fw)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate responses for inference')
    parser.add_argument('--ckpt_home', type=str, default="/sfmdataeastus2/nlm/kaiyuan/results/nlm/inst/inst_0621_bsz256_lr2e5_0624/global_step89920/")
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    inferencer = NLMInferencer(args.ckpt_home, args.input_file, args.output_file)
    inferencer.generate_responses()


if __name__ == '__main__':
    main()
