# -*- coding: utf-8 -*-
import argparse
from inference_module import NLMGenerator
import pickle as pkl
import os
from tqdm import tqdm
from glob import glob

class NLMInferencer:
    def __init__(self, ckpt_home, input_dir, output_dir):
        self.generator = NLMGenerator(ckpt_home)
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # self.file_paths = glob(os.path.join(input_dir, 'test*'))
        # self.file_paths.append(os.path.join(input_dir, 'sfmdata.prot.test.tsv'))
        self.file_paths = [
            os.path.join(input_dir, 'test.desc2mol.tsv'),
            os.path.join(input_dir, 'test.mol2desc.tsv'),
            os.path.join(input_dir, 'test.molinstruct.reaction.tsv'),
            os.path.join(input_dir, 'test.raw.i2s_i.txt'),
            os.path.join(input_dir, 'test.raw.s2i_s.txt'),
            os.path.join(input_dir, 'test.instruct.predict_bbbp.tsv'),
            os.path.join(input_dir, 'test.instruct.predict_bace.tsv'),
        ]

    def generate_responses(self):
        for fn in self.file_paths:
            print('Checking file: {}'.format(fn))
            basename = os.path.basename(fn)
            fn_out = basename + '.response.pkl'
            fn_out = os.path.join(self.output_dir, fn_out)

            if os.path.exists(fn_out):
                print('File {} already exists, skipping.'.format(fn_out))
                continue
            else:
                print('File {} does not exist, processing.'.format(fn_out))

            with open(fn, 'r', encoding='utf8') as fr:
                all_lines = [e.strip() for e in fr]

            buffer = []
            for idx, test_sample in tqdm(enumerate(all_lines), total=len(all_lines)):
                q = test_sample.split('\t')[0].strip()
                r0 = self.generator.chat(q, do_sample=False)
                r1 = self.generator.chat(q, do_sample=True)
                buffer.append((test_sample, r0, r1))

            with open(fn_out, 'wb') as fw:
                pkl.dump(buffer, fw)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate responses for inference')
    parser.add_argument('--ckpt_home', type=str, required=True, help='Checkpoint directory path')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
    return parser.parse_args()


def main():
    args = parse_args()
    inferencer = NLMInferencer(args.ckpt_home, args.input_dir, args.output_dir)
    inferencer.generate_responses()


if __name__ == '__main__':
    main()
