# -*- coding: utf-8 -*-
import argparse
from glob import glob

import pickle as pkl

import itertools

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_glob', action='store', help='', default='')
    parser.add_argument('--output_file', action='store', help='', default='')
    return parser.parse_args()

def main(input_glob, output_file):

    FF = glob(input_glob)

    merged_data = []

    for fn in FF:
        with open(fn, 'rb') as f:
            data = pkl.load(f)
            merged_data.extend([rec for rec in data if 'This is a padding sentence.' not in rec[0]])
        print(f'- {fn}: {len(data)}')
    with open(output_file, 'wb') as fw:
        pkl.dump(merged_data, fw)

    print(len(merged_data))

if __name__ == "__main__":

    args = parse_args()

    main(args.input_glob, args.output_file)

    ### Example:
    # python tools/nlm/x_merge.py --input_glob "/home/lihe/sfmdataeastus2_nlm/lihe/output/inference/sfmdata.prot.test.sampled30.tsv/llama2_7b/part*.pkl" --output_file /scratch/workspace/nlm/protein/output/sfmdata.prot.test.sampled30.tsv.llama2_7b.response.pkl
    # python tools/nlm/x_merge.py --input_glob "/home/lihe/sfmdataeastus2_nlm/lihe/output/inference/sfmdata.prot.test.sampled30.tsv/llama3_8b/part*.pkl" --output_file /scratch/workspace/nlm/protein/output/sfmdata.prot.test.sampled30.tsv.llama3_8b.response.pkl
