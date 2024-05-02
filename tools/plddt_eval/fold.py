# -*- coding: utf-8 -*-
from tqdm import tqdm
from transformers import EsmForProteinFolding
import argparse
import os

### Note the pytorch version of esmfold should be less than 2.0, e.g., 1.13.1 is okay.

def main(args):
    model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1')
    model.to('cuda')
    model.eval()


    # read sequences
    with open(args.seq_file, 'r') as f:
        sequences = f.readlines()
        sequences = [seq.strip() for seq in sequences]
        sequences = [seq for seq in sequences if len(seq) > 0]
        ### Note: it's possible that the sequence length would be long so that OOM will happen, you can ignore the long sequences.

    for idx, seq in tqdm(enumerate(sequences)):
        pdb_str = model.infer_pdb(seq)

        # write pdbs to file, create folder if it doesn't exist
        output_folder = args.out_dir
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open('{}/seq{idx}.pdb'.format(output_folder), 'w') as f:
            f.write(pdb_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fold sequences using ESM-1b')
    parser.add_argument('--seq_file', type=str, help='Path to file containing sequences')
    parser.add_argument('--out_dir', type=str, help='Path to output directory')
    args = parser.parse_args()

    main(args)
