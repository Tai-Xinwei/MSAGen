# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(".")

import pandas as pd
import glob
import argparse
from tqdm import tqdm


def main(args):

    # get result files in a directory
    fold_folder = args.fold_folder
    res_files = glob.glob(fold_folder + '/*.pdb')

    # # iterate over files, and concat top-1 alignments
    # res = []
    # for path in res_files:
    #     # read file
    #     try:
    #         df = pd.read_csv(path, sep='\t', header=None)

    #         # get top 1 alignment
    #         df = df.iloc[0]

    #         # select specific columns
    #         df = df[[0, 1, 2, 12]]
    #         res.append(df)
    #     except:
    #         print(f'Error reading file {path}')


    # # concat alignments
    # res = pd.concat(res, axis=1).T

    # # set column names
    # res.columns = ['query', 'target', 'identity', 'tmscore']

    # read each pdb file, and calculate plddt
    plddts = []

    def read_pdb(pdb_file):
        """Read a PDB file and extract B-factors."""
        with open(pdb_file, 'r') as file:
            b_factors = []
            for line in file:
                if line.startswith("ATOM"):
                    b_factor = float(line[60:66].strip())
                    b_factors.append(b_factor)
            return b_factors

    def calculate_average_pLDDT(b_factors):
        """Calculate the average pLDDT score."""
        if b_factors:
            return sum(b_factors) / len(b_factors)
        else:
            return None


    for pdb_file in tqdm([file for file in res_files]):

        b_factors = read_pdb(pdb_file)
        average_pLDDT = calculate_average_pLDDT(b_factors)

        if average_pLDDT is None:
            print("No B-factors found in the PDB file.")

        plddts.append(average_pLDDT)

    # add plddt to dataframe
    res = pd.DataFrame({'protein': res_files, 'plddt': plddts})

    # statistics of plddt
    print('plddt')
    print(res['plddt'].describe())

    # # statistics of tmscore
    # print('tmscore')
    # # find missing values
    # print(res['tmscore'].isna().sum())
    # # fill missing values
    # res['tmscore'] = res['tmscore'].fillna(0)
    # print(res['tmscore'].describe())

    # # statistics of identity
    # print('identity')
    # # find missing values
    # print(res['identity'].isna().sum())
    # # fill missing values
    # res['identity'] = res['identity'].fillna(0)
    # print(res['identity'].describe())

    # save results
    output_excel = args.out_excel
    res.to_excel(os.path.join(fold_folder, output_excel), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate plddt of models')
    parser.add_argument('--fold_folder', type=str, help='Path to file containing sequences')
    parser.add_argument('--out_excel', type=str, help='Path to output directory')
    args = parser.parse_args()

    main(args)
