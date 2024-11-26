# -*- coding: utf-8 -*-
from posebusters import PoseBusters
from multiprocessing import Pool, cpu_count
import pandas as pd
import tqdm
import argparse

def do_parallel_bust(csv, output, cores):
    posebuster = PoseBusters()
    target_list = pd.read_csv(csv)
    args_list = []
    for idx, target in target_list.iterrows():
        args_list.append((target['mol_pred'], target['mol_true'], target['mol_cond'], True))

    with Pool(cores) as p:
        results = list(tqdm.tqdm(p.starmap(posebuster.bust, args_list), total=len(args_list)))

    results_df = pd.concat(results, join='outer')
    results_df.to_csv(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="posebusters")
    parser.add_argument("--csv", type=str, default='targets.csv', help="csv containing mol_pred, mol_true and mol_cond")
    parser.add_argument("--output", type=str, default='output.csv', help="output csv")
    parser.add_argument("--cores", type=int, default=cpu_count(), help="number of cores. Default use all cores")

    args = parser.parse_args()
    do_parallel_bust(args.csv, args.output, args.cores)
    print(f"Results saved to {args.output}")
