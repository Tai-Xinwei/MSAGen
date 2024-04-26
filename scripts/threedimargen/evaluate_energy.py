# -*- coding: utf-8 -*-
import json
import sys
from argparse import ArgumentParser
from math import sqrt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

arg_parser = ArgumentParser()
arg_parser.add_argument("input", type=str, help="input file")
args = arg_parser.parse_args()


def evaluate(fname):
    gt = []
    pred = []
    with open(fname, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            data = json.loads(line)
            gt.append(data["energy"] / data["num_atoms"])
            pred.append(data["prediction"]["energy"] / data["num_atoms"])
    # get MAE on gt and pred
    mae = mean_absolute_error(gt, pred)
    # get RMSE on gt and pred
    rmse = mean_squared_error(gt, pred, squared=False)
    print("MAE (eV/atom): ", format(mae, ".4f"))
    print("RMSE (eV/atom): ", format(rmse, ".4f"))


if __name__ == "__main__":
    evaluate(args.input)
