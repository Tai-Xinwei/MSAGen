#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence
from typing import Tuple

import click
import numpy as np
import pandas as pd
from absl import logging
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from LGA4SinglePair import LGA4SinglePair
from TMscore4SinglePair import TMscore4SinglePair


@click.group()
def cli():
    pass


def _get_servers(target_flag: str) -> Mapping[str, str]:
    servers = {}
    if 'casp15' in target_flag:
        servers = {
            '229' : 'Yang-Server',
            '185' : 'BAKER',
            '270' : 'NBIS-AF2-std',
        }
    elif 'casp14' in target_flag:
        servers = {
            '427' : 'AlphaFold2',
            '473' : 'BAKER',
            '324' : 'Zhang-Server',
            }
    elif 'cameo' in target_flag:
        servers = {
            '999': 'BestSingleT',
            '19': 'RoseTTAFold',
            '20': 'SWISS-MODEL'
        }
    else:
        raise ValueError(f"Invalid target flag {target_flag}.")
    servers.update({
        '888': 'SFM',
        '887': 'ESMFoldGitHub',
        '886': 'AF2NoMSA',
        '885': 'AF2WithMSA',
        #'880': 'RoseTTAFoldGitHub',
        })
    return servers


def _get_groups(meta_information: str) -> Mapping[str, str]:
    groupdict = {}
    assert meta_information.endswith('.metainformation'), (
        f"ERROR: wrong meta information file {meta_information}.")
    if 'casp' in meta_information:
        for _, row in pd.read_csv(meta_information).iterrows():
            groupdict[ row['Domains'].split(':')[0] ] = row['Classification']
    elif 'cameo' in meta_information:
        tmpd = {0: 'Easy', 1: 'Medium', 2: 'Hard'}
        for _, row in pd.read_csv(meta_information).iterrows():
            pdbchain = row['ref. PDB [Chain]']
            groupdict[f'{pdbchain[:4]}_{pdbchain[6]}'] = tmpd[row['Difficulty']]
    else:
        raise ValueError(f"Invalid group information {meta_information}.")
    return groupdict


def _get_lengths(meta_information: str) -> Mapping[str, int]:
    lengthdict = {}
    assert meta_information.endswith('.metainformation'), (
        f"ERROR: wrong meta information file {meta_information}.")
    if 'casp' in meta_information:
        for _, row in pd.read_csv(meta_information).iterrows():
            lengthdict[row['Domains'].split(':')[0]] = row['Residues in domain']
    elif 'cameo' in meta_information:
        for _, row in pd.read_csv(meta_information).iterrows():
            pdbchain = row['ref. PDB [Chain]']
            lengthdict[f'{pdbchain[:4]}_{pdbchain[6]}'] = row['Sequence Length (residues)']
    else:
        raise ValueError(f"Invalid length information {meta_information}.")
    return lengthdict


def _collect_models(targets: Sequence[str],
                    servers: Mapping[str, str],
                    prediction_root: Path,
                    top5: bool,
                    ) -> Sequence[Tuple[str, str, int, str, str]]:
    # check target flag
    if 'casp' in str(prediction_root):
        is_casp = True
    elif 'cameo' in str(prediction_root):
        is_casp = False
    else:
        raise ValueError(f"Invalid prediction root {prediction_root}.")
    # collect models
    models = []
    MAXMODELID = 5 if top5 else 1
    for target in tqdm(targets, desc="Collecting models"):
        # parse target target and domain
        tarname, domain = target, ''
        if target[-3:-1] == '-D':
            tarname, domain = target[:-3], target[-3:]
        # get native model
        if is_casp:
            native = prediction_root / f'{target}.pdb'
        else:
            native = prediction_root / target / 'target.pdb'
        if not native.exists():
            logging.error(f"Native structure {native} does not exist.")
            continue
        # get server models and format (model, native) pairs
        for server in servers:
            for idx in range(1, MAXMODELID+1):
                if is_casp:
                    model = prediction_root / target / f'{tarname}TS{server}_{idx}{domain}'
                else:
                    model = prediction_root / target / 'servers' / f'server{server}' / f'model-{idx}' / f'model-{idx}.pdb'
                models.append( (target, server, idx, str(model), str(native)) )
    return models


@cli.command()
@click.option("--target-list",
              type=click.Path(exists=True),
              help="Input list for targets.",
              required=True)
@click.option("--meta-information",
              type=click.Path(exists=True),
              help="Meta information file for the targets.",
              required=True)
@click.option("--prediction-root",
              type=click.Path(exists=True),
              help="Input directory for prediction results.",
              required=True)
@click.option("--result-directory",
              type=click.Path(exists=True),
              help="Output directory for evaluation results.",
              required=True)
@click.option("--num-workers",
              type=int,
              default=-1,
              help="Number of workers.",
              show_default=True)
@click.option("--criterion",
              type=str,
              default="TMscore",
              help="Evaluation metric 'TMscore', 'GDT_TS' or 'GDT_HA'. If it is set to 'TMscore', the program TMscore will be used to evaluate predicted structure; if it is set to 'GDT_TS' or 'GDT_HA', the program LGA will be used.",
              show_default=True)
@click.option("--top5",
              is_flag=True,
              default=False,
              help="Whether to output top5 score.",
              show_default=True)
def evaluate(target_list: Path,
             meta_information: Path,
             prediction_root: Path,
             result_directory: Path,
             num_workers: int,
             criterion: str,
             top5: bool) -> None:
    # get server group
    servers = _get_servers(target_list)
    print(f"Evaluation for server groups: {servers}")

    # parse target list
    with open(target_list, 'r') as fp:
        targets = [_.strip() for _ in fp]
    assert all([5 <= len(_) <= 10 for _ in targets]), (
        f"Invalid target name exists in {target_list}.")
    print(f"Number of targets: {len(targets)}")

    # parse meta information
    groupdict = _get_groups(meta_information)
    lengthdict = _get_lengths(meta_information)
    print(f"Meta information for targets: {len(groupdict)}")

    # collect models
    print(f"Predictions root directory: {prediction_root}")
    prediction_root = Path(prediction_root)
    models = _collect_models(targets, servers, prediction_root, top5)
    print(f"Number of models: {len(models)}")
    #for target, server, idx, model, native in models:
    #    print(target, server, idx, model, native)

    # collect scores
    def _score4pair(model_info: Tuple[str, str, int, str, str]):
        target, server, idx, model, native = model_info
        if criterion == 'TMscore':
            s = TMscore4SinglePair(model, native)
        else:
            s = LGA4SinglePair(model, native)
        s['Target'], s['Server'], s['ModelIndex'] = target, server, idx
        return s
    scores = Parallel(n_jobs=num_workers)(
        delayed(_score4pair)(_) for _ in tqdm(models)
        )
    df = pd.DataFrame(scores)
    print(df)

    # analysis scores
    newscores = {}
    for target, gdf in df.groupby('Target', sort=True):
        score_dict = OrderedDict()
        score_dict['Groups'] = groupdict.get(target, 'NA')
        score_dict['Length'] = lengthdict.get(target, -1)
        for server, server_name in servers.items():
            subdf = gdf[ gdf['Server']==server ]
            k1 = f'{server_name}_{criterion}_Top1'
            score_dict[k1] = subdf[subdf['ModelIndex'] == 1][criterion].max()
            if top5:
                k5 = f'{server_name}_{criterion}_Top5'
                score_dict[k5] = subdf[criterion].max()
        newscores[target] = score_dict
    newdf = pd.DataFrame.from_dict(newscores, orient='index')
    print(newdf)

    # output results
    print(f"Evaluation results directory: {result_directory}")
    result_directory = Path(result_directory)
    outfile = result_directory / f'{target_list}_{criterion}.csv'
    print(f"Output results to {outfile}")
    newdf.to_csv(outfile)


if __name__ == "__main__":
    cli()
