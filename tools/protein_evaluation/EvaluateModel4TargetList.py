#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence
from typing import Tuple

import click
import pandas as pd
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from LGA4SinglePair import LGA4SinglePair
from metadata import metadata4target
from TMscore4SinglePair import TMscore4SinglePair


#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        '887': 'ESMFoldGitHub',
        '886': 'AF2NoMSA',
        '885': 'AF2WithMSA',
        '888': 'SFM',
        })
    return servers


def _collect_models(targets: Sequence[str],
                    servers: Mapping[str, str],
                    prediction_root: str,
                    top5: bool,
                    ) -> Sequence[Tuple[str, str, int, str, str]]:
    # check prediction directory
    rootdir = Path(prediction_root)
    cameodir = Path(rootdir) / 'cameo-official-targets.prediction'
    assert cameodir.exists(), f"{cameodir} does not exist."
    caspdir = Path(rootdir) / 'casp-official-targets.prediction'
    assert caspdir.exists(), f"{caspdir} does not exist."
    caspdomdir = Path(rootdir) / 'casp-official-trimmed-to-domains.prediction'
    assert caspdomdir.exists(), f"{caspdomdir} does not exist."
    # collect models
    models = []
    max_model_num = 5 if top5 else 1
    for t in tqdm(targets, desc="Collecting models"):
        # get server models and format (model, native) pairs
        for server in servers:
            for idx in range(1, max_model_num+1):
                if len(t) >= 8 and t[0] == 'T' and t[-3:-1] == '-D':
                    # e.g. T1024-D1
                    native = caspdomdir / f'{t}.pdb'
                    model = caspdomdir / t / f'{t[:-3]}TS{server}_{idx}{t[-3:]}'
                elif len(t) >= 5 and t[0] == 'T':
                    # e.g. T1024
                    native = caspdir / f'{t}.pdb'
                    model = caspdir / t / f'{t}TS{server}_{idx}'
                elif len(t) >= 6 and t[4] == '_':
                    # e.g. 1ctf_A
                    native = cameodir / t / 'target.pdb'
                    prefix = cameodir / t / 'servers' / f'server{server}'
                    model = prefix / f'model-{idx}' / f'model-{idx}.pdb'
                models.append( (t, server, idx, str(model), str(native)) )
    return models


@cli.command()
@click.option("--target-list",
              type=click.Path(exists=True),
              help="Input list for targets.",
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
             prediction_root: Path,
             result_directory: Path,
             num_workers: int,
             criterion: str,
             top5: bool) -> None:
    # get server group
    servers = _get_servers(target_list)
    logger.info(f"Evaluation for server groups: {servers}")

    # parse target list
    targets = []
    with open(target_list, 'r') as fp:
        for line in fp:
            assert 5 < len(line) < 12, f"Invalid target name {target}."
            targets.append(line.rstrip('\n'))
    logger.info(f"Number of targets: {len(targets)}")

    # convert metadata information to dictionary
    groupdict, lengthdict, typedict = {}, {}, {}
    for target in targets:
        key = target
        if len(target) >= 8 and target[-3:-1] == '-D':
            # e.g. T1024-D1
            key = target[:-3]
        logger.info(f"{metadata4target[key]}")
        for domstr, domlen, domgroup in metadata4target[key]['domain']:
            if domstr.split(':')[0] == target:
                groupdict[target] = domgroup
                lengthdict[target] = domlen
                typedict[target] = metadata4target[key]['type']
                break
        else:
            logger.error(f"{target} metadata information not found.")
    logger.info(f"Metadata information for targets: {len(groupdict)}")

    # collect models
    models = _collect_models(targets, servers, prediction_root, top5)
    logger.info(f"{len(models)} models in {prediction_root}")
    assert len(models) == len(targets) * len(servers) * (5 if top5 else 1)
    for target, server, idx, model, native in models:
        logger.debug(f"{target} {server} {idx} {model} {native}")

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
        score_dict['Group'] = groupdict.get(target, 'NA')
        score_dict['Length'] = lengthdict.get(target, -1)
        score_dict['Type'] = typedict.get(target, 'NA')
        for server, server_name in servers.items():
            subdf = gdf[ gdf['Server']==server ]
            k1 = f'{server_name}_{criterion}_Top1'
            score_dict[k1] = subdf[subdf['ModelIndex'] == 1][criterion].max()
            if top5:
                k5 = f'{server_name}_{criterion}_Top5'
                score_dict[k5] = subdf[criterion].max()
        newscores[target] = score_dict
    df = pd.DataFrame.from_dict(newscores, orient='index')
    print(df)

    # output results
    result_directory = Path(result_directory)
    outfile = result_directory / f'{target_list}_{criterion}.csv'
    df.to_csv(outfile)
    logger.info(f"Write evaluation results to {outfile}")

    # simple analysis for results
    CATEGORY = {
        "CAMEO  Easy": ["Easy"],
        "CAMEO  Medi": ["Medium", "Hard"],
        "CASP14 Easy": ["TBM-easy", "TBM-hard"],
        "CASP14 Hard": ["FM/TBM", "FM"],
        "CASP15 Easy": ["TBM-easy", "TBM-hard"],
        "CASP15 Hard": ["FM/TBM", "FM"],
    }
    print(f"{'Category':<12} {'Number':>6} {criterion}_for_different_servers")
    for category, groups in CATEGORY.items():
        _subdf = df[df["Type"] == category.split()[0]]
        dfsub = pd.concat(
            [_subdf[_subdf["Group"] == g] for g in groups],
            ignore_index=True,
            )
        dfsub.drop(columns=["Type", "Group", "Length"], axis=1, inplace=True)
        print(f"{category:<12s} {len(dfsub):>6d}", end=" "),
        print(" ".join([f"{_*100:>6.2f}" for _ in dfsub.mean()]))



if __name__ == "__main__":
    cli()
