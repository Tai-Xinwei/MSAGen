#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import pathlib
import pickle
import sys
import tempfile
import zlib
from typing import Any, Mapping, Sequence, Set, Tuple

import joblib
import lmdb
import pandas as pd
from tqdm import tqdm

from lddt4SinglePair import lddt4SinglePair
from TMscore4SinglePair import TMscore4SinglePair


logging.basicConfig(level=logging.INFO)


def bstr2obj(bstr: bytes):
  return pickle.loads(zlib.decompress(bstr))


def collect_models(
    lmdb_dir: str,
    result_dir: str,
) -> Sequence[Mapping[str, Any]]:
    """Collect models from result_dir and protein test lmdb."""
    models = []

    metadata = None
    with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
        metadata = bstr2obj(txn.get('__metadata__'.encode()))
    assert metadata is not None and 'keys' in metadata, (
        f'Failed to load metadata from {lmdb_dir}')

    print('-'*80)
    print(metadata['comment'], end='')
    for k, v in metadata.items():
        k != 'comment' and print(k, len(v))
    print(f"metadata['keys'][:10]={metadata['keys'][:10]}")
    print('-'*80)

    assert (
        {'pdbs', 'types', 'domains'}.issubset(metadata) and
        len(metadata['pdbs']) == len(metadata['keys']) and
        len(metadata['types']) == len(metadata['keys']) and
        len(metadata['domains']) == len(metadata['keys'])
    ), f'Wrong pdbs, types or domains in metadata for {lmdb_dir}'
    logging.info('%d keys, pdbs, types and domains.', len(metadata['keys']))

    for name, pdbstr, _type, _domain in tqdm(
        zip(
            metadata['keys'],
            metadata['pdbs'],
            metadata['types'],
            metadata['domains'],
        ),
        desc='Collecting models...',
    ):
        # if name not in ('T1104'):
        #     continue

        for p in pathlib.Path(result_dir).glob(f'{name}-*'):
            if p.stem.endswith('native'):
                continue
            models.append({
                'name': name,
                'model_index': int(p.stem.removeprefix(f'{name}-')),
                'refpdbstr': pdbstr,
                'inppdbstr': p.read_text(),
                'type': _type,
                'domain': _domain,
            })

    return models


def calculate_score(
    predlines: Sequence[str],
    natilines: Sequence[str],
    residx: Set[int],
) -> Mapping[str, Any]:
    """Calculate score between predicted and native structure by TM-score"""
    def _select_residues_by_residx(atomlines: list):
        lines = []
        for line in atomlines:
            if line.startswith('ATOM'):
                resnum = int(line[22:26].strip())
                if resnum in residx:
                    lines.append(line)
        lines.append('TER\n')
        lines.append('END\n')
        return lines

    with (
        tempfile.NamedTemporaryFile() as predpdb,
        tempfile.NamedTemporaryFile() as natipdb,
    ):
        with open(predpdb.name, 'w') as fp:
            fp.writelines(_select_residues_by_residx(predlines))
        with open(natipdb.name, 'w') as fp:
            fp.writelines(_select_residues_by_residx(natilines))
        score = TMscore4SinglePair(predpdb.name, natipdb.name)
        score['LDDT'] = lddt4SinglePair(predpdb.name, natipdb.name)['LDDT']
        return score


def evaluate_one_model(
    model: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    """Evalute one model by TM-score and LDDT."""
    scores = []

    try:
        # check model information
        assert {
            'name', 'model_index', 'refpdbstr', 'inppdbstr', 'type', 'domain',
        }.issubset(model.keys()), 'Failed to load model infomation.'
        # calculate score for each domain
        for domstr, domlen, domgroup in model['domain']:
            predlines = [f'{_}\n'for _ in model['inppdbstr'].split('\n')]
            assert predlines, f'Empty predicted file'
            natilines = [f'{_}\n'for _ in model['refpdbstr'].split('\n')]
            assert natilines, f'Empty native file'

            residx = set()
            domseg = domstr.split(':')[1]
            for seg in domseg.split(','):
                start, finish = [int(_) for _ in seg.split('-')]
                residx.update(range(start, finish + 1))
            assert domlen == len(residx), f'domain length!={domlen}'

            bf = [float(_[60:66].strip()) for _ in predlines if _[:4] == 'ATOM']
            plddt = sum(bf) / len(bf)
            score = calculate_score(predlines, natilines, residx)

            scores.append({
                'Name': domstr.split(':')[0],
                'Target': model['name'],
                'ModelIndex': model['model_index'],
                'Type': model['type'],
                'Length': domlen,
                'Group': domgroup,
                'pLDDT': plddt,
                'RMSD': score['RMSD'],
                'TMscore': score['TMscore'],
                'GDT_TS': score['GDT_TS'],
                'LDDT': score['LDDT'],
            })
    except Exception as e:
        logging.error('Failed to evaluate %s, %s.', model['name'], e)

    return scores


def calculate_average_score(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate average score for different categories."""
    CATEGORY = {
        'CAMEO   Full': ['Easy', 'Medium', 'Hard'],
        'CAMEO   Easy': ['Easy'],
        'CAMEO Medium': ['Medium'],
        'CAMEO   Hard': ['Hard'],
        'CASP14  Full': ['MultiDom'],
        'CASP14   TBM': ['TBM-easy', 'TBM-hard'],
        'CASP14    FM': ['FM/TBM', 'FM'],
        'CASP15  Full': ['MultiDom'],
        'CASP15   TBM': ['TBM-easy', 'TBM-hard'],
        'CASP15    FM': ['FM/TBM', 'FM'],
        '(   0,  384]': ['Easy', 'Medium', 'Hard', 'MultiDom'],
        '( 384,  512]': ['Easy', 'Medium', 'Hard', 'MultiDom'],
        '( 512, 8192]': ['Easy', 'Medium', 'Hard', 'MultiDom'],
    }
    # group score by target
    records = []
    for name, gdf in df.groupby('Name'):
        record = {
            'Name': name,
            'Target': gdf['Target'].iloc[0],
            'Length': gdf['Length'].iloc[0],
            'Group': gdf['Group'].iloc[0],
            'Type': gdf['Type'].iloc[0],
        }
        numtop1 = gdf['ModelIndex'][gdf['pLDDT'].idxmax()]
        for col in ['TMscore', 'LDDT']:
            top1score = gdf[gdf['ModelIndex'] == numtop1][col].iloc[0]
            maxscore = -10000.
            for num in gdf['ModelIndex'].to_list():
                score = gdf[gdf['ModelIndex'] == num][col].iloc[0]
                record[f'Model{num}_{col}'] = score
                maxscore = max(maxscore, score)
            record[f'ModelTop1_{col}'] = top1score
            record[f'ModelMax_{col}'] = maxscore
        records.append(record)
    newdf = pd.DataFrame(records)
    # calculate average score for each category
    scores = []
    for key, groups in CATEGORY.items():
        if key.startswith('('):
            low, high = [int(_) for _ in key.strip('(]').split(',')]
            subdf = newdf[
                (newdf['Length'] > low) &
                (newdf['Length'] <= high) &
                newdf['Group'].isin(groups)
            ]
        else:
            cate_type = key.split()[0]
            subdf = newdf[
                (newdf['Type'] == cate_type) &
                newdf['Group'].isin(groups)
            ]
        scores.append({
            'CatAndGroup': key,
            'Number': len(subdf),
            'Rnd1TMscore': subdf['Model1_TMscore'].mean() * 100,
            'Top1TMscore': subdf['ModelTop1_TMscore'].mean() * 100,
            'BestTMscore': subdf['ModelMax_TMscore'].mean() * 100,
            'Rnd1LDDT': subdf['Model1_LDDT'].mean() * 100,
            'Top1LDDT': subdf['ModelTop1_LDDT'].mean() * 100,
            'BestLDDT': subdf['ModelMax_LDDT'].mean() * 100,
        })
    # calculate average score for dataframe
    meandf = pd.DataFrame(scores).set_index('CatAndGroup')
    return newdf, meandf


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(f'Usage: {sys.argv[0]} <proteintest_lmdb> <result_dir>')
    ptlmdb, result_dir = sys.argv[1:3]

    assert pathlib.Path(ptlmdb).exists(), f'{ptlmdb} does not exists!'
    assert pathlib.Path(result_dir).exists(), f'{result_dir} does not exists!'
    result_dir = str(pathlib.Path(result_dir).resolve())

    logging.info('Collecting models for evaluation...')
    models = collect_models(ptlmdb, result_dir)
    assert len(models) > 0, f'No models collected from {result_dir}'
    logging.info('%d models collected from %s', len(models), result_dir)

    # print(evaluate_one_model(models[0]))
    # exit('Debug')
    scores = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(evaluate_one_model)(_)
        for _ in tqdm(models, desc='Evaluating models by TMscore and lddt...')
    )
    df = pd.DataFrame([_ for s in scores for _ in s])
    df['Name'] = df['Name'].apply(lambda x: x.rstrip('-D0'))
    df.sort_values(by=['Type', 'Group', 'Name', 'ModelIndex'], inplace=True)
    print(df)
    df.to_csv(f'{result_dir}_Score4EachModel.csv', index=False)
    logging.info('Saving score4model to %s', f'{result_dir}_Score4EachModel.csv')

    logging.info(f'Average metric for different categories.')
    newdf, meandf = calculate_average_score(df)
    print(newdf)
    newdf.to_csv(f'{result_dir}_Score4Target.csv', index=False)
    logging.info('Saving score4target to %s', f'{result_dir}_Score4Target.csv')
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(meandf)
