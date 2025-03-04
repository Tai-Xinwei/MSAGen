#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import logging
import pathlib
import sys
import tempfile
from typing import Any, Mapping, Sequence, Tuple

import joblib
import lmdb
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.Data.PDBData import protein_letters_3to1_extended as aa3to1
from scipy.spatial import distance_matrix
from tqdm import tqdm

from DockQ4SinglePair import DockQ4SinglePair


logging.basicConfig(level=logging.INFO)


def convert_to_mmcif_string(
  inpfile: str,
  data: Mapping[str, Any],
) -> str:
  """Convert input PDB structure to MMCIF string."""
  mmcif_str = ''
  try:
    assert pathlib.Path(inpfile).is_file(), 'Input structure does not exist'
    assert pathlib.Path(inpfile).suffix == '.pdb', 'Input structure must be PDB'
    assert data and 'name' in data and 'nonpoly_graphs' in data, (
      f'Wrong popcessed data format')

    key = pathlib.Path(inpfile).stem.split('-')[0]
    name = data['name']
    assert name == key, f'Wrong name for {inpfile}, {name} != {key}'
    pdbid, ligid = name.split('_')

    ccds = [_['name'] for _ in data['nonpoly_graphs']]
    assert 1 == len(set(ccds)), f'Multiple CCDs {ccds} in nonpoly_graphs'
    assert ligid == data['nonpoly_graphs'][0]['name'], 'Wrong ligand name'
    atomids = data['nonpoly_graphs'][0]['atomids']
    symbols = data['nonpoly_graphs'][0]['symbols']

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(f'{pdbid}_{ligid}', inpfile)
    assert len(structure) == 1, 'Must 1 model in input structure'

    hetres = []
    for chain in structure[0]:
      for residue in chain:
        if residue.id[0] == ' ':
          continue
        hetres.append(residue)
    assert (
      len(hetres) == 1 and
      len(hetres[0]) == len(atomids) and
      all(
        a.element.upper()==_.upper()
        for a, _ in zip(hetres[0].get_atoms(), symbols)
      )
    ), f'Wrong ligand {ligid} parsed from input structure'

    ligres = hetres[0]
    ligres.resname = ligid
    ligres.id = (f'H_{ligid}', 1, ' ')
    for atom, atomid in zip(ligres.get_atoms(), atomids):
      atom.id = atomid
      atom.name = atomid

    with io.StringIO() as sio:
      mmcifio = PDB.MMCIFIO()
      mmcifio.set_structure(structure)
      mmcifio.save(sio)
      mmcif_str = sio.getvalue()
  except Exception as e:
    logging.error('Failed to convert %s to mmcif, %s', inpfile, e)

  return mmcif_str


def collect_models(
  mmcif_dir: str,
  metadata_csv: str,
  result_dir: str,
) -> Sequence[Mapping[str, Any]]:
  """Collect models from LMDB and result directory."""
  models = []

  if True:
    metadf = pd.read_csv(metadata_csv)
    assert {
      'pdb_id', 'chain_id_1', 'chain_id_2', 'cluster_key_chain_1',
      'cluster_key_chain_2', 'interface_cluster_key'
    }.issubset(metadf.columns), 'Invalid metadata csv'
    logging.info('%d interfaces loaded from %s', len(metadf), metadata_csv)

    for _, r in pd.read_csv(metadata_csv).iterrows():
      rd = r.to_dict()
      pdb_id = rd['pdb_id'].lower()
      name = rd['pdb_id'].upper()

      # if pdb_id not in ('7sjo'):
      #   continue

      refcifpath = pathlib.Path(mmcif_dir).resolve() / f'{pdb_id}.cif'
      assert refcifpath.is_file(), f'{refcifpath} does not exist'
      refcifstr = refcifpath.read_text()

      for p in pathlib.Path(result_dir).resolve().glob(f'{name}-*'):
        if p.stem.endswith('native'):
          continue
        if p.suffix == '.pdb':
          inpcifstr = convert_to_mmcif_string(str(p), rd)
        elif p.suffix == '.cif':
          inpcifstr = p.read_text()
        else:
          logging.error('Invalid input structure %s', p)
          inpcifstr = ''
        models.append({
          'name': name,
          'model_index': int(p.stem.removeprefix(f'{name}-')),
          'pdb_id': pdb_id,
          'chain_id_1': rd['chain_id_1'],
          'chain_id_2': rd['chain_id_2'],
          'cluster_key_chain_1': rd['cluster_key_chain_1'],
          'cluster_key_chain_2': rd['cluster_key_chain_2'],
          'interface_cluster_key': rd['interface_cluster_key'],
          'refcifstr': refcifstr,
          'inpcifstr': inpcifstr,
        })

  return models


def extract_chains_from_mmcif(
  mmcif_str: str,
  label_asym_ids: Sequence[str],
) -> str:
  """Extract chains from mmcif string."""
  parser = PDB.MMCIFParser(QUIET=True)
  handle = io.StringIO(mmcif_str)
  full_structure = parser.get_structure('', handle)
  parsed_info = parser._mmcif_dict

  assert '_atom_site.label_asym_id' in parsed_info, (
    f'_atom_site.label_asym_id not in reference mmcif string')
  asym_ids = set(parsed_info['_atom_site.label_asym_id'])
  assert set(label_asym_ids).issubset(asym_ids), (
    f'{label_asym_ids} not in reference mmcif string')
  mask = [_ in label_asym_ids for _ in parsed_info['_atom_site.label_asym_id']]

  mmcifdict = {'data_': parsed_info['data_']}
  for key in (
    '_atom_site.group_PDB',
    '_atom_site.id',
    '_atom_site.type_symbol',
    '_atom_site.label_atom_id',
    '_atom_site.label_alt_id',
    '_atom_site.label_comp_id',
    '_atom_site.label_asym_id',
    '_atom_site.label_entity_id',
    '_atom_site.label_seq_id',
    '_atom_site.pdbx_PDB_ins_code',
    '_atom_site.Cartn_x',
    '_atom_site.Cartn_y',
    '_atom_site.Cartn_z',
    '_atom_site.occupancy',
    '_atom_site.B_iso_or_equiv',
    '_atom_site.pdbx_formal_charge',
    '_atom_site.auth_seq_id',
    '_atom_site.auth_comp_id',
    '_atom_site.auth_asym_id',
    '_atom_site.auth_atom_id',
    '_atom_site.pdbx_PDB_model_num',
  ):
    assert key in parsed_info, f'{key} not in mmcif string'
    mmcifdict[key] = [_ for _, l in zip(parsed_info[key], mask) if l]

  with io.StringIO() as sio:
    mmcifio = PDB.mmcifio.MMCIFIO()
    mmcifio.set_dict(mmcifdict)
    mmcifio.save(sio)
    return sio.getvalue()


def extract_chain_coords_from_mmcif(
  mmcif_str: str,
  label_asym_id: str,
) -> Sequence[Tuple[float, float, float]]:
  """Extract chain coordinates from mmcif string."""
  parser = PDB.MMCIFParser(QUIET=True)
  handle = io.StringIO(mmcif_str)
  full_structure = parser.get_structure('', handle)
  parsed_info = parser._mmcif_dict

  assert set([
    '_atom_site.label_asym_id',
    '_atom_site.type_symbol',
    '_atom_site.Cartn_x',
    '_atom_site.Cartn_y',
    '_atom_site.Cartn_z',
  ]).issubset(parsed_info.keys()), f'Invalid keys in mmcif string'

  coords = []
  for l, t, x, y, z in zip(
    parsed_info['_atom_site.label_asym_id'],
    parsed_info['_atom_site.type_symbol'],
    parsed_info['_atom_site.Cartn_x'],
    parsed_info['_atom_site.Cartn_y'],
    parsed_info['_atom_site.Cartn_z'],
  ):
    if l == label_asym_id and t != 'H':
      coords.append((float(x), float(y), float(z)))
  return coords


def evaluate_one_model(
  model: Mapping[str, Any],
) -> Mapping[str, Any]:
  """Evaluate one model by DockQ."""
  score = {
    'Name': model['name'],
    'ChainID1': model['chain_id_1'],
    'ChainID2': model['chain_id_2'],
    'ModelIndex': model['model_index'],
    'ClusterKey': model['interface_cluster_key'],
    'pLDDT': -1.0,
    'DockQ': -1.0,
    'iRMSD': -1.0,
    'LRMSD': -1.0,
    'fnat': -1.0,
    'fnonnat': -1.0,
    'F1': -1.0,
    'clashes': -1,
  }

  try:
    # Extract interface chains from reference structures
    label_asym_ids = set([model['chain_id_1'], model['chain_id_2']])
    refcifstr = extract_chains_from_mmcif(model['refcifstr'], label_asym_ids)
    # check interface chain is correct or not
    refcoord_1 = extract_chain_coords_from_mmcif(refcifstr, model['chain_id_1'])
    refcoord_2 = extract_chain_coords_from_mmcif(refcifstr, model['chain_id_2'])
    assert (
      len(refcoord_1) > 0 and len(refcoord_2) > 0 and
      np.min(distance_matrix(refcoord_1, refcoord_2)) <= 5.0
    ), 'Invalid interface {},{}'.format(model['chain_id_1'], model['chain_id_2'])
    # mapping interfaces to chain pairs in predicted structure
    with (
      tempfile.NamedTemporaryFile() as refcif,
      tempfile.NamedTemporaryFile() as inpcif,
    ):
      refcif.write(refcifstr.encode())
      refcif.flush()
      inpcif.write(model['inpcifstr'].encode())
      inpcif.flush()
      tmpsco = DockQ4SinglePair(inpcif.name, refcif.name)
      score.update({
        'pLDDT': 0.,
        'DockQ': tmpsco['DockQ'],
        'iRMSD': tmpsco['iRMSD'],
        'LRMSD': tmpsco['LRMSD'],
        'fnat': tmpsco['fnat'],
        'F1': tmpsco['F1'],
        'fnonnat': tmpsco['fnonnat'],
        'clashes': tmpsco['clashes'],
      })
  except Exception as e:
    logging.error('Failed to evaluate %s, %s', model['name'], e)

  return score


if __name__ == '__main__':
  if len(sys.argv) != 4:
    sys.exit(f'Usage: {sys.argv[0]} <mmcif_dir> <metadata_csv> <result_dir>')
  mmcif_dir, metadata_csv, result_dir = sys.argv[1:4]

  assert pathlib.Path(mmcif_dir).exists(), f'{mmcif_dir} does not exist'
  assert pathlib.Path(metadata_csv).is_file(), f'{metadata_csv} does not exist'
  assert pathlib.Path(result_dir).exists(), f'{result_dir} does not exist'
  result_dir = str(pathlib.Path(result_dir).resolve())

  logging.info('Collecting models for evaluation...')
  models = collect_models(mmcif_dir, metadata_csv, result_dir)
  assert len(models) > 0, f'No models collected from {result_dir}'
  logging.info('%d models collected from %s', len(models), result_dir)

  # print(evaluate_one_model(models[0]))
  # exit('Debug')
  scores = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(evaluate_one_model)(_)
    for _ in tqdm(models, desc='Evaluating models by PoseBusters...')
  )
  df = pd.DataFrame(scores)
  logging.info('%d records in evaluation result', len(df))

  logging.info('Processing evaluation result and get ranking...')
  if df.isnull().values.any():
    raise ValueError('NaN in evaluation result')
  if 'Ranking' not in df.columns:
    dfs = []
    for key, gdf in df.groupby(['Name', 'ChainID1', 'ChainID2']):
      gdf.sort_values(by=['pLDDT'], ascending=False, inplace=True)
      gdf['Ranking'] = range(1, len(gdf) + 1)
      dfs.append(gdf)
    df = pd.concat(dfs).sort_index()
  if 'Orcale' not in df.columns:
    dfs = []
    for name, gdf in df.groupby(['Name', 'ChainID1', 'ChainID2']):
      gdf.sort_values(by=['DockQ'], ascending=False, inplace=True)
      gdf['Oracle'] = range(1, len(gdf) + 1)
      dfs.append(gdf)
    df = pd.concat(dfs).sort_index()
  print(df)
  df.to_csv(f'{result_dir}_Score4EachModel.csv', index=False)
  logging.info('Saving score4model to %s', f'{result_dir}_Score4EachModel.csv')

  logging.info('Calculating average score for each target...')
  records = []
  cols = ['ModelIndex', 'Ranking', 'Oracle', 'DockQ']
  for k, gdf in df.groupby(['Name', 'ChainID1', 'ChainID2', 'ClusterKey']):
    record = {'Name':k[0], 'ChainID1':k[1], 'ChainID2':k[2], 'ClusterKey':k[3]}
    rnd1 = gdf[gdf['ModelIndex'] == 1]
    for k in cols:
      record[f'Rnd1_{k}'] = rnd1[k].iloc[0]
    top1 = gdf[gdf['Ranking'] == 1]
    for k in cols:
      record[f'Top1_{k}'] = top1[k].iloc[0]
    best = gdf[gdf['Oracle'] == 1]
    for k in cols:
      record[f'Best_{k}'] = best[k].iloc[0]
    records.append(record)
  newdf = pd.DataFrame(records)
  print(newdf)
  newdf.to_csv(f'{result_dir}_Score4Target.csv', index=False)
  logging.info('Saving score4target to %s', f'{result_dir}_Score4Target.csv')

  logging.info('Calculating average score for each cluster...')
  results = []
  for key, gdf in newdf.groupby('ClusterKey'):
    results.append({
      'ClusterKey': key,
      'ClusterSize': len(gdf),
      'Rnd1_DockQ': gdf['Rnd1_DockQ'].mean(),
      'Top1_DockQ': gdf['Top1_DockQ'].mean(),
      'Best_DockQ': gdf['Best_DockQ'].mean(),
    })
  scodf = pd.DataFrame(results)
  print(scodf)
  scodf.to_csv(f'{result_dir}_Score4Cluster.csv', index=False)
  logging.info('Saving score4cluster to %s', f'{result_dir}_Score4Cluster.csv')

  logging.info('Calculating success rate...')
  meandict = [{
    'Metric': 'SuccessRate',
    'Number': len(scodf),
    'Rnd1(%)': 1. * sum(scodf['Rnd1_DockQ'] > 0.23) / len(scodf),
    'Top1(%)': 1. * sum(scodf['Top1_DockQ'] > 0.23) / len(scodf),
    'Best(%)': 1. * sum(scodf['Best_DockQ'] > 0.23) / len(scodf),
  }]
  meandf = pd.DataFrame(meandict).set_index('Metric')
  with pd.option_context('display.float_format', '{:.2%}'.format):
    print(meandf)
