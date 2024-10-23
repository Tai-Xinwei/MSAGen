#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import click
import joblib
import lmdb
import numpy as np
from absl import logging
from tqdm import tqdm

import parse_mmcif
from commons import obj2bstr
from process_mmcif import (
  chunks,
  parse_mmcif_string,
  show_lmdb,
  STDRES,
)
from residue_constants import ATOMORDER, RESIDUEATOMS


logging.set_verbosity(logging.INFO)


def process_polymer_chain(polymer_chain: Sequence[dict]) -> Mapping[str, Any]:
  """Process one polymer chain."""
  resname, seqres, restype, center_coord, allatom_coord = [], [], [], [], []
  confidence = []
  for residue in polymer_chain:
    resname.append(residue['name'])
    seqres.append(residue['seqres'])
    restype.append(residue['restype'])
    r_plddt = np.nan
    c_coord = np.full(3, np.nan, dtype=np.float32)
    a_coord = np.full((len(ATOMORDER), 3), np.nan, dtype=np.float32)
    if not residue['is_missing']:
      _key = residue['name'] if residue['name'] in STDRES else 'UNK'
      for atom in residue['atoms']:
        pos = np.array([atom['x'], atom['y'], atom['z']])
        if atom['name'] == "CA" or atom['name'] == "C1'":
          # AlphaFold3 supplementary information section 2.5.4
          # For each token we also designate a token centre atom:
          # CA for standard amino acids
          # C1' for standard nucleotides
          c_coord = pos
        # AlphaFold3 supplementary information section 2.5.4
        # Filtering of bioassemblies: Hydrogens are removed.
        # Atoms should not exist in this residue are excluded.
        if atom['name'] in RESIDUEATOMS[_key]:
          a_coord[ATOMORDER[atom['name']], :] = pos
      r_plddt = np.mean([_['tempfactor'] for _ in residue['atoms']])
    center_coord.append(c_coord)
    allatom_coord.append(a_coord)
    confidence.append(r_plddt)
  data = {
    'resname': np.array(resname),
    'seqres': np.array(seqres),
    'restype': np.array(restype),
    'center_coord': np.array(center_coord, dtype=np.float32),
    'allatom_coord': np.array(allatom_coord, dtype=np.float32),
    'confidence': np.array(confidence, dtype=np.float32),
  }
  return data


def process_one_mgnify(mmcif_path: str) -> Mapping[str, Union[str, dict, list]]:
  """Parse a mmCIF file and convert data to list of dict."""
  try:
    logging.debug("File %s processed by JIANWZHU START.", mmcif_path)

    mmcif_path = str(Path(mmcif_path).resolve())
    pdbid = Path(mmcif_path).name.split('.')[0].split('_')[0]
    assert len(pdbid) == 16, f"Invalid 16 characters MGnify ID {pdbid}."
    mmcif_string = parse_mmcif_string(mmcif_path)
    assert mmcif_string, f"Failed to read mmcif string for {pdbid}."

    # Parse mmcif file by modified AlphaFold mmcif_parsing.py
    result = parse_mmcif.parse(file_id=pdbid, mmcif_string=mmcif_string)
    assert result.mmcif_object, f"The errors are {result.errors}"
    # print(result.mmcif_object.file_id, result.mmcif_object.header)

    full_chains = result.mmcif_object.chains

    num_poly = sum([1 if c.entity_type == 'polymer' else 0
                    for _, c in full_chains.items()])
    assert num_poly == 1, f"{num_poly} polymer chains, {pdbid}."

    polymer_chains = {}
    nonpoly_graphs = []
    for chain_id, chain in full_chains.items():
      # print('-'*80, f'{chain_id} {chain.type} {len(chain.seqres)}', sep='\n')
      seqres, residues = chain.seqres, chain.residues
      if chain.entity_type == 'polymer':
        aatype = 'p'
        current_chain = []
        for aa, r in zip(seqres, residues):
          # Process polymer residues for protein, DNA and RNA.
          resdict = dataclasses.asdict(r)
          resdict.update({'seqres': aa, 'restype': aatype})
          current_chain.append(resdict)
        polymer = process_polymer_chain(current_chain)
        # Check if polymer has center atom or all nan
        if not polymer or np.all(np.isnan(polymer['center_coord'])):
          logging.warning("Chain %s has no center atom, %s.", chain_id, pdbid)
        else:
          polymer_chains[chain_id] = polymer
      else:
        logging.warning("Unknown entity type %s for chain %s, %s. ",
                        chain.entity_type, chain_id, pdbid)
    assert polymer_chains, f"Has no desirable chains for {pdbid}."

    data = {
      'pdbid': pdbid,
      'structure_method': 'unknown',
      'release_date': '0001-01-01',
      'resolution': 0.0,
      'polymer_chains': polymer_chains,
      'nonpoly_graphs': nonpoly_graphs,
    }
    data.update(result.mmcif_object.header)

    logging.debug("File %s processed by JIANWZHU SUCCESS.", mmcif_path)
    return data
  except Exception as e:
    logging.error("File %s processed by JIANWZHU FAILED, %s.", mmcif_path, e)
    return {}


def show_one_structure(data: Mapping[str, Union[str, dict, list]]) -> None:
  """Show one processed data."""
  if not data:
    raise SystemExit("Processed data is empty and maybe wrong prcessing.")
  print(data.keys())
  print(data['pdbid'])
  print("structure_method:", data['structure_method'])
  print("release_date:", data['release_date'])
  print("resolution:", data['resolution'])
  print('-'*80)
  print("polymer_chains", len(data['polymer_chains']))
  for chain_id, polymer in data['polymer_chains'].items():
    print('-'*80)
    print(polymer.keys())
    print(f"{data['pdbid']}_{chain_id}", end=' ')
    restype = ''.join(polymer['restype'])
    _type = 'protein' if restype.count('p') >= restype.count('n') else 'na'
    print(f"polymer_type={_type} num_residues={len(restype)}")
    is_missing = [np.any(np.isnan(_)) for _ in polymer['center_coord']]
    print(''.join(polymer['seqres']))
    print("".join('-' if _ else c for _, c in zip(is_missing, restype)))
    arr = [f'{_:s}' for _ in polymer['resname'][:10]]
    print(f"resname[:10]          : [{', '.join(arr)}]")
    arr = [f'{_:.3f}' for _ in polymer['confidence'][:10]]
    print(f"confidence[:10]       : [{', '.join(arr)}]")
    for i, axis in enumerate('xyz'):
      arr = [f'{_:.3f}' for _ in polymer['center_coord'][:10, i]]
      print(f"center_coord[:10].{axis}   : [{', '.join(arr)}]")
    for i, axis in enumerate('xyz'):
      arr = [f'{_:.3f}' for _ in polymer['allatom_coord'][:10, 0, i]]
      print(f"allatom_coord[:10,0].{axis}: [{', '.join(arr)}]")


@click.command()
@click.option("--mmcif-dir",
              type=click.Path(exists=True),
              required=True,
              help="Input directory of mmCIF files rsync from RCSB.")
@click.option("--output-lmdb",
              type=click.Path(exists=False),
              default="output.lmdb",
              help="Output lmdb file.")
@click.option("--num-workers",
              type=int,
              default=-1,
              help="Number of workers.")
@click.option("--data-comment",
              type=str,
              default="MGnify data version 2019_05 predicted by mmseqs+colabfold_envdb+openfold_af2model3",
              help="Comments for output.")
def main(mmcif_dir: str,
         output_lmdb: str,
         num_workers: int,
         data_comment: str,
         ) -> None:
  """Process mmCIF files from directory and save to lmdb."""
  mmcif_dir = str(Path(mmcif_dir).resolve())
  mmcif_paths = {_.name.split('.')[0].split('_')[0]:str(_)
                 for _ in Path(mmcif_dir).rglob("*_unrelaxed.cif")}
  assert mmcif_paths and all(16==len(_) for _ in mmcif_paths), (
    f"MGnify ID should be 16 characters long in {mmcif_dir}.")
  logging.info("%d structures in %s.", len(mmcif_paths), mmcif_dir)

  pdbids = list(mmcif_paths.keys())
  logging.info("%d pdbids in structures and assemblies.", len(pdbids))

  output_lmdb = str(Path(output_lmdb).resolve())
  assert not Path(output_lmdb).exists(), f"ERROR: {output_lmdb} exists. Stop."
  logging.info(f"Will save processed data to %s.", output_lmdb)

  # data = process_one_mgnify(mmcif_paths[pdbids[0]])
  # data and show_one_structure(data)

  env = lmdb.open(output_lmdb, map_size=1024**4) # 1TB max size

  metadata = {
    'keys': [],
    'num_polymers': [],
    'num_nonpolys': [],
    'structure_methods': [],
    'release_dates': [],
    'resolutions': [],
    'comment': (
      f'Created time: {datetime.now()}\n'
      f'Input structures: {mmcif_dir}\n'
      f'Output lmdb: {output_lmdb}\n'
      f'Number of workers: {num_workers}\n'
      f'Comments: {data_comment}\n'
    ),
  }

  for pdbid_chunk in tqdm(list(chunks(pdbids, 10000)),
                          desc='Processing structures for 10k/chunk'):
    result_chunk = joblib.Parallel(n_jobs=num_workers)(
      joblib.delayed(process_one_mgnify)(mmcif_paths[_])
      for _ in tqdm(pdbid_chunk, desc='Processing cif')
    )
    with env.begin(write=True) as txn:
      for data in tqdm(result_chunk, desc='Saving to lmdb'):
        if not data:
          # skip empty data
          continue
        txn.put(data['pdbid'].encode(), obj2bstr(data))
        metadata['keys'].append(data['pdbid'])
        metadata['num_polymers'].append(len(data['polymer_chains']))
        metadata['num_nonpolys'].append(len(data['nonpoly_graphs']))
        metadata['structure_methods'].append(data['structure_method'])
        metadata['release_dates'].append(data['release_date'])
        metadata['resolutions'].append(data['resolution'])

  with env.begin(write=True) as txn:
    txn.put('__metadata__'.encode(), obj2bstr(metadata))

  env.close()

  show_lmdb(output_lmdb)


if __name__ == "__main__":
  main()
