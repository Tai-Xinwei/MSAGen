#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from datetime import datetime
from pathlib import Path

import click
import joblib
import lmdb
from absl import logging
from Bio.PDB import MMCIF2Dict
from tqdm import tqdm

import parse_mmcif
from commons import obj2bstr
from process_mmcif import chunks
from process_mmcif import parse_mmcif_string
from process_mmcif import process_one_structure
from process_mmcif import show_lmdb
from process_mmcif import show_one_structure


logging.set_verbosity(logging.INFO)


@click.command()
@click.option("--chem-comp-dir",
              type=click.Path(exists=True),
              required=True,
              help="Input directory of mmCIF files of all chemical components.")
@click.option("--mmcif-dir",
              type=click.Path(exists=True),
              required=True,
              help="Input directory of mmCIF files rsync from RCSB.")
@click.option("--assembly-dir",
              type=click.Path(exists=True),
              required=True,
              help="Input directory of assembly mmCIF files rsync from RCSB.")
@click.option("--output-lmdb",
              type=click.Path(exists=False),
              default="output.lmdb",
              help="Output lmdb file.")
@click.option("--num-workers",
              type=int,
              default=-1,
              help="Number of workers.")
@click.option("--release-date-cutoff",
              type=click.DateTime(formats=["%Y-%m-%d"]),
              default="2020-04-30",
              help="Release date cutoff for mmcif file.")
@click.option("--resolution-cutoff",
              type=float,
              default=9.,
              help="Resolution cutoff for mmcif file.")
@click.option("--data-comment",
              type=str,
              default="PDB snapshot from rsync://rsync.wwpdb.org::ftp/data/ with --port=33444 on 20240630.",
              help="Comments for output.")
def main(chem_comp_dir: str,
         mmcif_dir: str,
         assembly_dir: str,
         output_lmdb: str,
         num_workers: int,
         release_date_cutoff: datetime,
         resolution_cutoff: float,
         data_comment: str,
         ) -> None:
  """Process mmCIF files from directory and save to lmdb."""
  mmcif_dir = str(Path(mmcif_dir).resolve())
  mmcif_paths = {_.name.split('.')[0]:str(_)
                 for _ in Path(mmcif_dir).rglob("*.cif.gz")}
  assert mmcif_paths and all(4==len(_) for _ in mmcif_paths), (
    f"PDBID should be 4 characters long in {mmcif_dir}.")
  logging.info("%d structures in %s.", len(mmcif_paths), mmcif_dir)

  assembly_dir = str(Path(assembly_dir).resolve())
  assembly_paths = {_.name.split('.')[0].split('-assembly')[0]:str(_)
                    for _ in Path(assembly_dir).rglob("*-assembly1.cif.gz")}
  assert assembly_paths and all(4==len(_) for _ in assembly_paths), (
    f"PDBID should be 4 characters long in {assembly_dir}.")
  logging.info("%d assemblies in %s.", len(assembly_paths), assembly_dir)

  pdbids = list(mmcif_paths.keys() & assembly_paths.keys())
  logging.info("%d common pdbids for structures and assemblies.", len(pdbids))

  chem_comp_dir = str(Path(chem_comp_dir).resolve())
  logging.info("Chemical components information is in %s.", chem_comp_dir)

  output_lmdb = str(Path(output_lmdb).resolve())
  assert not Path(output_lmdb).exists(), f"ERROR: {output_lmdb} exists. Stop."
  logging.info("Will save processed data to %s.", output_lmdb)

  def _process_one(mmcif_path: str, assembly_path: str):
    # AlphaFold3 supplementary information section 2.5.4:
    # Filtering of targets
    # - The structure must have been released to the PDB before 2020-04-30.
    # - The structure must have a reported resolution of 9 angstrom or less.
    mmcif_string = parse_mmcif_string(mmcif_path)
    mmcif_dict = MMCIF2Dict.MMCIF2Dict(io.StringIO(mmcif_string))
    header = parse_mmcif._get_header(mmcif_dict)
    if 'release_date' in header and header['release_date'] is not None:
      release_date = datetime.strptime(header['release_date'], '%Y-%m-%d')
      if release_date > release_date_cutoff:
        logging.error("Release date %s > %s.", release_date, release_date_cutoff)
        return {}
    if 'resolution' in header and header['resolution'] is not None:
      resolution = float(header['resolution'])
      if resolution > resolution_cutoff:
        logging.error("Resolution %s > %s.", resolution, resolution_cutoff)
        return {}
    data = process_one_structure(chem_comp_dir, assembly_path)
    if data:
      data.update(header)
    return data
#   data = _process_one(mmcif_paths['7er0'], assembly_paths['7er0'])
#   data and show_one_structure(data)

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
      f'Chemical components: {chem_comp_dir}\n'
      f'Input structures: {mmcif_dir}\n'
      f'Input assemblies: {assembly_dir}\n'
      f'Output lmdb: {output_lmdb}\n'
      f'Number of workers: {num_workers}\n'
      f'Release date cutoff: {release_date_cutoff}\n'
      f'Resolution cutoff: {resolution_cutoff}\n'
      f'Comments: {data_comment}\n'
    ),
  }

  pbar = tqdm(total=len(pdbids)//10000+1, desc='Processing chunks (10k)')
  for pdbid_chunk in chunks(pdbids, 10000):
    result_chunk = joblib.Parallel(n_jobs=num_workers)(
      joblib.delayed(_process_one)(mmcif_paths[_], assembly_paths[_])
      for _ in pdbid_chunk
    )
    with env.begin(write=True) as txn:
      for data in result_chunk:
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
    pbar.update(1)
  pbar.close()

  with env.begin(write=True) as txn:
    txn.put('__metadata__'.encode(), obj2bstr(metadata))

  env.close()

  show_lmdb(output_lmdb)


if __name__ == "__main__":
    main()
