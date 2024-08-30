#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import click
import lmdb
from absl import logging
from joblib import delayed, Parallel

from commons import obj2bstr
from process_mmcif import process_one_structure, remove_hydrogens_from_graph
from process_mmcif import show_lmdb, show_one_structure


logging.set_verbosity(logging.INFO)


def main(args):
    output_lmdb = str(Path(args.output_lmdb).resolve())
    assert not Path(output_lmdb).exists(), f"ERROR: {output_lmdb} exists. Stop."
    logging.info(f"Will save processed data to {output_lmdb}")

    chem_comp_path = str(Path(args.chem_comp_path).resolve())
    assert Path(chem_comp_path).exists(), f"{chem_comp_path} not found."
    logging.info(f"Chemical components information is in {chem_comp_path}")

    mmcif_dir = str(Path(args.mmcif_dir).resolve())
    assert Path(mmcif_dir).exists(), f"{mmcif_dir} not found."
    mmcif_paths = {_.name.split('.')[0]:_
                   for _ in Path(mmcif_dir).rglob('*.cif.gz')}
    assert mmcif_paths and all(4==len(_) for _ in mmcif_paths), (
        f"PDBID should be 4 characters long in {mmcif_dir}.")
    logging.info(f"{len(mmcif_paths)} structures in {mmcif_dir}.")

    benchmarkset_dir = str(Path(args.benchmarkset_dir).resolve())
    assert Path(benchmarkset_dir).exists(), f"{benchmarkset_dir} not found."
    inpdir_paths = {_.name.split('.')[0]:_
                    for _ in Path(benchmarkset_dir).rglob('*/')}
    assert inpdir_paths and all(_[4] == '_' for _ in inpdir_paths), (
        f"Number of targets different in {benchmarkset_dir} and {mmcif_dir}.")
    logging.info(f"{len(inpdir_paths)} directories in {benchmarkset_dir}.")

    assert mmcif_paths.keys() == set([_.split('_')[0] for _ in inpdir_paths]), (
        f"PDBID mismatch between {mmcif_dir} and {benchmarkset_dir}.")

    def _process_one(pdbid_ligid: str):
        try:
            pdbid, ligid = pdbid_ligid.split('_')
            data = process_one_structure(chem_comp_path, mmcif_paths[pdbid])
            nonpoly_graphs =[]
            for graph in data['nonpoly_graphs']:
                if args.remove_hydrogens:
                    graph = remove_hydrogens_from_graph(graph)
                if not args.remove_ligands or graph['name'] == ligid:
                    nonpoly_graphs.append(graph)
            data['nonpoly_graphs'] = nonpoly_graphs
            return data
        except Exception as e:
            logging.error(f"Failed to process {pdbid_ligid}, {e}")
            return {}

    # data = _process_one('7BJJ_TVW')
    # show_one_structure(data)
    selected_pdbid_ligid = inpdir_paths.keys()
    results = Parallel(n_jobs=args.num_workers)(
        delayed(_process_one)(_) for _ in selected_pdbid_ligid
    )

    metadata = {
        'keys': [],
        'num_polymers': [],
        'num_nonpolys': [],
        'structure_methods': [],
        'release_dates': [],
        'resolutions': [],
        'comment': (
            f'Created time: {datetime.now()}\n'
            f'Chemical components: {chem_comp_path}\n'
            f'Input structures: {mmcif_dir}\n'
            f'Input posebusters: {benchmarkset_dir}\n'
            f'Output lmdb: {output_lmdb}\n'
            f'Number of workers: {args.num_workers}\n'
            f'Comments: {args.data_comment}\n'
            ),
        }
    with lmdb.open(output_lmdb, map_size=1024**4).begin(write=True) as txn: # 1TB max size
        for data in results:
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
        txn.put('__metadata__'.encode(), obj2bstr(metadata))

    show_lmdb(output_lmdb)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--chem-comp-path",
                        type=click.Path(exists=True),
                        required=True,
                        help="Input mmCIF file of all chemical components.")
    parser.add_argument("--mmcif-dir",
                        type=click.Path(exists=True),
                        required=True,
                        help="Input directory of mmCIF files rsync from RCSB.")
    parser.add_argument("--benchmarkset-dir",
                        type=click.Path(exists=True),
                        required=True,
                        help="Input directory of pdb and sdf files downloaded.")
    parser.add_argument("--output-lmdb",
                        type=click.Path(exists=False),
                        default="output.lmdb",
                        help="Output lmdb file.")
    parser.add_argument("--remove-ligands",
                        action="store_true",
                        help="Key all ligands in the output lmdb.")
    parser.add_argument("--remove-hydrogens",
                        action="store_true",
                        help="Remove hydrogens for ligands in the output lmdb.")
    parser.add_argument("--num-workers",
                        type=int,
                        default=-1,
                        help="Number of workers.")
    parser.add_argument("--data-comment",
                        type=str,
                        default="PoseBusters mmCIF from RCSB before 20240630.",
                        help="Comments for output.")
    args = parser.parse_args()
    main(args)
