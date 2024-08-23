#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import sys
from pathlib import Path

import numpy as np
from absl import logging
from Bio import PDB
from joblib import delayed, Parallel

from parse_mmcif import mmcif_loop_to_list
from process_mmcif import parse_mmcif_string


def check_one_assembly_from_mmcif(mmcif_path: str):
    mmcif_path = Path(mmcif_path).resolve()
    pdbid = str(mmcif_path.name).split('.')[0]
    assert len(pdbid) == 4, f"Invalid 4 characters PDBID {pdbid}."

    mmcif_string = parse_mmcif_string(str(mmcif_path))
    assert mmcif_string, f"Failed to read mmcif string for {pdbid}."

    parser = PDB.MMCIFParser(QUIET=True)
    handle = io.StringIO(mmcif_string)
    full_structure = parser.get_structure('', handle)
    parsed_info = parser._mmcif_dict

    # Check if the assembly is the same as the full structure
    assembly1 = []
    for gen in mmcif_loop_to_list('_pdbx_struct_assembly_gen.', parsed_info):
        if gen['_pdbx_struct_assembly_gen.assembly_id'] == '1':
            assembly1 = gen['_pdbx_struct_assembly_gen.asym_id_list'].split(',')
            break
    else:
        raise ValueError(f"No assembly 1 found for {pdbid}.")
    assert assembly1 and len(assembly1) == len(set(assembly1)), (
        f"{pdbid} Assembly 1 asym_id_list error.")

    # check _pdbx_struct_oper_list
    operation = []
    for oper in mmcif_loop_to_list('_pdbx_struct_oper_list.', parsed_info):
        if oper['_pdbx_struct_oper_list.id'] == '1':
            operation = [
                [
                    float(oper['_pdbx_struct_oper_list.matrix[1][1]']),
                    float(oper['_pdbx_struct_oper_list.matrix[1][2]']),
                    float(oper['_pdbx_struct_oper_list.matrix[1][3]']),
                    float(oper['_pdbx_struct_oper_list.vector[1]']),
                ],
                [
                    float(oper['_pdbx_struct_oper_list.matrix[2][1]']),
                    float(oper['_pdbx_struct_oper_list.matrix[2][2]']),
                    float(oper['_pdbx_struct_oper_list.matrix[2][3]']),
                    float(oper['_pdbx_struct_oper_list.vector[2]']),
                ],
                [
                    float(oper['_pdbx_struct_oper_list.matrix[3][1]']),
                    float(oper['_pdbx_struct_oper_list.matrix[3][2]']),
                    float(oper['_pdbx_struct_oper_list.matrix[3][3]']),
                    float(oper['_pdbx_struct_oper_list.vector[3]']),
                ],
            ]
            break
    else:
        raise ValueError(f"Assembly 1 does not have oper_list for {pdbid}.")
    operation = np.array(operation + [[0., 0., 0., 1.]])
    assert np.all(np.equal(np.eye(4), np.array(operation))), (
        f"{pdbid} Assembly 1 operation error.")

    return (pdbid, assembly1, operation)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(f'Usage: {sys.argv[0]} <mmcif_directory>')
    mmcif_dir = sys.argv[1]

    def _check_one(mmcif_path: str):
        mmcif_path = str(mmcif_path)
        try:
            data = check_one_assembly_from_mmcif(mmcif_path)
            logging.info(f"{data[0]} SUCCESS")
        except Exception as e:
            logging.error(f"Error in {mmcif_path}: {e}")

    mmcif_dir = Path(mmcif_dir).resolve()
    mmcif_paths = [_ for _ in Path(mmcif_dir).rglob("*.cif.gz")]
    assert mmcif_paths and all(11==len(_.name) for _ in mmcif_paths), (
        f"PDBID should be 4 characters long in {mmcif_dir}.")
    logging.info(f"Processing {len(mmcif_paths)} structures in {mmcif_dir}.")

    Parallel(n_jobs=-1)(delayed(_check_one)(p) for p in mmcif_paths)
