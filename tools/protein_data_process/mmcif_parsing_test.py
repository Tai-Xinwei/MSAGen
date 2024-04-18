#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time

from Bio.PDB import MMCIF2Dict

import mmcif_parsing


if len(sys.argv) != 2:
    sys.exit(f"Usage: {sys.argv[0]} <input_mmcif_path>")
inppath = sys.argv[1]

filename = os.path.basename(inppath)
assert len(filename) == 8 and filename.endswith('.cif'), (
    f"ERROR: Invalid filename: {filename}")

# parse mmcif file
with open(inppath, 'r') as fp:
    cif_string = fp.read()
pdb_code = filename[:4]

curr = time.time()
parsing_result = mmcif_parsing.parse_structure(file_id=pdb_code, mmcif_string=cif_string)
print("Elapsed time for mmcif_parsing.parse_structure: ", time.time()-curr)
assert parsing_result.mmcif_object, (
    f"ERROR: Parsing failed for {inppath} {parsing_result}")

# show mmcif parsing result
obj = parsing_result.mmcif_object
assert obj.chain_to_seqres.keys() == obj.seqres_to_structure.keys(), (
    f"ERROR: chain_to_seqres and seqres_to_structure have different chains")
for chain_id, seqres in sorted(obj.chain_to_seqres.items(), key=lambda x: x[0]):
    print('-'*80, f"Chain_{chain_id} {seqres}", sep='\n')
    for i, (residue, atoms) in enumerate(obj.seqres_to_structure[chain_id]):
        print(i, residue, len(atoms), atoms[0] if atoms else None)
print(obj.file_id)
print(obj.header)
print(obj.structure)
print( sorted(obj.chain_to_seqres.keys()) )
print( type(obj.raw_string) )

#with open(inppath, 'r') as fp:
#    parsed_info = MMCIF2Dict.MMCIF2Dict(fp)
#print("----------Processing polymer chains------------------------------------")
#polymer = mmcif_parsing._get_polymer_structure(parsed_info=parsed_info)
#for chain_id, str_info in polymer.items():
#    print('-'*80, f"Chain_{chain_id} {len(str_info)}", sep='\n')
#    for idx, (residue, atoms) in str_info.items():
#        print(idx, residue, len(atoms))
#
#print("----------Processing nonpoly chains----------------------------------------")
#nonpoly = mmcif_parsing._get_nonpoly_structure(parsed_info=parsed_info)
#for chain_id, str_info in nonpoly.items():
#    print('-'*80, f"Chain_{chain_id} {len(str_info)}", sep='\n')
#    for idx, (residue, atoms) in str_info.items():
#        print(idx, residue, len(atoms))
