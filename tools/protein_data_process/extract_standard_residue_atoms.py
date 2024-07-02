#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

from Bio.PDB import MMCIF2Dict

from mmcif_parsing import mmcif_loop_to_list


STDRESIDUES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK', 'A', 'C', 'G', 'U', 'N', 'DA', 'DC', 'DG', 'DT', 'DN']


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(f'Usage: {sys.argv[0]} <chem_comp_directory>')
    chem_comp_dir = sys.argv[1]

    allatoms = []
    for name in STDRESIDUES:
        path = Path(chem_comp_dir) / f'{name}.cif'
        mmcif_obj = MMCIF2Dict.MMCIF2Dict( str(path) )
        atoms = mmcif_loop_to_list('_chem_comp_atom.', parsed_info=mmcif_obj)
        _atm = []
        for atom in atoms:
            if atom['_chem_comp_atom.type_symbol'] != 'H':
              _atm.append(atom['_chem_comp_atom.atom_id'])
        allatoms.append( (name, _atm) )
    for name, atoms in allatoms:
        key = f'"{name}":'
        value = '{"' + '", "'.join(atoms) + '"},'
        print(key, value)

    baseorder, atomorder = [], []
    for name, atoms in allatoms:
        if STDRESIDUES.index(name) < 21:
            for atom in atoms:
                if atom not in baseorder:
                    baseorder.append(atom)
        else:
            for atom in atoms:
                if atom not in atomorder:
                    atomorder.append(atom)
    line = '["' + '", "'.join(baseorder) + '"]'
    print(line)
    line = '["' + '", "'.join(atomorder) + '"]'
    print(line)
