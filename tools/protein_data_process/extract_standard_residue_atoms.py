#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Mapping, Optional, Sequence, Tuple

from Bio.PDB import MMCIF2Dict
from pathlib import Path
from mmcif_parsing import mmcif_loop_to_list, mmcif_loop_to_dict


STDRESIDUES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK', 'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT', 'N']

if __name__ == '__main__':

    chem_comp_dir = 'chem_comp_files'

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
        print(f"\"{name}\": {atoms},".replace('[', '{').replace(']', '}'))

    na = {'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT', 'N'}
    order = []
    for name, atoms in allatoms:
        if name not in na: continue
        for atom in atoms:
            if atom not in order:
                order.append(atom)
    print('[', end='')
    for a in order:
        print(f'"{a}", ', end='')
    print(']')
