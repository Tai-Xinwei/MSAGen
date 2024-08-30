#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dataclasses
import gzip
import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union

import click
import lmdb
import numpy as np
from absl import logging
from Bio.PDB import MMCIF2Dict
from joblib import delayed, Parallel
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

import parse_mmcif
from commons import bstr2obj, obj2bstr
from parse_mmcif import ResidueAtPosition
from parse_mmcif import mmcif_loop_to_list
from residue_constants import ATOMORDER, RESIDUEATOMS

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from sfm.data.mol_data.utils.molecule import mol2graph


logging.set_verbosity(logging.INFO)

NUM2SYM = {_: Chem.GetPeriodicTable().GetElementSymbol(_+1) for _ in range(118)}
SYM2NUM = {Chem.GetPeriodicTable().GetElementSymbol(_+1): _ for _ in range(118)}

STDRES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK', # Protein
    'DA', 'DC', 'DG', 'DT', 'DN', # DNA
    'A', 'C', 'G', 'U', 'N', # RNA
}

SKIPCCD = {
    'HOH', # Water
    'SO4', 'GOL', 'EDO', 'PO4', 'ACT', 'PEG', 'DMS', 'TRS', 'PGE', 'PG4', 'FMT', 'EPE', 'MPD', 'MES', 'CD', 'IOD', # AlphaFold3 crystallization aids
    '0VI', 'A9J', 'BF5', 'H9C', 'USN', 'VOB', # Wrong mmCIF files
    '07D', '8P8', 'ASX', 'BCB', 'CHL', 'CL0', 'CL1', 'CL2', 'CL7', 'D3O', 'D8U', 'DOD', 'DUM', 'GLX', 'HE5', 'HEG', 'HES', 'ND4', 'NWN', 'PMR', 'S5Q', 'SPW', 'TSD', 'UNL', 'UNX', # Failed to generate RDKit molecule
    '08T', '0I7', '0MI', '0OD', '10R', '10S', '1KW', '1MK', '1WT', '25X', '25Y', '26E', '2FK', '34B', '39B', '39E', '3JI', '3UQ', '3ZZ', '4A6', '4EX', '4IR', '4LA', '5L1', '6BP', '6ER', '7Q8', '7RZ', '8M0', '8WV', '9JA', '9JJ', '9JM', '9TH', '9UK', 'A1ALJ', 'A1H7J', 'A1H8D', 'A1ICR', 'AOH', 'B1M', 'B8B', 'BBQ', 'BVR', 'CB5', 'CFN', 'COB', 'CWO', 'D0X', 'D6N', 'DAE', 'DAQ', 'DGQ', 'DKE', 'DVT', 'DW1', 'DW2', 'E52', 'EAQ', 'EJ2', 'ELJ', 'FDC', 'FEM', 'FLL', 'FNE', 'FO4', 'GCR', 'GIX', 'GXW', 'GXZ', 'HB1', 'HFW', 'HUJ', 'I8K', 'ICE', 'ICG', 'ICH', 'ICS', 'ICZ', 'IK6', 'IV9', 'IWL', 'IWO', 'J7T', 'J8B', 'JGH', 'JI8', 'JSU', 'K6G', 'K9G', 'KCO', 'KEG', 'KHN', 'KK5', 'KKE', 'KKH', 'KYS', 'KYT', 'LD3', 'M6O', 'M7E', 'ME3', 'MNQ', 'MO7', 'MYW', 'N1B', 'NA2', 'NA5', 'NA6', 'NAO', 'NAW', 'NE5', 'NFC', 'NFV', 'NMQ', 'NMR', 'NT3', 'O1N', 'O93', 'OEC', 'OER', 'OEX', 'OEY', 'ON6', 'ONP', 'OS1', 'OSW', 'OT1', 'OWK', 'OXV', 'OY5', 'OY8', 'OZN', 'P5F', 'P5T', 'P6D', 'P6Q', 'P7H', 'P7Z', 'P82', 'P8B', 'PHF', 'PNQ', 'PQJ', 'Q2Z', 'Q38', 'Q3E', 'Q3H', 'Q3K', 'Q3N', 'Q3Q', 'Q3T', 'Q3W', 'Q4B', 'Q65', 'Q7V', 'QIY', 'QT4', 'R1N', 'R5N', 'R5Q', 'RAX', 'RBN', 'RCS', 'REI', 'REJ', 'REP', 'REQ', 'RIR', 'RTC', 'RU7', 'RUC', 'RUD', 'RUH', 'RUI', 'S18', 'S31', 'S5T', 'S9F', 'SIW', 'SWR', 'T0P', 'TEW', 'U8G', 'UDF', 'UGO', 'UO3', 'UTX', 'UZC', 'V22', 'V9G', 'VA3', 'VAV', 'VFY', 'VI6', 'VL9', 'VOF', 'VPC', 'VSU', 'VTU', 'VTZ', 'WCO', 'WGB', 'WJS', 'WK5', 'WNI', 'WO2', 'WO3', 'WRK', 'WUX', 'WZW', 'X33', 'X3P', 'X5M', 'X5W', 'XC3', 'XCO', 'XCU', 'XZ6', 'Y59', 'Y77', 'YIA', 'YJ6', 'YJK', 'YQ1', 'YQ4', 'ZIV', 'ZJ5', 'ZKG', 'ZPT', 'ZRW', 'ZV2', # RDKit fail reading
    '0H2', '0KA', '1CL', '1Y8', '2NO', '2PT', '3T3', '402', '4KV', '4WV', '4WW', '4WX', '6ML', '6WF', '72B', '74C', '8CY', '8JU', '8ZR', '9CO', '9S8', '9SQ', '9UX', 'ARS', 'B51', 'BF8', 'BGQ', 'BJ8', 'BRO', 'CFM', 'CH2', 'CLF', 'CLO', 'CLP', 'CU6', 'CUV', 'CYA', 'CYO', 'CZZ', 'DML', 'DW5', 'EL9', 'ER2', 'ETH', 'EXC', 'F3S', 'F4S', 'FDD', 'FLO', 'FS2', 'FS3', 'FS4', 'FS5', 'FSF', 'FSX', 'FU8', 'FV2', 'GAK', 'GFX', 'GK8', 'GTE', 'GXB', 'H', 'H1T', 'H79', 'HEO', 'HME', 'HNN', 'ICA', 'IDO', 'IF6', 'IHW', 'ITM', 'IWZ', 'IX3', 'J7Q', 'J85', 'J8E', 'J9H', 'JCT', 'JQJ', 'JSC', 'JSD', 'JSE', 'JY1', 'KBW', 'L8W', 'LFH', 'LPJ', 'MAP', 'MEO', 'MHM', 'MHX', 'MNH', 'MNR', 'MTN', 'NFS', 'NGN', 'NH', 'NH2', 'NMO', 'NO', 'NYN', 'O', 'OET', 'OL3', 'OL4', 'OL5', 'OLS', 'OMB', 'OME', 'OX', 'OXA', 'OXO', 'P4J', 'PT7', 'Q61', 'QTR', 'R1B', 'R1F', 'R7A', 'R9H', 'RCY', 'RFQ', 'RPS', 'RQM', 'RRE', 'RXR', 'S', 'S32', 'S3F', 'SE', 'SF3', 'SF4', 'SFO', 'SFS', 'SI0', 'SI7', 'SVP', 'T9T', 'TBY', 'TDJ', 'TE', 'TL', 'TML', 'U0J', 'UFF', 'UJI', 'UJY', 'V1A', 'VHR', 'VQ8', 'VV2', 'VV7', 'WCC', 'XCC', 'XX2', 'YF8', 'YPT', 'ZJZ', 'ZKP', # SMILES different (lone-pair electron)
}


@click.group()
def cli():
    pass


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def parse_mmcif_string(mmcif_path: str) -> str:
    mmcif_string = ''
    if mmcif_path.endswith('.cif'):
        with open(mmcif_path, 'r') as fp:
            mmcif_string = fp.read()
    elif mmcif_path.endswith('.cif.gz'):
        with gzip.open(mmcif_path, 'rt') as fp:
            mmcif_string = fp.read()
    else:
        logging.error(f"{mmcif_path} must endswith .cif or .cif.gz.")
    return mmcif_string


def split_chem_comp_full_string(chem_comp_full_string: str) -> Mapping[str, str]:
    chem_comp_strings = {}
    if chem_comp_full_string.startswith('data_'):
        strings = chem_comp_full_string.split('data_')
        del strings[0] # empty string for first element
        for s in strings:
            lines = s.split('\n')
            chem_comp_strings[lines[0]] = 'data_' + s
    return chem_comp_strings


def check_chem_comps(parsed_info: MMCIF2Dict) -> bool:
    # chem_comps ids in _chem_comp
    _entries = mmcif_loop_to_list('_chem_comp.', parsed_info)
    chem_comps = set( [_['_chem_comp.id'] for _ in _entries] )
    # all atom comps parsed from _atom_site
    _entries = mmcif_loop_to_list('_atom_site.', parsed_info)
    atom_comps = set( [_['_atom_site.label_comp_id'] for _ in _entries] )
    # polymer comps parsed from _entity_poly_seq
    _entries = mmcif_loop_to_list('_entity_poly_seq.', parsed_info)
    polyseq_comps = set( [_['_entity_poly_seq.mon_id'] for _ in _entries] )
    # polymer comps parsed from _pdbx_poly_seq_scheme
    _entries = mmcif_loop_to_list('_pdbx_poly_seq_scheme.', parsed_info)
    pdbx_polyseq = set( [_['_pdbx_poly_seq_scheme.mon_id'] for _ in _entries] )
    return  atom_comps.issubset(chem_comps) and polyseq_comps == pdbx_polyseq


def process_polymer_chain(polymer_chain: Sequence[dict]) -> Mapping[str, Any]:
    """Process one polymer chain."""
    resname, seqres, restype, center_coord, allatom_coord = [], [], [], [], []
    for residue in polymer_chain:
        resname.append(residue['name'])
        seqres.append(residue['seqres'])
        restype.append(residue['restype'])
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
        center_coord.append(c_coord)
        allatom_coord.append(a_coord)
    data = {
        'resname': np.array(resname),
        'seqres': np.array(seqres),
        'restype': np.array(restype),
        'center_coord': np.array(center_coord, dtype=np.float32),
        'allatom_coord': np.array(allatom_coord, dtype=np.float32),
    }
    return data


def create_rdkitmol(atoms: Sequence[str],
                    charge: int,
                    bond_orders: Sequence[Tuple[int, int, str]],
                    atom_charges: Sequence[int]
                    ) -> Chem.Mol:
    """Create an RDKit molecule using atom types and bond orders."""
    with Chem.RWMol() as mw:
        for a, c in zip(atoms, atom_charges):
            atom = Chem.Atom(a)
            atom.SetFormalCharge(c)
            mw.AddAtom(atom)

        # Ignore RDKit Warnings (https://github.com/rdkit/rdkit/issues/2683)
        # Explicit Valence Error - Partial Sanitization (https://www.rdkit.org/docs/Cookbook.html#explicit-valence-error-partial-sanitization)
        # Pre-condition Violation (https://github.com/rdkit/rdkit/issues/1596)
        mw.UpdatePropertyCache(strict=False)

        for i, j, order in bond_orders:
            if order == 1:
                mw.AddBond(i, j, Chem.BondType.SINGLE)
            elif order == 2:
                mw.AddBond(i, j, Chem.BondType.DOUBLE)
            elif order == 3:
                mw.AddBond(i, j, Chem.BondType.TRIPLE)
            else:
                mw.AddBond(i, j, Chem.BondType.SINGLE)
        Chem.SanitizeMol(mw)

        if Chem.GetFormalCharge(mw) != charge:
            raise ValueError(f"mol charge {Chem.GetFormalCharge(mw)}!={charge}")

        return mw


def chemcomp2graph(chem_comp_string: str) -> Mapping[str, Any]:
    # process ideal chemical component
    handle = io.StringIO(chem_comp_string)
    mmcifdict = MMCIF2Dict.MMCIF2Dict(handle)
    chem_comp_id = mmcifdict['_chem_comp.id'][0]
    pdbx_formal_charge = int(mmcifdict['_chem_comp.pdbx_formal_charge'][0])
    # parse chem_comp_atoms
    id2index = {}
    chem_comp_atoms = []
    for i, atom in enumerate(mmcif_loop_to_list('_chem_comp_atom.', mmcifdict)):
        _id = atom['_chem_comp_atom.atom_id']
        assert _id not in id2index, f"Duplicate atom {_id} in {chem_comp_id}."
        id2index[_id] = i
        _symbol = atom['_chem_comp_atom.type_symbol']
        _charge = atom['_chem_comp_atom.charge']
        _mc = [np.nan, np.nan, np.nan]
        if atom['_chem_comp_atom.model_Cartn_x'] != '?':
            _mc = [float(atom['_chem_comp_atom.model_Cartn_x']),
                   float(atom['_chem_comp_atom.model_Cartn_y']),
                   float(atom['_chem_comp_atom.model_Cartn_z'])]
        _ci = [np.nan, np.nan, np.nan]
        if atom['_chem_comp_atom.pdbx_model_Cartn_x_ideal'] != '?':
            _ci = [float(atom['_chem_comp_atom.pdbx_model_Cartn_x_ideal']),
                   float(atom['_chem_comp_atom.pdbx_model_Cartn_y_ideal']),
                   float(atom['_chem_comp_atom.pdbx_model_Cartn_z_ideal'])]
        chem_comp_atoms.append({
            'id': _id,
            'symbol': _symbol[0].upper() + _symbol[1:].lower(),
            'charge': int(_charge) if _charge != '?' else 0,
            'model_cartn': np.array(_mc),
            'cartn_ideal': np.array(_ci),
            })
    # parse chem_comp_bonds
    atompairs = set()
    chem_comp_bonds = []
    for bond in mmcif_loop_to_list('_chem_comp_bond.', mmcifdict):
        id1 = bond['_chem_comp_bond.atom_id_1']
        id2 = bond['_chem_comp_bond.atom_id_2']
        assert id1 in id2index and id2 in id2index, (
            f"Invalid bond for atom pair ({id1}, {id2}) in {chem_comp_id}.")
        assert (id1, id2) not in atompairs and (id2, id1) not in atompairs, (
            f"Duplicate atom pair ({id1}, {id2}) in {chem_comp_id}.")
        atompairs.add((id1, id2))
        chem_comp_bonds.append({
            'id1': id1,
            'id2': id2,
            'order': bond['_chem_comp_bond.value_order'],
            })
    # parse ideal coordinates
    model_cartn = np.array([_['model_cartn'] for _ in chem_comp_atoms])
    cartn_ideal = np.array([_['cartn_ideal'] for _ in chem_comp_atoms])
    if sum(np.isnan(model_cartn).ravel()) < sum(np.isnan(cartn_ideal).ravel()):
        cartn = np.nan_to_num(model_cartn)
    else:
        cartn = np.nan_to_num(cartn_ideal)
    # extract feature for chem_comp atoms and bonds
    atomids = [_['id'] for _ in chem_comp_atoms]
    symbols = [_['symbol'] for _ in chem_comp_atoms]
    charges = [_['charge'] for _ in chem_comp_atoms]
    BONDORDER = {'SING': 1, 'DOUB': 2, 'TRIP': 3}
    orders = [(id2index[_['id1']], id2index[_['id2']], BONDORDER[_['order']])
              for _ in chem_comp_bonds]
    try:
        # convert atom symbols and bond orders to mol by using RDKit
        rdkitmol = create_rdkitmol(symbols, pdbx_formal_charge, orders, charges)
        # RDKit generate conformers for molecules using ETKDGv3 method
        params = AllChem.ETKDGv3()
        params.randomSeed = 12345
        stat = AllChem.EmbedMolecule(rdkitmol, params)
        if stat == 0:
            cartn = np.array(rdkitmol.GetConformer().GetPositions())
        rdkitmol.RemoveAllConformers()
        graph = {
            'name': chem_comp_id,
            'pdbx_formal_charge': pdbx_formal_charge,
            'atomids': np.array(atomids),
            'symbols': np.array(symbols),
            'charges': np.array(charges),
            'coords': np.array(cartn, dtype=np.float32),
            'orders': np.array(orders),
            'rdkitmol': rdkitmol,
            }
        # use sfm.data.mol_data.utils.molecule.mol2graph to generate mol graph
        # graph update key node_feat, edge_index, edge_feat, num_nodes
        graph.update(mol2graph(rdkitmol))
        return graph
    except Exception as e:
        logging.error(f"Failed to parse {chem_comp_id}, {e}")
        return {}


def process_nonpoly_residue(residue: ResidueAtPosition,
                            chem_comp_graph: dict) -> Mapping[str, Any]:
    graph = {'chain_id': residue.position.chain_id,
             'residue_number': residue.position.residue_number}
    graph.update(chem_comp_graph)
    node_coord = [[np.nan]*3 for _ in chem_comp_graph['atomids']]
    id2index = {id: _ for _, id in enumerate(chem_comp_graph['atomids'])}
    for atom in residue.atoms:
        if atom.name not in id2index:
            # AlphaFold3 supplementary information section 2.5.4
            # Filtering of bioassemblies: For residues or small molecules
            # with CCD codes, atoms outside of the CCD code’s defined set
            # of atom names are removed.
            continue
        node_coord[id2index[atom.name]] = [atom.x, atom.y, atom.z]
    graph['node_coord'] = np.array(node_coord, dtype=np.float32)
    return graph


def remove_hydrogens_from_graph(graph):
    data = {
        'chain_id': graph['chain_id'],
        'residue_number': graph['residue_number'],
        'name': graph['name'],
        'pdbx_formal_charge': graph['pdbx_formal_charge'],
    }
    mask = graph['symbols'] != 'H'
    idx_old2new = {idx:i for i, idx in enumerate(np.where(mask)[0])}
    new_orders = [(idx_old2new[i], idx_old2new[j], _)
                  for i, j, _ in graph['orders'] if mask[i] and mask[j]]
    rdkitmol = Chem.RemoveHs(graph['rdkitmol'])
    data.update({
        'atomids': graph['atomids'][mask],
        'symbols': graph['symbols'][mask],
        'charges': graph['charges'][mask],
        'coords': graph['coords'][mask],
        'node_coord': graph['node_coord'][mask],
        'orders': np.array(new_orders),
        'rdkitmol': rdkitmol,
    })
    rdkitmol.RemoveAllConformers()
    data.update(mol2graph(rdkitmol))
    return data


def chain_type_to_one_letter(chain_type: str) -> str:
    code = '?'
    if chain_type == 'polypeptide(L)':
        code = 'p'
    elif chain_type == 'polypeptide(D)':
        code = 'p'
    elif chain_type == 'polydeoxyribonucleotide':
        code = 'd'
    elif chain_type == 'polyribonucleotide':
        code = 'r'
    return code


def process_one_structure(chem_comp_path: str,
                          mmcif_path: str,
                          assembly_path: str = None,
                          ) -> Mapping[str, Union[str, dict, list]]:
    """Parse a mmCIF file and convert data to list of dict."""
    try:
        chem_comp_path = str(Path(chem_comp_path).resolve())
        chem_comp_full_string = parse_mmcif_string(chem_comp_path)
        assert chem_comp_full_string.startswith('data_'), (
            f"Failed to read chem_comp from {chem_comp_path}.")
        chem_comp_strings = split_chem_comp_full_string(chem_comp_full_string)
        assert chem_comp_strings, f"Failed to split {chem_comp_path}."

        mmcif_path = str(Path(mmcif_path).resolve())
        pdbid = Path(mmcif_path).name.split('.')[0]
        assert len(pdbid) == 4, f"Invalid 4 characters PDBID {pdbid}."
        mmcif_string = parse_mmcif_string(mmcif_path)
        assert mmcif_string, f"Failed to read mmcif string for {pdbid}."

        header = parse_mmcif._get_header(
            MMCIF2Dict.MMCIF2Dict(io.StringIO(mmcif_string)))
        assert header, f"Failed to parse header for {pdbid}."

        if assembly_path is not None:
            assembly_path = str(Path(assembly_path).resolve())
            newid = Path(assembly_path).name.split('.')[0].split('-')[0]
            assert pdbid == newid, f"Invalid ID in {assembly_path} for {pdbid}."
            mmcif_string = parse_mmcif_string(str(assembly_path))
            assert mmcif_string, f"Failed to read assembly string for {pdbid}."

        # Parse mmcif file by modified AlphaFold mmcif_parsing.py
        result = parse_mmcif.parse(file_id=pdbid, mmcif_string=mmcif_string)
        assert result.mmcif_object, f"The errors are {result.errors}"
        # print(result.mmcif_object.file_id, result.mmcif_object.header)

        # Process polymer chains
        polymer_chains = {}
        for chain_id, chain in result.mmcif_object.polymer_chains.items():
            if len(chain.residues) < 4:
                # AlphaFold3 supplementary information section 2.5.4
                # Filtering of targets: Polymer chain containing fewer than 4
                # resolved residues is filtered out
                continue
            # print('-'*80, f"{pdbid}_{chain_id}", chain.type, sep='\n')
            seqres = chain.pdbx_can.replace('\n', '')
            aatype = chain_type_to_one_letter(chain.type)
            if aatype == '?':
                # Exclude unknown type, e.g., 'other', 'peptide nucleic acid',
                # and 'polydeoxyribonucleotide/polyribonucleotide hybrid'
                logging.warning(f"Unknown chain type {chain.type} for {pdbid}.")
                continue
            current_chain = []
            for aa, residue in zip(seqres, chain.residues):
                if residue.position and chain_id != residue.position.chain_id:
                    raise ValueError(f"Chain '{chain_id}' has wrong {residue}")
                # Process polymer residues for protein, DNA and RNA.
                resdict = dataclasses.asdict(residue)
                resdict.update({'seqres': aa, 'restype': aatype})
                current_chain.append(resdict)
            polymer = process_polymer_chain(current_chain)
            # Check if polymer has center atom or all nan
            if not polymer or np.all(np.isnan(polymer['center_coord'])):
                logging.warning(f"Chain {pdbid}_{chain_id} has no center atom.")
                continue
            polymer_chains[chain_id] = polymer
        assert polymer_chains, f"Has no desirable chains for {pdbid}."

        # Process non-polymer chains
        chem_comp_graphs = {}
        nonpoly_graphs = []
        for chain_id, residues in result.mmcif_object.nonpoly_chains.items():
            # print('-'*80, f"{pdbid}_{chain_id}", len(residues), sep='\n')
            for residue in residues:
                if residue.name in SKIPCCD or not residue.atoms:
                    # Skip residue in exclusion list or has no atom
                    continue
                if residue.name not in chem_comp_graphs:
                    chem_comp_graphs[residue.name] = chemcomp2graph(
                        chem_comp_strings[residue.name])
                graph = process_nonpoly_residue(residue,
                                                chem_comp_graphs[residue.name])
                if not graph or np.all(np.isnan(graph['node_coord'])):
                    logging.warning(f"Residue {pdbid} {residue.name} no coord.")
                    continue
                nonpoly_graphs.append(graph)

        logging.debug(f"{mmcif_path} processed successfully.")
        data = {
            'pdbid': pdbid,
            'structure_method': header['structure_method'],
            'release_date': header['release_date'],
            'resolution': header['resolution'],
            'polymer_chains': polymer_chains,
            'nonpoly_graphs': nonpoly_graphs,
        }
        return data
    except Exception as e:
        logging.error(f"{mmcif_path} processed failed, {e}")
        return {}


def show_one_structure(data: Mapping[str, Union[str, dict, list]]) -> None:
    """Show one processed data."""
    print(data.keys())
    print(data['pdbid'])
    print(data['structure_method'])
    print(data['release_date'])
    print("resolution", data['resolution'])
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
        for i, axis in enumerate('xyz'):
            arr = [f'{_:.3f}' for _ in polymer['center_coord'][:10, i]]
            print(f"center_coord[:10].{axis}   : [{', '.join(arr)}]")
        for i, axis in enumerate('xyz'):
            arr = [f'{_:.3f}' for _ in polymer['allatom_coord'][:10, 0, i]]
            print(f"allatom_coord[:10,0].{axis}: [{', '.join(arr)}]")
    print('-'*80)
    print("nonpoly_graphs", len(data['nonpoly_graphs']))
    for i, graph in enumerate(data['nonpoly_graphs']):
        print('-'*80)
        print(graph.keys())
        print(f"{i}_{data['pdbid']}_{graph['chain_id']}_{graph['name']} "
              f"num_nodes={graph['num_nodes']} "
              f"charge={graph['pdbx_formal_charge']} ",
              end='')
        for k in ['node_coord', 'node_feat', 'edge_index', 'edge_feat']:
            print(f"{k}={graph[k].shape}", end=' ')
        print()
        arr = [f'{_+1:<2}' for _ in graph['node_feat'][:, 0]]
        print("".join(arr))
        arr = [f'{NUM2SYM[_]:<2}' for _ in graph['node_feat'][:, 0]]
        print("".join(arr))
        arr = [f'{_:<2}' for _ in graph['charges']]
        print("".join(arr))
        for i, axis in enumerate('xyz'):
            arr = [f'{_:.3f}' for _ in graph['coords'][:10, i]]
            print(f"coords[:10].{axis}    : [{', '.join(arr)}]")
        for i, axis in enumerate('xyz'):
            arr = [f'{_:.3f}' for _ in graph['node_coord'][:10, i]]
            print(f"node_coord[:10].{axis}: [{', '.join(arr)}]")
        print(f"node_feat[:10,:3]:", graph['node_feat'][:10, :3].tolist())
        print(f"edge_index[:,:10]:", graph['edge_index'][:, :10].tolist())
        print(f"edge_feat[:10,:] :", graph['edge_feat'][:10, :].tolist())


def show_lmdb(lmdbdir: Path):
    with lmdb.open(str(lmdbdir), readonly=True).begin(write=False) as txn:
        metavalue = txn.get('__metadata__'.encode())
        assert metavalue, f"'__metadata__' not found in {lmdbdir}."

        metadata = bstr2obj(metavalue)

        assert 'keys' in metadata, (
            f"'keys' not in metadata for {lmdbdir}.")
        assert 'structure_methods' in metadata, (
            f"'structure_methods' not in metadata for {lmdbdir}.")
        assert 'release_dates' in metadata, (
            f"'release_dates' not in metadata for {lmdbdir}.")
        assert 'resolutions' in metadata, (
            f"'resolutions' not in metadata for {lmdbdir}.")
        assert 'comment' in metadata, (
            f"'comment' not in metadata for {lmdbdir}.")

        print('-'*80)
        print(metadata['comment'], end='')
        for k, v in metadata.items():
            k != 'comment' and print(k, len(v))
        print(f"{len(metadata['keys'])} samples in {lmdbdir}" )
        print(f"metadata['keys'][:10]={metadata['keys'][:10]}")


@cli.command()
@click.option("--chem-comp-path",
              type=click.Path(exists=True),
              required=True,
              help="Input mmCIF file of all chemical components.")
@click.option("--mmcif-path",
              type=click.Path(exists=True),
              required=True,
              help="Input path of one mmCIF file rsync from RCSB.")
@click.option("--assembly-path",
              type=click.Path(exists=True),
              default=None,
              help="Input path of one assembly mmCIF file rsync from RCSB.")
def process_one(chem_comp_path: str, mmcif_path: str, assembly_path: str):
    """Process one mmCIF file and print the result."""
    chem_comp_path = str(Path(chem_comp_path).resolve())
    mmcif_path = str(Path(mmcif_path).resolve())
    if assembly_path is not None:
        assembly_path = str(Path(assembly_path).resolve())
    print(chem_comp_path)
    print(mmcif_path)
    print(assembly_path)
    data = process_one_structure(chem_comp_path, mmcif_path, assembly_path)
    data and show_one_structure(data)


@cli.command()
@click.option("--chem-comp-path",
              type=click.Path(exists=True),
              required=True,
              help="Input mmCIF file of all chemical components.")
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
@click.option("--data-comment",
              type=str,
              default="PDB snapshot from rsync://rsync.wwpdb.org::ftp/data/ with --port=33444 on 20240630.",
              help="Comments for output.")
def process(chem_comp_path: str,
            mmcif_dir: str,
            assembly_dir: str,
            output_lmdb: str,
            num_workers: int,
            data_comment: str,
            ) -> None:
    """Process mmCIF files from directory and save to lmdb."""
    mmcif_dir = str(Path(mmcif_dir).resolve())
    mmcif_paths = {_.name.split('.')[0]:_
                   for _ in Path(mmcif_dir).rglob("*.cif.gz")}
    assert mmcif_paths and all(4==len(_) for _ in mmcif_paths), (
        f"PDBID should be 4 characters long in {mmcif_dir}.")
    logging.info(f"{len(mmcif_paths)} structures in {mmcif_dir}.")

    assembly_dir = str(Path(assembly_dir).resolve())
    assembly_paths = {_.name.split('.')[0].split('-')[0]:_
                      for _ in Path(assembly_dir).rglob("*-assembly1.cif.gz")}
    assert assembly_paths and all(4==len(_) for _ in assembly_paths), (
        f"PDBID should be 4 characters long in {assembly_dir}.")
    logging.info(f"{len(assembly_paths)} assemblies in {assembly_dir}.")

    pdbids = list(set(mmcif_paths.keys()) & set(assembly_paths.keys()))
    logging.info(f"{len(pdbids)} pdbids in structures and assemblies.")

    chem_comp_path = str(Path(chem_comp_path).resolve())
    logging.info(f"Chemical components information is in {chem_comp_path}")

    output_lmdb = str(Path(output_lmdb).resolve())
    assert not Path(output_lmdb).exists(), f"ERROR: {output_lmdb} exists. Stop."
    logging.info(f"Will save processed data to {output_lmdb}")

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
            f'Chemical components: {chem_comp_path}\n'
            f'Input structures: {mmcif_dir}\n'
            f'Input assemblies: {assembly_dir}\n'
            f'Output lmdb: {output_lmdb}\n'
            f'Number of workers: {num_workers}\n'
            f'Comments: {data_comment}\n'
            ),
        }

    pbar = tqdm(total=len(pdbids)//10000+1, desc='Processing chunks (10k)')
    for pdbid_chunk in chunks(pdbids, 10000):
        result_chunk = Parallel(n_jobs=num_workers)(
            delayed(process_one_structure)(
                chem_comp_path,
                mmcif_paths[_],
                assembly_paths[_],
            )
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


@cli.command()
@click.option("--chem-comp-path",
              type=click.Path(exists=True),
              required=True,
              help="Input mmCIF file of all chemical components.")
@click.option("--sdf-dir",
              type=click.Path(exists=True),
              required=True,
              help="Input directory of SDF files.")
def check_chem_comp(chem_comp_path: str, sdf_dir: str) -> None:
    chem_comp_path = Path(chem_comp_path).resolve()
    logging.info(f"Chemical components information is in {chem_comp_path}")

    sdf_dir = Path(sdf_dir).resolve()
    logging.info(f"The ideal SDF files in {sdf_dir}")

    chem_comp_path = Path(chem_comp_path).resolve()
    chem_comp_full_string = parse_mmcif_string(str(chem_comp_path))
    assert chem_comp_full_string.startswith('data_'), (
        f"Failed to read chem_comp from {chem_comp_path}.")

    chem_comp_strings = split_chem_comp_full_string(chem_comp_full_string)
    assert chem_comp_strings, f"Failed to split {chem_comp_path}."
    logging.info(f"{len(chem_comp_strings)} chem_comp in {chem_comp_path}.")

    def _check_one(name: str) -> int:
        if name in SKIPCCD:
            return -2
        try:
            sdf_path = sdf_dir / f"{name}_ideal.sdf"
            assert sdf_path.exists(), f"SDF file does not exist for {name}."
            with open(sdf_path, 'r') as fp:
                lines = fp.readlines()
            idx = lines.index('> <OPENEYE_ISO_SMILES>\n')
            smiles = lines[idx+1].strip()
            assert smiles, f"No SMILES in sdf file {name}."

            mol1 = Chem.MolFromSmiles(smiles)
            assert mol1, f"failed to create molecule {name} from SMILES."
            mol1 = Chem.RemoveHs(mol1)
            can1 = Chem.MolToSmiles(mol1, isomericSmiles=False)
            mol1 = Chem.MolFromSmiles(can1)
            can1 = Chem.MolToSmiles(mol1, isomericSmiles=False)
            # iso1 = Chem.MolToSmiles(mol1, isomericSmiles=True)
            # print(can1)

            mol2 = chemcomp2graph(chem_comp_strings[name])['rdkitmol']
            assert mol2, f"failed to create RDKit molecule {name}."
            mol2 = Chem.RemoveHs(mol2)
            can2 = Chem.MolToSmiles(mol2, isomericSmiles=False)
            mol2 = Chem.MolFromSmiles(can2)
            can2 = Chem.MolToSmiles(mol2, isomericSmiles=False)
            # iso2 = Chem.MolToSmiles(mol2, isomericSmiles=True)
            # print(can2)

            return 1 if can1 == can2 else 0
        except Exception as e:
            logging.error(f"Check {name} failed, {e}")
            return -1

    logging.info(f"Checking {len(chem_comp_strings)} chem_comp one by one ...")
    names = sorted(chem_comp_strings.keys())
    results = [(_check_one(_), _) for _ in tqdm(names)]
    STATUS = {1: 'EQUAL', 0: 'DIFFERENT', -1: 'FAILED', -2: 'SKIP'}
    for r in results:
        print(STATUS[r[0]], r[1])


if __name__ == "__main__":
    cli()
