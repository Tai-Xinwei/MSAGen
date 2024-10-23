#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dataclasses
import gzip
import io
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union

import click
import joblib
import lmdb
import numpy as np
from absl import logging
from Bio.PDB import MMCIF2Dict
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

import parse_mmcif
from commons import bstr2obj, obj2bstr
from parse_mmcif import MmcifChain
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
  """Parse mmcif file .cif and .cif.gz to string."""
  mmcif_string = ''
  if mmcif_path.endswith('.cif'):
    with open(mmcif_path, 'r') as fp:
      mmcif_string = fp.read()
  elif mmcif_path.endswith('.cif.gz'):
    with gzip.open(mmcif_path, 'rt') as fp:
      mmcif_string = fp.read()
  else:
    logging.error("File %s must endswith .cif or .cif.gz.", mmcif_path)
  return mmcif_string


def split_chem_comp_full_string(chem_comp_full_string: str) -> Mapping[str, str]:
  """Split chem_comp_full_string to chem_comp_strings by comp_id."""
  chem_comp_strings = {}
  if chem_comp_full_string.startswith('data_'):
    strings = chem_comp_full_string.split('data_')
    del strings[0] # empty string for first element
    for s in strings:
      lines = s.split('\n')
      chem_comp_strings[lines[0]] = 'data_' + s
  return chem_comp_strings


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
      elif order == 4:
        mw.AddBond(i, j, Chem.BondType.AROMATIC)
      else:
        mw.AddBond(i, j, Chem.BondType.SINGLE)
    Chem.SanitizeMol(mw)

    if Chem.GetFormalCharge(mw) != charge:
      raise ValueError(f"mol charge {Chem.GetFormalCharge(mw)}!={charge}")

    return mw


def chemcomp2graph(chem_comp_dir: str, chem_comp_id: str) -> Mapping[str, Any]:
  """Convert chem_comp to graph dict."""
  try:
    # read chem_comp file
    chem_comp_path = Path(chem_comp_dir) / f"{chem_comp_id}.cif"
    assert chem_comp_path.is_file(), f"Invalid file {chem_comp_path}."
    chem_comp_string = parse_mmcif_string(str(chem_comp_path))
    assert chem_comp_string, f"Failed to read mmcif string for {chem_comp_id}."
    # process ideal chemical component
    handle = io.StringIO(chem_comp_string)
    mmcifdict = MMCIF2Dict.MMCIF2Dict(handle)
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
    # convert atom symbols and bond orders to mol by using RDKit
    rdkitmol = create_rdkitmol(symbols, pdbx_formal_charge, orders, charges)
    # RDKit generate conformers for molecules using ETKDGv3 method
    if len(atomids) > 1:
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
    logging.error("Failed to convert %s to graph, %s", chem_comp_id, e)
    return {}


def process_nonpoly_residue(residue: ResidueAtPosition,
                            chem_comp_graph: dict) -> Mapping[str, Any]:
  """Process one non-polymer residue."""
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
  """Remove hydrogens from the graph."""
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


def remove_short_polymer_chains(chains: Mapping[str, MmcifChain],
                                cutoff: int = 4,
                                ) -> Mapping[str, MmcifChain]:
  """Remove short polymer chains from the parsed mmcif chains.

  AlphaFold3 supplementary information section 2.5.4:
  Filtering of targets:
  - Any polymer chain containing fewer than 4 resolved residues is filtered out.
  """
  deleted_keys = set()
  for k, c in chains.items():
    if c.entity_type == 'polymer':
      if len(c.residues) < cutoff:
        logging.debug("Chain %s less than %d resolved residues.", k, cutoff)
        deleted_keys.add(k)

  return {k: v for k, v in chains.items() if k not in deleted_keys}


def remove_clashing_chains(chains: Mapping[str, MmcifChain],
                           cutoff: float = 1.7,
                           percent: float = 0.3,
                           ) -> Mapping[str, Any]:
  """Remove clashing chains from the parsed mmcif chains.

  AlphaFold3 supplementary information section 2.5.4:
  Filtering of bioassemblies:
  - Clashing chains are removed. Clashing chains are defined as those with >30%
    of atoms within 1.7 Å of an atom in another chain. If two chains are
    clashing with each other, the chain with the greater percentage of clashing
    atoms will be removed. If the same fraction of atoms are clashing, the chain
    with fewer total atoms is removed. If the chains have the same number of
    atoms, then the chain with the larger chain id is removed.
  """
  sorted_chains = sorted(chains.items(), key=lambda x: x[0])
  deleted_keys = set()
  start_time = time.time()
  for i, (k, c) in enumerate(sorted_chains):
    if k in deleted_keys:
      # skip the clashing chains
      continue
    pos = np.array([[a.x, a.y, a.z] for r in c.residues for a in r.atoms])
    tmp_keys = set()
    for j, (k2, c2) in enumerate(sorted_chains[i+1:], i+1):
      if k2 in deleted_keys:
        # skip the clashing chains
        continue
      pos2 = np.array([[a.x, a.y, a.z] for r in c2.residues for a in r.atoms])
      dist = np.linalg.norm(pos[:, None] - pos2[None, :], axis=-1)
      r = 1.0 * np.sum(np.min(dist, axis=1) < cutoff) / len(pos)
      r2 = 1.0 * np.sum(np.min(dist, axis=0) < cutoff) / len(pos2)
      if r > percent or r2 > percent:
        if r2 >= r:
          tmp_keys.add(k2)
        else:
          deleted_keys.add(k)
          tmp_keys.clear()
          logging.debug("Chain %s is clashing with %s", k, k2)
          break
      if time.time() - start_time > 7200:
        raise ValueError("Chains processing takes more than 2 hour, aborted")
    if tmp_keys:
      deleted_keys.update(tmp_keys)
      logging.debug("Chains %s are clashing with %s", tmp_keys, k)

  return {k: v for k, v in chains.items() if k not in deleted_keys}


def filter_nonconsecutive_chains(chains: Mapping[str, MmcifChain],
                                 ca_cutoff: float = 10.0,
                                 ) -> Mapping[str, MmcifChain]:
  """Remove non-consecutive chains from the parsed mmcif chains.

  AlphaFold3 supplementary information section 2.5.4:
  Filtering of bioassemblies:
  - Protein chains with consecutive CA atoms >10 angstrom apart are filtered out.
  """
  deleted_keys = set()
  for k, c in chains.items():
    if c.entity_type == 'polymer':
      pos = []
      for r in c.residues:
        coord = [np.nan, np.nan, np.nan]
        for a in r.atoms:
          if a.name == 'CA':
            coord = [a.x, a.y, a.z]
            break
        pos.append(coord)
      dist = np.sqrt(np.sum(np.diff(pos, axis=0)**2, axis=1))
      if np.any(dist > ca_cutoff):
        deleted_keys.add(k)
        logging.debug("Chain %s has non-consecutive CA atoms.", k)

  return {k: v for k, v in chains.items() if k not in deleted_keys}


def process_one_structure(chem_comp_dir: str,
                          mmcif_path: str,
                          ) -> Mapping[str, Union[str, dict, list]]:
  """Parse a mmCIF file and convert data to list of dict."""
  try:
    logging.info("File %s processed by JIANWZHU START.", mmcif_path)

    chem_comp_dir = str(Path(chem_comp_dir).resolve())
    assert Path(chem_comp_dir).is_dir(), f"Invalid directory {chem_comp_dir}."

    # Parse mmcif file by modified AlphaFold mmcif_parsing.py
    mmcif_path = str(Path(mmcif_path).resolve())
    pdbid = Path(mmcif_path).name.split('.')[0].split('-assembly')[0]
    assert len(pdbid) == 4, f"Invalid 4 characters PDBID {pdbid}."
    mmcif_string = parse_mmcif_string(mmcif_path)
    assert mmcif_string, f"Failed to read mmcif string for {pdbid}."
    result = parse_mmcif.parse(file_id=pdbid, mmcif_string=mmcif_string)
    assert result.mmcif_object, f"The errors are {result.errors}"

    # Filter short polymer chains as AlphaFold3.
    full_chains = result.mmcif_object.chains
    full_chains = remove_short_polymer_chains(full_chains)

    # Check the number of polymer chains as AlphaFold3.
    # Filtering of targets:
    # - The maximum number of polymer chains in a considered structure is 300
    #   for training and 1000 for evaluation.
    num_poly = sum(
      [1 if c.entity_type == 'polymer' else 0 for _, c in full_chains.items()])
    assert num_poly <= 300, f"{num_poly} polymer chains, {pdbid}."

    # Remove clashing chains as AlphaFold3.
    clean_chains = remove_clashing_chains(full_chains)
    if clean_chains.keys() != full_chains.keys():
      logging.warning("Chains %s are clashing and removed, %s.",
                      full_chains.keys()-clean_chains.keys(), pdbid)
    full_chains = clean_chains
    del clean_chains

    # Filter non-consecutive chains as AlphaFold3.
    filtered_chains = filter_nonconsecutive_chains(full_chains)
    if filtered_chains.keys() != full_chains.keys():
      logging.warning("Chains %s are non-consecutive and removed, %s.",
                      full_chains.keys()-filtered_chains.keys(), pdbid)
    full_chains = filtered_chains
    del filtered_chains

    chem_comp_graphs = {}
    polymer_chains = {}
    nonpoly_graphs = []
    for chain_id, chain in full_chains.items():
      # print('-'*80, f'{chain_id} {chain.type} {len(chain.seqres)}', sep='\n')
      seqres, residues = chain.seqres, chain.residues
      if chain.entity_type == 'polymer':
        aatype = chain_type_to_one_letter(chain.type)
        if aatype != 'p':
          # Exclude DNA and RNA chains
          # Exclude unknown type, e.g., 'other', 'peptide nucleic acid',
          # and 'polydeoxyribonucleotide/polyribonucleotide hybrid'
          logging.warning("Remove chain type %s, %s.", chain.type, pdbid)
          continue
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
      elif chain.entity_type == 'non-polymer':
        for r in residues:
          if r.name in SKIPCCD or not r.atoms:
            # Skip residue in exclusion list or has no atom
            continue
          if r.name not in chem_comp_graphs:
            chem_comp_graphs[r.name] = chemcomp2graph(chem_comp_dir, r.name)
          graph = process_nonpoly_residue(r, chem_comp_graphs[r.name])
          if not graph or np.all(np.isnan(graph['node_coord'])):
            logging.warning("Residue %s has no coord, %s.", r.name, pdbid)
          else:
            nonpoly_graphs.append(graph)
      else:
        logging.warning("Unknown entity type %s, %s. ", chain.entity_type, pdbid)
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

    logging.info("File %s processed by JIANWZHU SUCCESS.", mmcif_path)
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
@click.option("--chem-comp-dir",
              type=click.Path(exists=True),
              required=True,
              help="Input directory contains of all chemical components.")
@click.option("--mmcif-path",
              type=click.Path(exists=True),
              required=True,
              help="Input path of one mmCIF file rsync from RCSB.")
def process_one(chem_comp_dir: str, mmcif_path: str) -> None:
  """Process one mmCIF file and print the result."""
  chem_comp_dir = str(Path(chem_comp_dir).resolve())
  mmcif_path = str(Path(mmcif_path).resolve())
  data = process_one_structure(chem_comp_dir, mmcif_path)
  show_one_structure(data)


@cli.command()
@click.option("--chem-comp-dir",
              type=click.Path(exists=True),
              required=True,
              help="Input directory contains of all chemical components.")
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
              default="PDB snapshot from rsync://rsync.wwpdb.org::ftp/data/ with --port=33444 on 20240630.",
              help="Comments for output.")
def process(chem_comp_dir: str,
            mmcif_dir: str,
            output_lmdb: str,
            num_workers: int,
            data_comment: str,
            ) -> None:
  """Process mmCIF files from directory and save to lmdb."""
  mmcif_dir = str(Path(mmcif_dir).resolve())
  mmcif_paths = {_.name.split('.')[0]:str(_)
                 for _ in Path(mmcif_dir).rglob("*.cif.gz")}
  assert mmcif_paths and all(4==len(_) for _ in mmcif_paths), (
    f"PDBID should be 4 characters long in {mmcif_dir}.")
  logging.info("There are %d structures in %s.", len(mmcif_paths), mmcif_dir)

  pdbids = list(mmcif_paths.keys())
  logging.info("There are %d pdbids for processing.", len(pdbids))

  chem_comp_dir = str(Path(chem_comp_dir).resolve())
  logging.info("Chemical components information is in %s.", chem_comp_dir)

  output_lmdb = str(Path(output_lmdb).resolve())
  assert not Path(output_lmdb).exists(), f"ERROR: {output_lmdb} exists. Stop."
  logging.info("Will save processed data to %s.", output_lmdb)

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
      f'Output lmdb: {output_lmdb}\n'
      f'Number of workers: {num_workers}\n'
      f'Comments: {data_comment}\n'
    ),
  }

  pbar = tqdm(total=len(pdbids)//10000+1, desc='Processing chunks (10k)')
  for pdbid_chunk in chunks(pdbids, 10000):
    result_chunk = joblib.Parallel(n_jobs=num_workers)(
      joblib.delayed(process_one_structure)(chem_comp_dir, mmcif_paths[_])
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
  """Check the consistency of chemical components and ideal SDF files."""
  chem_comp_path = Path(chem_comp_path).resolve()
  logging.info("Chemical components information is in %s.", chem_comp_path)

  sdf_dir = Path(sdf_dir).resolve()
  logging.info("The ideal SDF files in %s.", sdf_dir)

  chem_comp_path = Path(chem_comp_path).resolve()
  chem_comp_full_string = parse_mmcif_string(str(chem_comp_path))
  assert chem_comp_full_string.startswith('data_'), (
    f"Failed to read chem_comp from {chem_comp_path}.")

  chem_comp_strings = split_chem_comp_full_string(chem_comp_full_string)
  assert chem_comp_strings, f"Failed to split {chem_comp_path}."
  logging.info("%d chem_comp in %s.", len(chem_comp_strings), chem_comp_path)

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
      logging.error("Check %s failed, %s.", name, e)
      return -1

  logging.info("Checking %d chem_comp one by one ...", len(chem_comp_strings))
  names = sorted(chem_comp_strings.keys())
  results = [(_check_one(_), _) for _ in tqdm(names)]
  STATUS = {1: 'EQUAL', 0: 'DIFFERENT', -1: 'FAILED', -2: 'SKIP'}
  for r in results:
    print(STATUS[r[0]], r[1])


@cli.command()
@click.option("--mmcif-dir",
              type=click.Path(exists=True),
              required=True,
              help="Input directory of mmCIF files rsync from RCSB.")
@click.option("--num-workers",
              type=int,
              default=-1,
              help="Number of workers.")
def check_files(mmcif_dir: str, num_workers: int) -> None:
  """Check mmCIF files from directory."""
  mmcif_dir = str(Path(mmcif_dir).resolve())
  mmcif_paths = {_.name.split('.')[0].split('-assembly')[0]:_
                 for _ in Path(mmcif_dir).rglob("*-assembly1.cif.gz")}
  assert mmcif_paths and all(4==len(_) for _ in mmcif_paths), (
    f"PDBID should be 4 characters long in {mmcif_dir}.")
  logging.info(f"There are %d structures in %s.", len(mmcif_paths), mmcif_dir)

  pdbids = list(mmcif_paths.keys())
  logging.info("There are %d pdbids for processing.", len(pdbids))

  def _check_one(mmcif_path: str) -> Tuple[str, bool]:
    logging.set_verbosity(logging.INFO)
    start_time = time.time()
    try:
      logging.info("File %s processed by JIANWZHU START.", mmcif_path)
      mmcif_path = str(Path(mmcif_path).resolve())
      pdbid = Path(mmcif_path).name.split('.')[0].split('-assembly')[0]
      assert len(pdbid) == 4, f"Invalid 4 characters PDBID {pdbid}."
      mmcif_string = parse_mmcif_string(mmcif_path)
      assert mmcif_string, f"Failed to read mmcif string for {pdbid}."
      result = parse_mmcif.parse(file_id=pdbid, mmcif_string=mmcif_string)
      assert result.mmcif_object, f"The errors are {result.errors}"
      logging.info("File %s processed by JIANWZHU SUCCESS.", mmcif_path)
      status = True
    except Exception as e:
      logging.error("File %s processed by JIANWZHU FAILED, %s.", mmcif_path, e)
      status = False
    elapsed_time = time.time() - start_time
    logging.info("File %s processed in %.3f seconds.", mmcif_path, elapsed_time)
    return pdbid, status, elapsed_time

  results = joblib.Parallel(n_jobs=num_workers)(
    joblib.delayed(_check_one)(mmcif_paths[_])
    for _ in tqdm(pdbids, desc='Checking files')
  )
  for name, status, elapsed_time in results:
    print(name, status, elapsed_time)


if __name__ == "__main__":
  cli()
