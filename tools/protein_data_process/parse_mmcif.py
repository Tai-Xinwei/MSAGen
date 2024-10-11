# -*- coding: utf-8 -*-
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parses the mmCIF file format."""
import collections
import dataclasses
import functools
import io
from typing import Any, Mapping, Optional, Sequence, Tuple

from absl import logging
from Bio import PDB
from Bio.Data import PDBData


# Type aliases:
ChainId = str
PdbHeader = Mapping[str, Any]
PdbStructure = PDB.Structure.Structure
SeqRes = str
MmCIFDict = Mapping[str, Sequence[str]]


@dataclasses.dataclass(frozen=True)
class Monomer:
  id: str
  num: int


# Note - mmCIF format provides no guarantees on the type of author-assigned
# sequence numbers. They need not be integers.
@dataclasses.dataclass(frozen=True)
class AtomSite:
  residue_name: str
  author_chain_id: str
  mmcif_chain_id: str
  author_seq_num: str
  mmcif_seq_num: str
  insertion_code: str
  hetatm_atom: str
  model_num: str
  name: str
  type: str
  x: float
  y: float
  z: float
  altloc: str
  occupancy: float
  tempfactor: float


# Used to map SEQRES index to a residue in the structure.
@dataclasses.dataclass(frozen=True)
class ResiduePosition:
  chain_id: str
  residue_number: int
  insertion_code: str


@dataclasses.dataclass(frozen=True)
class ResidueAtPosition:
  position: Optional[ResiduePosition]
  name: str
  type: str
  is_missing: bool
  hetflag: str
  atoms: Optional[Sequence[AtomSite]]


@dataclasses.dataclass(frozen=True)
class MmcifChain:
  mmcif_chain_id: str
  author_chain_id: str
  entity_id: str
  entity_type: str
  type: str
  seqres: str
  residues: Optional[Sequence[ResidueAtPosition]]


@dataclasses.dataclass(frozen=True)
class MmcifObject:
  """Representation of a parsed mmCIF file.

  Contains:
    file_id: A meaningful name, e.g. a pdb_id. Should be unique amongst all
      files being processed.
    header: Biopython header.
    structure: Biopython structure.
    chains: Dict; each chain_id indicate a MmcifChain contains a mapping between
      SEQRES index and a ResidueAtPosition. e.g. {'A': {1: ResidueAtPosition,
                                                        2: ResidueAtPosition,
                                                        ...}}
    raw_string: The raw string used to construct the MmcifObject.
  """
  file_id: str
  header: PdbHeader
  structure: PdbStructure
  chains: Mapping[ChainId, MmcifChain]
  raw_string: Any


@dataclasses.dataclass(frozen=True)
class ParsingResult:
  """Returned by the parse function.

  Contains:
    mmcif_object: A MmcifObject, may be None if no chain could be successfully
      parsed.
    errors: A dict mapping (file_id, chain_id) to any exception generated.
  """
  mmcif_object: Optional[MmcifObject]
  errors: Mapping[Tuple[str, str], Any]


class ParseError(Exception):
  """An error indicating that an mmCIF file could not be parsed."""


def mmcif_loop_to_list(prefix: str,
                       parsed_info: MmCIFDict) -> Sequence[Mapping[str, str]]:
  """Extracts loop associated with a prefix from mmCIF data as a list.

  Reference for loop_ in mmCIF:
    http://mmcif.wwpdb.org/docs/tutorials/mechanics/pdbx-mmcif-syntax.html

  Args:
    prefix: Prefix shared by each of the data items in the loop.
      e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
      _entity_poly_seq.mon_id. Should include the trailing period.
    parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
      parser.

  Returns:
    Returns a list of dicts; each dict represents 1 entry from an mmCIF loop.
  """
  cols = []
  data = []
  for key, value in parsed_info.items():
    if key.startswith(prefix):
      cols.append(key)
      data.append(value)

  assert all([len(xs) == len(data[0]) for xs in data]), (
      'mmCIF error: Not all loops are the same length: %s' % cols)

  return [dict(zip(cols, xs)) for xs in zip(*data)]


def mmcif_loop_to_dict(prefix: str,
                       index: str,
                       parsed_info: MmCIFDict,
                       ) -> Mapping[str, Mapping[str, str]]:
  """Extracts loop associated with a prefix from mmCIF data as a dictionary.

  Args:
    prefix: Prefix shared by each of the data items in the loop.
      e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
      _entity_poly_seq.mon_id. Should include the trailing period.
    index: Which item of loop data should serve as the key.
    parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
      parser.

  Returns:
    Returns a dict of dicts; each dict represents 1 entry from an mmCIF loop,
    indexed by the index column.
  """
  entries = mmcif_loop_to_list(prefix, parsed_info)
  return {entry[index]: entry for entry in entries}


@functools.lru_cache(16, typed=False)
def parse(*,
          file_id: str,
          mmcif_string: str,
          catch_all_errors: bool = True) -> ParsingResult:
  """Entry point, parses an mmcif_string.

  Args:
    file_id: A string identifier for this file. Should be unique within the
      collection of files being processed.
    mmcif_string: Contents of an mmCIF file.
    catch_all_errors: If True, all exceptions are caught and error messages are
      returned as part of the ParsingResult. If False exceptions will be allowed
      to propagate.

  Returns:
    A ParsingResult.
  """
  errors = {}
  try:
    parser = PDB.MMCIFParser(QUIET=True)
    handle = io.StringIO(mmcif_string)
    full_structure = parser.get_structure('', handle)
    first_model_structure = _get_first_model(full_structure)
    # Extract the _mmcif_dict from the parser, which contains useful fields not
    # reflected in the Biopython structure.
    parsed_info = parser._mmcif_dict  # pylint:disable=protected-access

    # Ensure all values are lists, even if singletons.
    for key, value in parsed_info.items():
      if not isinstance(value, list):
        parsed_info[key] = [value]

    header = _get_header(parsed_info)

    # Determine the valid mmcif chains, and their start numbers according to the
    # internal mmCIF numbering scheme (likely but not guaranteed to be 1).
    valid_chains = _get_valid_chains(parsed_info=parsed_info)

    # Loop over the atoms for which we have coordinates, and their start numbers
    # according to the internal mmCIF numbering scheme.
    valid_atoms = collections.defaultdict(dict)
    for atom in _get_atom_site_list(parsed_info):
      if atom.model_num != '1':
        # We only process the first model at the moment.
        continue
      if atom.mmcif_chain_id not in valid_chains:
        # We only process chains that are valid.
        logging.warning('Chain %s not valid: %s', atom.mmcif_chain_id, file_id)
        continue
      if atom.mmcif_seq_num == '.':
        idx = int(atom.author_seq_num)
      else:
        idx = int(atom.mmcif_seq_num)
      current = valid_atoms[atom.mmcif_chain_id].get(idx, [])
      current.append(atom)
      valid_atoms[atom.mmcif_chain_id][idx] = current

    # AlphaFold3 supplementary information section 2.1
    # - Keep alternative locations with the largest occupancy.
    for chain_id, chain_data in valid_atoms.items():
      for idx, atoms in chain_data.items():
        if any([_.altloc != '.' for _ in atoms]):
          chain_data[idx] = _select_atoms(atoms)

    # Get chemical component information.
    chem_comps = mmcif_loop_to_dict('_chem_comp.', '_chem_comp.id', parsed_info)

    # Loop over the atoms for which we have coordinates. Populate one mapping:
    # -seq_to_structure_mappings: (maps idx into sequence to ResidueAtPosition)
    seq_to_structure_mappings = collections.defaultdict(dict)
    for chain_id, chain_data in valid_atoms.items():
      for idx, atoms in chain_data.items():
        if atoms and _is_same(atoms):
          atom = atoms[0]
          hetflag = ' '
          if atom.hetatm_atom == 'HETATM':
            # Water atoms are assigned a special hetflag of W in Biopython. We
            # need to do the same, so that this hetflag can be used to fetch
            # a residue from the Biopython structure by id.
            if atom.residue_name in ('HOH', 'WAT'):
              hetflag = 'W'
            else:
              hetflag = 'H_' + atom.residue_name
          insertion_code = atom.insertion_code
          if not _is_set(atom.insertion_code):
            insertion_code = ' '
          residue_type = chem_comps[atom.residue_name]['_chem_comp.type']
          position = ResiduePosition(chain_id=atom.author_chain_id,
                                     residue_number=int(atom.author_seq_num),
                                     insertion_code=insertion_code)
          current = ResidueAtPosition(position=position,
                                      name=atom.residue_name,
                                      type=residue_type,
                                      is_missing=False,
                                      hetflag=hetflag,
                                      atoms=atoms)
          seq_to_structure_mappings[chain_id][idx] = current
        else:
          logging.warning('Chain %s resiude %s may have wrong atoms: %s',
                          chain_id, idx, file_id)

    # Add missing residue information to seq_to_structure_mappings.
    for chain_id, mapping in seq_to_structure_mappings.items():
      for mon in valid_chains[chain_id]:
        if mon.num not in mapping:
          residue_type = chem_comps[mon.id]['_chem_comp.type']
          mapping[mon.num] = ResidueAtPosition(position=None,
                                               name=mon.id,
                                               type=residue_type,
                                               is_missing=True,
                                               hetflag=' ',
                                               atoms=[])

    # AlphaFold3 supplementary information section 2.1
    # - Keep alternative locations with the largest occupancy.
    # - MSE residues are converted to MET residues.
    # - Fix arginine naming ambiguities (ensuring NH1 closer to CD than NH2).
    for chain_id, mapping in seq_to_structure_mappings.items():
      for idx, residue in mapping.items():
        # # Keep alternative locations with the largest occupancy.
        # if any([_.altloc != '.' for _ in residue.atoms]):
        #   _atoms = _select_atoms(residue.atoms)
        #   residue = dataclasses.replace(residue, atoms=_atoms)
        # MSE residues are converted to MET residues.
        if residue.name == 'MSE':
          _atoms = []
          for a in residue.atoms:
            if a.name != 'SE':
              a = dataclasses.replace(a, residue_name='MET')
            else:
              a = dataclasses.replace(a, residue_name='MET', name='SD', type='S')
            _atoms.append(a)
          residue = dataclasses.replace(residue, name='MET', atoms=_atoms)
        # Fix arginine naming ambiguities (ensuring NH1 closer to CD than NH2).
        if residue.name == 'ARG':
          _cd, _nh1, _nh2 = None, None, None
          _atoms = []
          for i, a in enumerate(residue.atoms):
            if a.name == 'CD':
              _cd = i
            elif a.name == 'NH1':
              _nh1 = i
            elif a.name == 'NH2':
              _nh2 = i
            _atoms.append(a)
          if _cd and _nh1 and _nh2:
            cd, nh1, nh2 = _atoms[_cd], _atoms[_nh1], _atoms[_nh2]
            d1 = (cd.x - nh1.x)**2 + (cd.y - nh1.y)**2 + (cd.z - nh1.z)**2
            d2 = (cd.x - nh2.x)**2 + (cd.y - nh2.y)**2 + (cd.z - nh2.z)**2
            if d1 > d2:
              _atoms[_nh1] = dataclasses.replace(nh2, name='NH1')
              _atoms[_nh2] = dataclasses.replace(nh1, name='NH2')
          residue = dataclasses.replace(residue, atoms=_atoms)
        seq_to_structure_mappings[chain_id][idx] = residue

    # Process entity information and mapping mmcif asyms with metadata
    entitys = mmcif_loop_to_dict('_entity.', '_entity.id', parsed_info)

    # Get entity information. Will allow us to identify which of these polymers
    # are proteins, DNAs and RNAs.
    entity_polys = mmcif_loop_to_dict('_entity_poly.',
                                      '_entity_poly.entity_id',
                                      parsed_info)

    # Get chains information for each entity. Necessary so that we can return a
    # dict keyed on chain id rather than entity.
    struct_asyms = mmcif_loop_to_dict('_struct_asym.',
                                      '_struct_asym.id',
                                      parsed_info)

    mmcif_chains = {}
    for chain_id, mapping in seq_to_structure_mappings.items():
      entity_id = struct_asyms[chain_id]['_struct_asym.entity_id']
      entity_type = entitys[entity_id]['_entity.type']
      if entity_type == 'water':
        # AlphaFold3 supplementary section 2.1: waters are removed.
        continue
      if entity_id in entity_polys:
        chain_type = entity_polys[entity_id]['_entity_poly.type']
      else:
        chain_type = entity_type

      sorted_indices = sorted(mapping.keys())
      residues = [mapping[_] for _ in sorted_indices]
      allkeys = set(range(sorted_indices[0], sorted_indices[-1] + 1))
      assert mapping.keys() == allkeys, (
        f'Chain {chain_id} may miss residues {allkeys - mapping.keys()}')

      for r in residues:
        if not r.is_missing:
          author_chain_id = r.position.chain_id
          break
      else:
        raise ValueError(f'No author chain id found for chain {chain_id}')

      if chain_type in ('polypeptide(L)', 'polypeptide(D)'):
        seq = [PDBData.protein_letters_3to1.get(f'{_.name:<3s}', 'X')
               for _ in residues]
      elif chain_type in ('polydeoxyribonucleotide', 'polyribonucleotide'):
        seq = [PDBData.nucleic_letters_3to1.get(f'{_.name:<3s}', 'N')
               for _ in residues]
      else:
        seq = ['?'] * len(residues)
      seqres = ''.join(seq)

      mmcif_chains[chain_id] = MmcifChain(mmcif_chain_id=chain_id,
                                          author_chain_id=author_chain_id,
                                          entity_id=entity_id,
                                          entity_type=entity_type,
                                          type=chain_type,
                                          seqres=seqres,
                                          residues=residues)
    if all([v.entity_type != 'polymer' for _, v in mmcif_chains.items()]):
      return ParsingResult(
        None, {(file_id, ''): 'No polymer chains found in this file.'})

    mmcif_object = MmcifObject(
        file_id=file_id,
        header=header,
        structure=first_model_structure,
        chains=mmcif_chains,
        raw_string=parsed_info)

    return ParsingResult(mmcif_object=mmcif_object, errors=errors)
  except Exception as e:  # pylint:disable=broad-except
    errors[(file_id, '')] = e
    if not catch_all_errors:
      raise
    return ParsingResult(mmcif_object=None, errors=errors)


def _get_first_model(structure: PdbStructure) -> PdbStructure:
  """Returns the first model in a Biopython structure."""
  return next(structure.get_models())


_MIN_LENGTH_OF_CHAIN_TO_BE_COUNTED_AS_PEPTIDE = 21


def get_release_date(parsed_info: MmCIFDict) -> str:
  """Returns the oldest revision date."""
  revision_dates = parsed_info['_pdbx_audit_revision_history.revision_date']
  return min(revision_dates)


def _get_header(parsed_info: MmCIFDict) -> PdbHeader:
  """Returns a basic header containing method, release date and resolution."""
  header = {}

  experiments = mmcif_loop_to_list('_exptl.', parsed_info)
  header['structure_method'] = ','.join([
      experiment['_exptl.method'].lower() for experiment in experiments])

  # Note: The release_date here corresponds to the oldest revision. We prefer to
  # use this for dataset filtering over the deposition_date.
  if '_pdbx_audit_revision_history.revision_date' in parsed_info:
    header['release_date'] = get_release_date(parsed_info)
  else:
    logging.debug('Could not determine release_date: %s',
                  parsed_info['_entry.id'])

  header['resolution'] = 0.00
  for res_key in ('_refine.ls_d_res_high', '_em_3d_reconstruction.resolution',
                  '_reflns.d_resolution_high'):
    if res_key in parsed_info:
      try:
        raw_resolution = parsed_info[res_key][0]
        header['resolution'] = float(raw_resolution)
        break
      except ValueError:
        logging.debug('Invalid resolution format: %s', parsed_info[res_key])

  return header


def _get_atom_site_list(parsed_info: MmCIFDict) -> Sequence[AtomSite]:
  """Returns list of atom sites; contains data not present in the structure."""
  atoms = []
  for site in zip( # pylint:disable=g-complex-comprehension
    parsed_info['_atom_site.label_comp_id'],      # residue_name: str
    parsed_info['_atom_site.auth_asym_id'],       # author_chain_id: str
    parsed_info['_atom_site.label_asym_id'],      # mmcif_chain_id: str
    parsed_info['_atom_site.auth_seq_id'],        # author_seq_num: str
    parsed_info['_atom_site.label_seq_id'],       # mmcif_seq_num: str
    parsed_info['_atom_site.pdbx_PDB_ins_code'],  # insertion_code: str
    parsed_info['_atom_site.group_PDB'],          # hetatm_atom: str
    parsed_info['_atom_site.pdbx_PDB_model_num'], # model_num: str
    parsed_info['_atom_site.label_atom_id'],      # name: str
    parsed_info['_atom_site.type_symbol'],        # type: str
    parsed_info['_atom_site.Cartn_x'],            # x: float
    parsed_info['_atom_site.Cartn_y'],            # y: float
    parsed_info['_atom_site.Cartn_z'],            # z: float
    parsed_info['_atom_site.label_alt_id'],       # altloc: str
    parsed_info['_atom_site.occupancy'],          # occupancy: float
    parsed_info['_atom_site.B_iso_or_equiv'],     # tempfactor: float
    ):
    atoms.append(AtomSite(
      residue_name=site[0],
      author_chain_id=site[1],
      mmcif_chain_id=site[2],
      author_seq_num=site[3],
      mmcif_seq_num=site[4],
      insertion_code=site[5],
      hetatm_atom=site[6],
      model_num=site[7],
      name=site[8],
      type=site[9],
      x=float(site[10]),
      y=float(site[11]),
      z=float(site[12]),
      altloc=site[13],
      occupancy=float(site[14]),
      tempfactor=float(site[15]),
      ))
  return atoms


def _get_valid_chains(
    *, parsed_info: Mapping[str, Any]) -> Mapping[ChainId, Sequence[Monomer]]:
  """Extracts valid chains information for mmcif chains.

  Args:
    parsed_info: _mmcif_dict produced by the Biopython parser.

  Returns:
    A dict mapping mmcif chain id to a list of Monomers.
  """
  # Get sequence information for each entity in the structure.
  valid_chains = collections.defaultdict(list)

  # Get pdbx_poly_seq_scheme information for each entity in the structure.
  for poly_seq in mmcif_loop_to_list('_pdbx_poly_seq_scheme.', parsed_info):
    asym_id = poly_seq['_pdbx_poly_seq_scheme.asym_id']
    mon_id = poly_seq['_pdbx_poly_seq_scheme.mon_id']
    seq_num = int(poly_seq['_pdbx_poly_seq_scheme.seq_id'])
    valid_chains[asym_id].append(Monomer(id=mon_id, num=seq_num))

  # Get pdbx_branch_scheme information for each entity in the structure.
  for branch in mmcif_loop_to_list('_pdbx_branch_scheme.', parsed_info):
    asym_id = branch['_pdbx_branch_scheme.asym_id']
    mon_id = branch['_pdbx_branch_scheme.mon_id']
    seq_num = int(branch['_pdbx_branch_scheme.pdb_seq_num'])
    valid_chains[asym_id].append(Monomer(id=mon_id, num=seq_num))

  # Get pdbx_nonpoly_scheme information for each entity in the structure.
  for nonpoly in mmcif_loop_to_list('_pdbx_nonpoly_scheme.', parsed_info):
    asym_id = nonpoly['_pdbx_nonpoly_scheme.asym_id']
    mon_id = nonpoly['_pdbx_nonpoly_scheme.mon_id']
    seq_num = int(nonpoly['_pdbx_nonpoly_scheme.pdb_seq_num'])
    valid_chains[asym_id].append(Monomer(id=mon_id, num=seq_num))

  return valid_chains


def _select_atoms(atoms: Sequence[AtomSite]) -> Sequence[AtomSite]:
  """Selects atoms as AlphaFold3 supplementary information section 2.1:
     Alternative locations for atoms/residues are resolved by taking the one
     with the largest occupancy.

  Args:
    atoms: all atoms in mmcif file for one residue.

  Returns:
    A list contains selected atoms with largets occupancy.
  """
  group_atoms = collections.defaultdict(list)
  for atom in atoms:
      group_atoms[atom.altloc].append(atom)
  common_atoms = group_atoms.pop('.', [])
  max_key, max_occupancy = None, -1.0
  for alt_id, group in group_atoms.items():
    current_occupancy = sum([_.occupancy for _ in group]) / len(group)
    if current_occupancy > max_occupancy:
      max_key, max_occupancy = alt_id, current_occupancy
  return group_atoms[max_key] + common_atoms


def _is_same(atoms: Sequence[AtomSite]) -> bool:
  """Check if all atoms are in same residue or not."""
  if not atoms:
    return True
  refa = atoms[0]
  for a in atoms[1:]:
    if not (a.residue_name == refa.residue_name and
            a.mmcif_chain_id == refa.mmcif_chain_id and
            a.mmcif_seq_num == refa.mmcif_seq_num and
            a.author_chain_id == refa.author_chain_id and
            a.author_seq_num == refa.author_seq_num and
            a.insertion_code == refa.insertion_code):
      return False
  return True


def _is_set(data: str) -> bool:
  """Returns False if data is a special mmCIF character indicating 'unset'."""
  return data not in ('.', '?')


if __name__ == '__main__':
  import gzip
  import sys
  from pathlib import Path

  if len(sys.argv) != 2:
    sys.exit(f'Usage: {sys.argv[0]} <input_mmcif_path, e.g., 1ctf.cif(.gz)>')
  inppath = Path(sys.argv[1])

  if inppath.name.endswith('.cif'):
    with open(inppath, 'r') as fp:
      mmcif_string = fp.read()
  elif inppath.name.endswith('.cif.gz'):
    with gzip.open(inppath, 'rt') as fp:
        mmcif_string = fp.read()
  else:
    sys.exit(f'Input mmCIF file should be *.cif or *.cif.gz, {inppath}')

  file_id = inppath.name.split('.')[0].split('-assembly')[0]
  result = parse(file_id=file_id, mmcif_string=mmcif_string)
  assert result.mmcif_object, f'Parsing failed for {inppath}, {result.errors}'

  # show mmcif parsing result
  print(file_id)
  print(result.mmcif_object.header)
  print(result.mmcif_object.structure)
  print(result.mmcif_object.chains.keys())
  print(type(result.mmcif_object.raw_string))

  print('-'*80)
  print('chains[id]:', MmcifChain)
  for kc, vc in MmcifChain.__annotations__.items():
    print(f'  {kc}: {vc}')
    if kc == 'residues':
      for kr, vr in ResidueAtPosition.__annotations__.items():
        print(f'    {kr}: {vr}')
        if kr == 'position':
          for kp, vp in ResiduePosition.__annotations__.items():
            print(f'       {kp}: {vp}')
        if kr == 'atoms':
          for ka, va in AtomSite.__annotations__.items():
            print(f'       {ka}: {va}')

  sorted_chains = sorted(result.mmcif_object.chains.items(), key=lambda x: x[0])
  for chain_id, chain in sorted_chains:
    print('-'*80)
    print(f'mmcif_asym={chain.mmcif_chain_id} entity_id={chain.entity_id} '
          f'author_chain={chain.author_chain_id} {chain.entity_type} '
          f'{chain.type} {len(chain.residues)} reisudes')
    print(chain.seqres)
    for idx, residue in enumerate(chain.residues):
      if idx < 5 or idx >= len(chain.residues) - 5:
        print(f'{idx:<4d} {residue.name:<3s} {len(residue.atoms):3d} atoms',
              residue.position, residue.is_missing)
