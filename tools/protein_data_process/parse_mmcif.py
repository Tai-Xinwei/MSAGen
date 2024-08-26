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
  mmcif_seq_num: int
  insertion_code: str
  hetatm_atom: str
  model_num: int
  name: str
  type: str
  x: float
  y: float
  z: float
  mmcif_alt_id: str
  occupancy: float

@dataclasses.dataclass(frozen=True)
class AtomCartn:
  name: str
  type: str
  x: float
  y: float
  z: float
  mmcif_alt_id: str
  occupancy: float


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
  atoms: Optional[Sequence[AtomCartn]] = None


@dataclasses.dataclass(frozen=True)
class PolyChain:
  entity_id: str
  orig_entity_id: str
  type: str
  pdbx_seq: str
  pdbx_can: str
  mmcif_chain_id: str
  orig_mmcif_chain_id: str
  author_chain_id: str
  orig_author_chain_id: str
  monomers: Optional[Sequence[Monomer]] = None
  residues: Optional[Sequence[ResidueAtPosition]] = None


@dataclasses.dataclass(frozen=True)
class MmcifObject:
  """Representation of a parsed mmCIF file.

  Contains:
    file_id: A meaningful name, e.g. a pdb_id. Should be unique amongst all
      files being processed.
    header: Biopython header.
    structure: Biopython structure.
    chain_to_seqres: Dict mapping chain_id to 1 letter amino acid sequence. E.g.
      {'A': 'ABCDEFG'}
    seqres_to_structure: Dict; for each chain_id contains a mapping between
      SEQRES index and a ResidueAtPosition. e.g. {'A': {0: ResidueAtPosition,
                                                        1: ResidueAtPosition,
                                                        ...}}
    raw_string: The raw string used to construct the MmcifObject.
  """
  file_id: str
  header: PdbHeader
  structure: PdbStructure
  polymer_chains: Mapping[ChainId, PolyChain]
  nonpoly_chains: Mapping[ChainId, Sequence[ResidueAtPosition]]
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

    # Get chemical component information.
    chem_comps = mmcif_loop_to_dict('_chem_comp.', '_chem_comp.id', parsed_info)

    # Determine the polymer chains, and their start numbers according to the
    # internal mmCIF numbering scheme (likely but not guaranteed to be 1).
    valid_chains = _get_polymer_chains(parsed_info=parsed_info)
    if not valid_chains:
      return ParsingResult(
          None, {(file_id, ''): 'No polymer chains found in this file.'})
    if not all([len(seq.monomers) == len(seq.pdbx_can.replace('\n', ''))
                for _, seq in valid_chains.items()]):
      return ParsingResult(
          None, {(file_id, ''): 'Mismatch for entity polymer sequence.'})
    seq_start_num = {chain_id: min([monomer.num for monomer in seq.monomers])
                     for chain_id, seq in valid_chains.items()}

    # Loop over the atoms for which we have coordinates. Populate two mappings:
    # -mmcif_to_author_chain_id (maps internal mmCIF chain ids to chain ids used
    # the authors / Biopython).
    # -seq_to_structure_mappings (maps idx into sequence to ResidueAtPosition).
    # -seq_to_structure_coords (maps idx into sequence to a list of AtomCartn).
    # -nonpoly_structure_mappings (maps ResiduePosition to ResidueAtPosition).
    # -nonpoly_structure_coords (maps ResiduePosition to a list of AtomCartn).
    mmcif_to_author_chain_id = {}
    seq_to_structure_mappings = {}
    seq_to_structure_coords = collections.defaultdict(dict)
    nonpoly_structure_mappings = {}
    nonpoly_structure_coords = collections.defaultdict(dict)
    for atom in _get_atom_site_list(parsed_info):
      if atom.model_num != '1':
        # We only process the first model at the moment.
        continue

      mmcif_to_author_chain_id[atom.mmcif_chain_id] = atom.author_chain_id

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

      restype = chem_comps[atom.residue_name]['_chem_comp.type']

      position = ResiduePosition(chain_id=atom.author_chain_id,
                                 residue_number=int(atom.author_seq_num),
                                 insertion_code=insertion_code)
      current_residue = ResidueAtPosition(position=position,
                                          name=atom.residue_name,
                                          type=restype,
                                          is_missing=False,
                                          hetflag=hetflag,
                                          atoms=None)
      current_atom = AtomCartn(name=atom.name,
                               type=atom.type,
                               x=float(atom.x),
                               y=float(atom.y),
                               z=float(atom.z),
                               mmcif_alt_id=atom.mmcif_alt_id,
                               occupancy=float(atom.occupancy))

      if atom.mmcif_chain_id in valid_chains: # Polymer residue and atoms.
        seq_idx = int(atom.mmcif_seq_num) - seq_start_num[atom.mmcif_chain_id]
        current = seq_to_structure_mappings.get(atom.author_chain_id, {})
        current[seq_idx] = current_residue
        seq_to_structure_mappings[atom.author_chain_id] = current
        current = seq_to_structure_coords[atom.author_chain_id].get(seq_idx, [])
        current.append(current_atom)
        seq_to_structure_coords[atom.author_chain_id][seq_idx] = current
      else: # Non-polymer residue and atoms.
        current = nonpoly_structure_mappings.get(atom.author_chain_id, {})
        current[position] = current_residue
        nonpoly_structure_mappings[atom.author_chain_id] = current
        current = nonpoly_structure_coords[atom.author_chain_id].get(position, [])
        current.append(current_atom)
        nonpoly_structure_coords[atom.author_chain_id][position] = current

    # Merge atoms into the ResidueAtPosition for all residues
    for author_chain, chain_data in seq_to_structure_mappings.items():
      for idx, residue in chain_data.items():
        atoms = seq_to_structure_coords[author_chain].get(idx, [])
        # AlphaFold3 supplementary information section 2.1
        # Keep alternative locations with the largest occupancy.
        seq_to_structure_mappings[author_chain][idx] = dataclasses.replace(
            residue, atoms=_select_atoms(atoms))
    for author_chain, chain_data in nonpoly_structure_mappings.items():
      for pos, residue in chain_data.items():
        atoms = nonpoly_structure_coords[author_chain].get(pos, [])
        # AlphaFold3 supplementary information section 2.1
        # Keep alternative locations with the largest occupancy.
        nonpoly_structure_mappings[author_chain][pos] = dataclasses.replace(
            residue, atoms=_select_atoms(atoms))

    # Add missing residue information to seq_to_structure_mappings.
    for chain_id, seq_info in valid_chains.items():
      if chain_id not in mmcif_to_author_chain_id:
        raise ValueError(f'Coordinates of chain {chain_id} missed in model 1.')
      author_chain = mmcif_to_author_chain_id[chain_id]
      current_mapping = seq_to_structure_mappings[author_chain]
      for idx, monomer in enumerate(seq_info.monomers):
        if idx not in current_mapping:
          restype = chem_comps[monomer.id]['_chem_comp.type']
          current_mapping[idx] = ResidueAtPosition(position=None,
                                                   name=monomer.id,
                                                   type=restype,
                                                   is_missing=True,
                                                   hetflag=' ',
                                                   atoms=[])

    # If parse assembly mmCIF, we should remapping the entity_id and asym_id.
    entity_remapping = {}
    if '_pdbx_entity_remapping.entity_id' in parsed_info:
      entity_remapping = mmcif_loop_to_dict('_pdbx_entity_remapping.',
                                            '_pdbx_entity_remapping.entity_id',
                                            parsed_info)

    # If parse assembly mmCIF, we should remapping the asym_id.
    chain_remapping = {}
    if '_pdbx_chain_remapping.label_asym_id' in parsed_info:
      chain_remapping = mmcif_loop_to_dict('_pdbx_chain_remapping.',
                                           '_pdbx_chain_remapping.label_asym_id',
                                           parsed_info)

    # Extract all polymer and non-polymer chains
    polymer_chains = {}
    for chain_id, seq_info in valid_chains.items():
      author_chain = mmcif_to_author_chain_id[chain_id]
      current_residues = []
      for idx in sorted(seq_to_structure_mappings[author_chain].keys()):
        residue = seq_to_structure_mappings[author_chain][idx]
        if residue.name == 'MSE':
          # AlphaFold3 supplementary information section 2.1
          # MSE residues are converted to MET residues
          atoms = []
          for atom in residue.atoms:
            if atom.name == 'SE':
              atom = dataclasses.replace(atom, name='SD', type='S')
            atoms.append(atom)
          residue = dataclasses.replace(
            residue, name='MET', hetflag=' ', atoms=atoms)
        if residue.name == 'ARG':
          # AlphaFold3 supplementary information section 2.1
          # arginine naming ambiguities are fixed
          # (ensuring NH1 is always closer to CD than NH2)
          _cd = None
          _nh1, idx_nh1 = None, None
          _nh2, idx_nh2 = None, None
          atoms = []
          for i, atom in enumerate(residue.atoms):
            if atom.name == 'CD':
              _cd = atom
            elif atom.name == 'NH1':
              _nh1, idx_nh1 = atom, i
            elif atom.name == 'NH2':
              _nh2, idx_nh2 = atom, i
            atoms.append(atom)
          if _cd and _nh1 and _nh2:
            d1 = (_cd.x - _nh1.x)**2 + (_cd.y - _nh1.y)**2 + (_cd.z - _nh1.z)**2
            d2 = (_cd.x - _nh2.x)**2 + (_cd.y - _nh2.y)**2 + (_cd.z - _nh2.z)**2
            if d1 > d2:
              atoms[idx_nh1] = dataclasses.replace(_nh2, name='NH1')
              atoms[idx_nh2] = dataclasses.replace(_nh1, name='NH2')
          residue = dataclasses.replace(residue, atoms=atoms)
        current_residues.append(residue)

      orig_entity_id = entity_remapping.get(seq_info.entity_id, {}).get(
        '_pdbx_entity_remapping.orig_entity_id', seq_info.entity_id)
      orig_mmcif_chain_id = chain_remapping.get(chain_id, {}).get(
        '_pdbx_chain_remapping.orig_label_asym_id', chain_id)
      orig_author_chain_id = chain_remapping.get(chain_id, {}).get(
        '_pdbx_chain_remapping.orig_auth_asym_id', author_chain)

      polymer_chains[author_chain] = PolyChain(
        entity_id=seq_info.entity_id,
        orig_entity_id=orig_entity_id,
        type=seq_info.type,
        pdbx_seq=seq_info.pdbx_seq,
        pdbx_can=seq_info.pdbx_can,
        mmcif_chain_id=chain_id,
        orig_mmcif_chain_id=orig_mmcif_chain_id,
        author_chain_id=author_chain,
        orig_author_chain_id=orig_author_chain_id,
        monomers=seq_info.monomers,
        residues=current_residues,
        )
      assert len(current_residues) == len(seq_info.monomers), (
          f'Mismatch between sequence and structure for chain {chain_id}')
    nonpoly_chains = {}
    for chain_id, chain_data in nonpoly_structure_mappings.items():
      current = []
      for pos, residue in chain_data.items():
        if residue.name in ('HOH', 'WAT'):
          # AlphaFold3 supplementary section 2.1: waters are removed.
          continue
        current.append(residue)
      if current:
        nonpoly_chains[chain_id] = current

    mmcif_object = MmcifObject(
        file_id=file_id,
        header=header,
        structure=first_model_structure,
        polymer_chains=polymer_chains,
        nonpoly_chains=nonpoly_chains,
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
    logging.warning('Could not determine release_date: %s',
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
  return [AtomSite(*site) for site in zip(  # pylint:disable=g-complex-comprehension
      parsed_info['_atom_site.label_comp_id'],      # residue_name: str
      parsed_info['_atom_site.auth_asym_id'],       # author_chain_id: str
      parsed_info['_atom_site.label_asym_id'],      # mmcif_chain_id: str
      parsed_info['_atom_site.auth_seq_id'],        # author_seq_num: str
      parsed_info['_atom_site.label_seq_id'],       # mmcif_seq_num: int
      parsed_info['_atom_site.pdbx_PDB_ins_code'],  # insertion_code: str
      parsed_info['_atom_site.group_PDB'],          # hetatm_atom: str
      parsed_info['_atom_site.pdbx_PDB_model_num'], # model_num: str
      parsed_info['_atom_site.label_atom_id'],      # name: str
      parsed_info['_atom_site.type_symbol'],        # type: str
      parsed_info['_atom_site.Cartn_x'],            # x: str
      parsed_info['_atom_site.Cartn_y'],            # y: str
      parsed_info['_atom_site.Cartn_z'],            # z: str
      parsed_info['_atom_site.label_alt_id'],       # mmcif_alt_id: str
      parsed_info['_atom_site.occupancy'],          # occupancy: float
      )]


def _get_polymer_chains(
    *, parsed_info: Mapping[str, Any]) -> Mapping[ChainId, Sequence[Monomer]]:
  """Extracts polymer information for protein, DNA and RNA chains.

  Args:
    parsed_info: _mmcif_dict produced by the Biopython parser.

  Returns:
    A dict mapping mmcif chain id to a list of Monomers.
  """
  # Get polymer information for each entity in the structure.
  entity_poly_seqs = mmcif_loop_to_list('_entity_poly_seq.', parsed_info)

  polymers = collections.defaultdict(list)
  for entity_poly_seq in entity_poly_seqs:
    polymers[entity_poly_seq['_entity_poly_seq.entity_id']].append(
        Monomer(id=entity_poly_seq['_entity_poly_seq.mon_id'],
                num=int(entity_poly_seq['_entity_poly_seq.num'])))

  # Get entity information. Will allow us to identify which of these polymers
  # are proteins, DNAs and RNAs.
  entity_polys = mmcif_loop_to_dict('_entity_poly.',
                                    '_entity_poly.entity_id',
                                    parsed_info)

  # Get chains information for each entity. Necessary so that we can return a
  # dict keyed on chain id rather than entity.
  struct_asyms = mmcif_loop_to_list('_struct_asym.', parsed_info)

  entity_to_mmcif_chains = collections.defaultdict(list)
  for struct_asym in struct_asyms:
    chain_id = struct_asym['_struct_asym.id']
    entity_id = struct_asym['_struct_asym.entity_id']
    entity_to_mmcif_chains[entity_id].append(chain_id)

  # Identify and return the valid polymer chains.
  valid_chains = {}
  for entity_id, seq_info in polymers.items():
    entity_poly = entity_polys[entity_id]
    for chain_id in entity_to_mmcif_chains[entity_id]:
      valid_chains[chain_id] = PolyChain(
        entity_id=entity_id,
        orig_entity_id=entity_id,
        type=entity_poly['_entity_poly.type'],
        pdbx_seq=entity_poly['_entity_poly.pdbx_seq_one_letter_code'],
        pdbx_can=entity_poly['_entity_poly.pdbx_seq_one_letter_code_can'],
        mmcif_chain_id=chain_id,
        orig_mmcif_chain_id=chain_id,
        author_chain_id='',
        orig_author_chain_id='',
        monomers=seq_info,
        residues=None,
        )

  return valid_chains


def _is_set(data: str) -> bool:
  """Returns False if data is a special mmCIF character indicating 'unset'."""
  return data not in ('.', '?')


def _select_atoms(atoms: Sequence[AtomSite]) -> Sequence[AtomSite]:
  """Selects atoms as AlphaFold3 supplementary information section 2.1:
     Alternative locations for atoms/residues are resolved by taking the one
     with the largest occupancy.

  Args:
    atoms: all atoms in mmcif file for one residue.

  Returns:
    A list contains selected atoms with largets occupancy.
  """
  if all([_.mmcif_alt_id == '.' for _ in atoms]): # atoms=[] also returns True.
    return atoms
  group_atoms = collections.defaultdict(list)
  for atom in atoms:
      group_atoms[atom.mmcif_alt_id].append(atom)
  common_atoms = group_atoms.pop('.', [])
  max_key, max_occupancy = None, -1.0
  for alt_id, group in group_atoms.items():
    current_occupancy = sum([_.occupancy for _ in group]) / len(group)
    if current_occupancy > max_occupancy:
      max_key, max_occupancy = alt_id, current_occupancy
  return group_atoms[max_key] + common_atoms


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

  file_id = inppath.name.split('.')[0].split('-')[0]
  result = parse(file_id=file_id, mmcif_string=mmcif_string)
  assert result.mmcif_object, f'Parsing failed for {inppath}, {result.errors}'
  polymer_chains = result.mmcif_object.polymer_chains
  nonpoly_chains = result.mmcif_object.nonpoly_chains

  # show mmcif parsing result
  print(file_id)
  print(result.mmcif_object.header)
  print(result.mmcif_object.structure)
  print(type(result.mmcif_object.raw_string))
  print('-'*80)
  print('polymer_chains', len(polymer_chains))
  for chain_id, chain in sorted(polymer_chains.items(), key=lambda x: x[0]):
    print('-'*80)
    print(f'{file_id}_{chain_id}', chain.type, len(chain.residues), 'reisudes')
    print(chain.pdbx_seq)
    # print(chain_data.pdbx_can)
    idx = 1
    for residue in chain.residues:
      if idx <= 5 or idx >= len(chain.residues) - 5:
        print(residue.position, residue.name, len(residue.atoms), 'atoms')
      idx += 1
  print('-'*80)
  print('nonpoly_chains', len(nonpoly_chains))
  for chain_id, residues in sorted(nonpoly_chains.items(), key=lambda x: x[0]):
    residues = nonpoly_chains[chain_id]
    print('-'*80)
    print(f'{file_id}_{chain_id}')
    for residue in residues:
      print(residue.position, residue.name, len(residue.atoms), 'atoms')
