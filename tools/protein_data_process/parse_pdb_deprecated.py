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

"""Parses the pdb file format."""
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
ResType = str
PdbDict = Mapping[str, Sequence[str]]


@dataclasses.dataclass(frozen=True)
class Monomer:
  id: str
  num: int


@dataclasses.dataclass(frozen=True)
class AtomCartn:
  name: str
  type: str
  x: float
  y: float
  z: float


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
  is_missing: bool
  hetflag: str


@dataclasses.dataclass(frozen=True)
class PdbObject:
  """Representation of a parsed pdb file.

  Contains:
    file_id: A meaningful name, e.g. a pdb_id. Should be unique amongst all
      files being processed.
    header: Biopython header.
    structure: Biopython structure.
    chain_to_seqres: Dict mapping chain_id to 1 letter amino acid sequence. E.g.
      {'A': 'ABCDEFG'}
    chain_to_restpye: Dict mapping chain_id to 1 letter amino acid type. E.g.
      {'A': 'ppppppp'}
    seqres_to_structure: Dict; for each chain_id contains a mapping between
      SEQRES index and a ResidueAtPosition. e.g. {'A': {0: ResidueAtPosition,
                                                        1: ResidueAtPosition,
                                                        ...}}
  """
  file_id: str
  header: PdbHeader
  structure: PdbStructure
  chain_to_seqres: Mapping[ChainId, SeqRes]
  chain_to_restype: Mapping[ChainId, ResType]
  seqres_to_structure: Mapping[ChainId, Sequence[Tuple]]


@dataclasses.dataclass(frozen=True)
class ParsingResult:
  """Returned by the parse function.

  Contains:
    pdb_object: A PdbObject, may be None if no chain could be successfully
      parsed.
    errors: A dict mapping (file_id, chain_id) to any exception generated.
  """
  pdb_object: Optional[PdbObject]
  errors: Mapping[Tuple[str, str], Any]


class ParseError(Exception):
  """An error indicating that an pdb file could not be parsed."""


def _get_first_model(structure: PdbStructure) -> PdbStructure:
  """Returns the first model in a Biopython structure."""
  return next(structure.get_models())


def _get_header(parsed_header: PdbHeader) -> PdbHeader:
  """Returns a basic header containing method, release date and resolution."""
  header = {'structure_method': parsed_header.get('structure_method', ''),
            'release_date': parsed_header.get('release_date', ''),
            'resolution': parsed_header.get('resolution', None)}
  return header


@functools.lru_cache(16, typed=False)
def parse_structure(*,
          file_id: str,
          pdb_string: str,
          catch_all_errors: bool = True) -> ParsingResult:
  """Entry point, parses an pdb_string.

  Args:
    file_id: A string identifier for this file. Should be unique within the
      collection of files being processed.
    pdb_string: Contents of an pdb file.
    catch_all_errors: If True, all exceptions are caught and error messages are
      returned as part of the ParsingResult. If False exceptions will be allowed
      to propagate.

  Returns:
    A ParsingResult.
  """
  errors = {}
  try:
    parser = PDB.PDBParser(QUIET=True)
    handle = io.StringIO(pdb_string)
    full_structure = parser.get_structure(file_id, handle)
    first_model_structure = _get_first_model(full_structure)

    full_pdb_header = parser.get_header()
    header = _get_header(full_pdb_header)

    # Determine all chains no matter polymer or non-polymer
    seq_to_structure = {}
    for chain in first_model_structure.get_chains():
      chain_id = chain.get_id()
      current_chain = {}
      for residue in chain.get_residues():
        hetflag, residue_number, insertion_code = residue.get_id()
        if insertion_code not in (' ', 'A'):
          # We only process residue insertion code in (' ', 'A').
          # Similar to AlphaFold3 supplementary section 2.1
          # Alternative locations for atoms/residues are resolved by taking the one
          # with the largest occupancy.
          continue
        if hetflag == 'W':
          # Similar to AlphaFold3 supplementary section 2.1: waters are removed.
          continue
        _position = ResiduePosition(chain_id=chain_id,
                                    residue_number=residue_number,
                                    insertion_code=insertion_code)
        _residue = ResidueAtPosition(position=_position,
                                     name=residue.get_resname(),
                                     is_missing=False,
                                     hetflag=hetflag)
        _atoms = [AtomCartn(_.name, _.element, *_.coord) for _ in residue]
        current_chain[residue_number] = (_residue, _atoms)
      seq_to_structure[chain_id] = current_chain

    # check chains without any residues
    for chain_id, chain_data in seq_to_structure.copy().items():
      if not chain_data:
        logging.warning(f"Chain '{chain_id}' in {file_id} has no residues.")
        del seq_to_structure[chain_id]
    if not seq_to_structure:
      return ParsingResult(
          None, {(file_id, ''): 'No polymer chains found in this file.'})

    # Add missing residue information and check if there are missing residues
    for residue in full_pdb_header['missing_residues']:
      resname = residue['res_name']
      chain_id = residue['chain']
      residue_number = residue['ssseq']
      current_chain = seq_to_structure.get(chain_id, {})
      _position = ResiduePosition(chain_id=chain_id,
                                  residue_number=residue_number,
                                  insertion_code=' ')
      _residue = ResidueAtPosition(position=_position,
                                   name=resname,
                                   is_missing=True,
                                   hetflag=' ')
      _atoms = []
      current_chain[residue_number] = (_residue, _atoms)
      seq_to_structure[chain_id] = current_chain

    # check missing residues for polymer chains
    for chain_id, chain_data in seq_to_structure.items():
      polymer = {i:_ for i, _ in chain_data.items() if _[0].hetflag == ' '}
      if not polymer:
        # Skip non-polymer chains
        continue
      if not all([PDB.Polypeptide.is_aa(f'{_[0].name:<3s}'.upper())
              for i, _ in polymer.items()]):
        # Skip processing DNA and RNA, only process protein chains
        # TODO: process DNA and RNA chains
        continue
      _min, _max = min(polymer.keys()), max(polymer.keys())
      for residue_number in set(range(_min, _max+1)) - polymer.keys():
        _residue = ResidueAtPosition(None, 'UNK', True, ' ')
        seq_to_structure[chain_id][residue_number] = (_residue, [])

    # convert seq_to_structure[chain_id] from dict to list
    for chain_id, chain in seq_to_structure.items():
      seq_to_structure[chain_id] = [chain[_] for _ in sorted(chain.keys())]

    # AlphaFold3 supplementary information section 2.1
    # MSE residues are converted to MET residues
    for chain_id, str_info in seq_to_structure.items():
      for residue, atoms in str_info:
        if residue.name == 'MSE':
          _residue = ResidueAtPosition(position=residue.position,
                                       name='MET',
                                       is_missing=residue.is_missing,
                                       hetflag=' ')
          _atoms = atoms.copy()
          for i, atom in enumerate(atoms):
            if atom.name == 'SE':
              _atoms[i] = AtomCartn('SD', 'S', atom.x, atom.y, atom.z)
          seq_to_structure[chain_id][i] = (_residue, _atoms)

    # Convert the sequence information to a string.
    author_chain_to_sequence = {}
    author_chain_to_restypes = {}
    for author_chain, str_info in seq_to_structure.items():
      seq = []
      res = []
      for residue, atoms in str_info:
        resname = f'{residue.name:<3s}'.upper()
        _code, _type = '?', '*' # including polymer and non-polymer residues
        if residue.hetflag == ' ':
          if PDB.Polypeptide.is_aa(resname) or resname == 'UNK':
            _code = PDB.Polypeptide.protein_letters_3to1.get(resname, 'X')
            _type = 'p'
          elif PDB.Polypeptide.is_nucleic(resname) or resname in ('DN ', 'N  '):
            _code = PDB.Polypeptide.nucleic_letters_3to1.get(resname, 'N')
            _type = 'n'
        seq.append(_code)
        res.append(_type)
      if '*' in res:
        if res.count('p') >= res.count('n'):
          seq = ['X' if _ == '?' else _ for _ in seq]
          res = ['p' if _ == '*' else _ for _ in res]
        else:
          seq = ['N' if _ == '?' else _ for _ in seq]
          res = ['n' if _ == '*' else _ for _ in res]
      author_chain_to_sequence[author_chain] = ''.join(seq)
      author_chain_to_restypes[author_chain] = ''.join(res)

    pdb_object = PdbObject(
        file_id=file_id,
        header=header,
        structure=first_model_structure,
        chain_to_seqres=author_chain_to_sequence,
        chain_to_restype=author_chain_to_restypes,
        seqres_to_structure=seq_to_structure,
        )

    return ParsingResult(pdb_object=pdb_object, errors=errors)
  except Exception as e:  # pylint:disable=broad-except
    errors[(file_id, '')] = e
    if not catch_all_errors:
      raise
    return ParsingResult(pdb_object=None, errors=errors)


if __name__ == '__main__':
  import sys
  from pathlib import Path

  if len(sys.argv) != 2:
    sys.exit(f"Usage: {sys.argv[0]} <input_pdb_path>")
  inppath = Path(sys.argv[1])

  assert inppath.suffix == '.pdb', f"Input pdb file should be .pdb, {inppath}"
  file_id = inppath.stem
  with open(inppath, 'r') as fp:
    pdb_string = fp.read()

  result = parse_structure(file_id=file_id, pdb_string=pdb_string)
  assert result.pdb_object, f"Parsing failed for {inppath}, {result.errors}"
  seqres = result.pdb_object.chain_to_seqres
  restype = result.pdb_object.chain_to_restype
  struct = result.pdb_object.seqres_to_structure

  # show pdb parsing result
  print(file_id)
  print(result.pdb_object.header)
  print(sorted(result.pdb_object.chain_to_seqres.keys()))
  print(result.pdb_object.structure)
  for chain_id in sorted(seqres.keys(), key=lambda x: x[0]):
    print('-'*80)
    print(f"{file_id}_{chain_id}")
    print(seqres[chain_id])
    print(restype[chain_id])
    idx = 1
    for residue, atoms in struct[chain_id]:
      if idx <= 5 or idx >= len(struct[chain_id]) - 5:
        print(idx, residue, len(atoms), "atoms")
      idx += 1
