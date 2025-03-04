#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import logging
import pathlib
import pickle
import sys
import tempfile
import zlib
from typing import Any, Mapping, Sequence, Tuple

import joblib
import lmdb
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
from Bio.Data.PDBData import protein_letters_3to1_extended as aa3to1
from Bio.SVDSuperimposer import SVDSuperimposer
from posebusters import PoseBusters
from rdkit import Chem
from scipy.spatial import distance_matrix
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)

IMPOSE_BOUNDARY = 10
FULL_REPORT = True


def bstr2obj(bstr: bytes):
  return pickle.loads(zlib.decompress(bstr))


def convert_to_mmcif_string(
  inpfile: str,
  data: Mapping[str, Any],
) -> str:
  """Convert input PDB structure to MMCIF string."""
  mmcif_str = ''

  try:
    assert pathlib.Path(inpfile).is_file(), 'Input structure does not exist'
    assert pathlib.Path(inpfile).suffix == '.pdb', 'Input structure must be PDB'
    assert data and 'name' in data and 'nonpoly_graphs' in data, (
      f'Wrong popcessed data format')

    key = pathlib.Path(inpfile).stem.split('-')[0]
    name = data['name']
    assert name == key, f'Wrong name for {inpfile}, {name} != {key}'
    pdbid, ligid = name.split('_')

    ccds = [_['name'] for _ in data['nonpoly_graphs']]
    assert 1 == len(set(ccds)), f'Multiple CCDs {ccds} in nonpoly_graphs'
    assert ligid == data['nonpoly_graphs'][0]['name'], 'Wrong ligand name'
    atomids = data['nonpoly_graphs'][0]['atomids']
    symbols = data['nonpoly_graphs'][0]['symbols']

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(f'{pdbid}_{ligid}', inpfile)
    assert len(structure) == 1, 'Must 1 model in input structure'

    hetres = []
    for chain in structure[0]:
      for residue in chain:
        if residue.id[0] == ' ':
          continue
        hetres.append(residue)
    assert (
      len(hetres) > 0 and
      len(hetres[0]) == len(atomids) and
      all(
        a.element.upper()==_.upper()
        for a, _ in zip(hetres[0].get_atoms(), symbols)
      )
    ), f'Wrong ligand {ligid} parsed from input structure'

    ligres = hetres[0]
    ligres.resname = ligid
    ligres.id = (f'H_{ligid}', 1, ' ')
    for atom, atomid in zip(ligres.get_atoms(), atomids):
      atom.id = atomid
      atom.name = atomid

    with io.StringIO() as sio:
      mmcifio = PDB.MMCIFIO()
      mmcifio.set_structure(structure)
      mmcifio.save(sio)
      mmcif_str = sio.getvalue()
  except Exception as e:
    logging.error('Failed to convert %s to mmcif, %s', inpfile, e)

  return mmcif_str


def collect_models(
  lmdb_dir: str,
  result_dir: str,
) -> Sequence[Mapping[str, Any]]:
  """Collect models from LMDB and result directory."""
  models = []

  with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
    metadata = bstr2obj(txn.get('__metadata__'.encode()))
    print('-'*80)
    print(metadata['comment'], end='')
    for k, v in metadata.items():
      k != 'comment' and print(k, len(v))
    print(f"metadata['keys'][:10]={metadata['keys'][:10]}")
    print('-'*80)

    assert 'keys' in metadata, f'keys not in metadata for {lmdb_dir}.'
    assert 'rawstrings' in metadata, f'rawstrings not in metadata for {lmdb_dir}.'
    names, strdicts = metadata['keys'], metadata['rawstrings']
    assert len(strdicts) == len(names), f'Mismatch between keys and rawstrings'
    logging.info('%d keys and rawstrings from %s', len(names), lmdb_dir)
    assert 428 == len(names) and all('_'==_[4] and _.isupper() for _ in names), (
      f'Expected 428 samples in XXXX_X(XX) format, got {len(names)}')

    for name, strdict in tqdm(zip(names, strdicts), desc='Collecting models...'):
      # if name not in ('7K0V_VQP'):
      #   continue

      data = bstr2obj(txn.get(name.encode()))
      assert data is not None, f'Failed to get {name} from lmdb {lmdb_dir}'

      inppath = pathlib.Path(result_dir).resolve()
      for p in inppath.glob(f'{name}-*'):
        if p.stem.endswith('native'):
          continue
        if p.suffix == '.pdb':
          inpcifstr = convert_to_mmcif_string(str(p), data)
        elif p.suffix == '.cif':
          inpcifstr = p.read_text()
        else:
          logging.error('Invalid input structure %s', p)
          inpcifstr = ''

        mmcifdict = PDB.MMCIF2Dict.MMCIF2Dict(io.StringIO(inpcifstr))
        assert '_atom_site.B_iso_or_equiv' in mmcifdict, f'Wrong B_iso in {p}'
        bf = [float(_) for _ in mmcifdict['_atom_site.B_iso_or_equiv']]
        score = sum(bf) / len(bf)

        models.append({
          'name': name,
          'model_index': int(p.stem.removeprefix(f'{name}-')),
          'ranking_score': score,
          'refccdstr': strdict['refccd'],
          'refcifstr': strdict['refcif'],
          'refpdbstr': strdict['refpdb'],
          'refsdfstr': strdict['refsdf'],
          'inpcifstr': inpcifstr,
        })

  return models


def convert_ligand_to_rdkitmol(
  ligand_coords: Mapping[str, Tuple[float, float, float]],
  ligand_refccdstr: str,
) -> Chem.Mol:
  """Convert ligand coordinates to RDKit mol by using reference CCD.cif."""
  try:
    # process ideal chemical component
    handle = io.StringIO(ligand_refccdstr)
    mmcifdict = PDB.MMCIF2Dict.MMCIF2Dict(handle)
    # parse chem_comp_atoms
    id2index = {}
    chem_comp_atoms = []
    for i, (_id, _symbol, _charge) in enumerate(zip(
      mmcifdict['_chem_comp_atom.atom_id'],
      mmcifdict['_chem_comp_atom.type_symbol'],
      mmcifdict['_chem_comp_atom.charge'],
    )):
      assert _id not in id2index, f'Duplicate atom {_id} in CCD.cif.'
      id2index[_id] = i
      chem_comp_atoms.append({
        'id': _id,
        'symbol': _symbol[0].upper() + _symbol[1:].lower(),
        'charge': int(_charge) if _charge != '?' else 0,
      })
    # parse chem_comp_bonds
    atompairs = set()
    chem_comp_bonds = []
    for id1, id2, order in zip(
      mmcifdict['_chem_comp_bond.atom_id_1'],
      mmcifdict['_chem_comp_bond.atom_id_2'],
      mmcifdict['_chem_comp_bond.value_order'],
    ):
      assert id1 in id2index and id2 in id2index, (
        f'Invalid bond for atom pair ({id1}, {id2}) in CCD.cif.')
      assert (id1, id2) not in atompairs and (id2, id1) not in atompairs, (
        f'Duplicate atom pair ({id1}, {id2}) in CCD.cif.')
      atompairs.add((id1, id2))
      chem_comp_bonds.append({'id1': id1, 'id2': id2,'order': order})
    # extract feature for chem_comp atoms and bonds
    symbols = [_['symbol'] for _ in chem_comp_atoms]
    charges = [_['charge'] for _ in chem_comp_atoms]
    BONDORDER = {'SING': 1, 'DOUB': 2, 'TRIP': 3}
    orders = [(id2index[_['id1']], id2index[_['id2']], BONDORDER[_['order']])
              for _ in chem_comp_bonds]
    # check consistency between ligand coords and refccd
    # assert set(ligand_coords.keys()).issubset(set(id2index.keys())), (
    #   f'Mismatch between refccd and ligand')
    coords = [ligand_coords.get(_['id'], (0., 0., 0.)) for _ in chem_comp_atoms]
    # convert atom symbols and bond orders to mol by using RDKit
    with Chem.RWMol() as mw:
      conf = Chem.Conformer()
      for i, a, c, (x, y, z) in zip(range(len(symbols)), symbols, charges, coords):
        atom = Chem.Atom(a)
        atom.SetFormalCharge(c)
        mw.AddAtom(atom)
        conf.SetAtomPosition(i, (x, y, z))
      mw.AddConformer(conf)

      for i, j, order in orders:
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

      return mw
  except Exception as e:
    logging.error('Failed to convert cif to mol, %s', e)
    return None


def read_structure_and_ligand_from_mmcif(
  mmcif_str: str,
  refccd_str: str,
  ligid: str,
) -> PDB.Structure.Structure:
  """Read structure and ligang mols from mmcif string."""
  # Read structure from mmcif string
  parser = PDB.MMCIFParser(QUIET=True)
  handle = io.StringIO(mmcif_str)
  full_structure = parser.get_structure('', handle)
  first_model_structure = next(full_structure.get_models())

  # Check if the first line of mmcif string is data_LIGID
  first_line = refccd_str.split('\n')[0]
  assert first_line.startswith('data_') and first_line.endswith(ligid), (
    f'Invalid reference ccd {ligid} string: {first_line}')

  # Get ligand molecules from structure
  ligands = []
  for residue in first_model_structure.get_residues():
    if residue.get_resname() != ligid:
      continue
    coords = {_.get_name(): _.get_coord().tolist() for _ in residue}
    mol = Chem.RemoveHs(convert_ligand_to_rdkitmol(coords, refccd_str))
    conf = np.mean([_.get_bfactor() for _ in residue])
    ligands.append((mol, conf))
  assert len(ligands) > 0, f'No ligand {ligid} in sturcture'

  return first_model_structure, ligands


def make_alignmets_by_biopython(
  seq: str,
  pdbseq: str,
) -> Any:
  """Align predicted sequence onto reference sequence by Biopython."""
  alignments = PairwiseAligner(scoring='blastp').align(seq, pdbseq)
  if len(alignments) > 1:
    # parameters copy from hh-suite/scripts/renumberpdb.pl
    # https://github.com/soedinglab/hh-suite/blob/master/scripts/renumberpdb.pl
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = -3
    aligner.target_open_gap_score = -20
    aligner.extend_gap_score = -0.1
    aligner.end_gap_score = -0.09
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    alignments = aligner.align(seq, pdbseq)
  return alignments


def calc_rmsd(
  x1: np.ndarray,
  x2: np.ndarray,
  ref1: np.ndarray | None = None,
  ref2: np.ndarray | None = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
  """Calculate RMSD between two sets of coordinates."""
  if ref1 is None or ref2 is None:
    ref1, ref2 = x1, x2
  sup = SVDSuperimposer()
  sup.set(ref1, ref2)
  sup.run()
  rot, tran = sup.get_rotran()
  x2_t = np.dot(x2, rot) + tran
  rmsd = np.sqrt(((((x1 - x2_t) ** 2)) * 3).mean())
  return rmsd, rot, tran


def evaluate_one_model(
  model: Mapping[str, Any],
  impose_boundary: float = 10.,
  full_report: bool = True,
  save_ligand: bool = False,
) -> Mapping[str, Any]:
  """Evaluate one model by USalign TM-score and ligand RMSD."""
  score = {
    'ID': '{}-{}'.format(model['name'], model['model_index']),
    'Name': model['name'],
    'ModelIndex': model['model_index'],
    'RankingScore': model['ranking_score'],
    'PocketConf': 0.,
    'PocketRMSD': 10000.,
    'LigandConf': 0.,
    'LigandRMSD': 10000.,
    'PocketAlignedConf': 0.,
    'PocketAlignedRMSD': 10000.,
    'OutSDFString': '',
    'BustRMSD': 10000.,
    'BustKabschRMSD': 10000.,
    'BustCentroidDistance': 10000.,
  }

  try:
    pdbid, ligid = model['name'].split('_')
    # Read reference and predicted structures
    refmodel, refligs = read_structure_and_ligand_from_mmcif(
      model['refcifstr'], model['refccdstr'], ligid)
    inpmodel, inpligs = read_structure_and_ligand_from_mmcif(
      model['inpcifstr'], model['refccdstr'], ligid)
    # Check the consistency between predicted and reference ligand molecules
    smiles = Chem.MolToSmiles(refligs[0][0])
    assert all(Chem.MolToSmiles(_[0]) == smiles for _ in refligs + inpligs), (
      'Mismatch ligand molecules between predicted and reference')
    # Calculate ligand RMSD, pocket RMSD and pocket aligned RMSD
    for refmol, refconf in refligs:
      # Get reference ligand coordinates
      refligcrd = refmol.GetConformer().GetPositions()
      # Get primary binding polymer chain for reference mol
      refid, maxnum = None, 0
      for chain in refmodel:
        coords = [_['CA'].coord for _ in chain if _.id[0] == ' ' and 'CA' in _]
        if len(coords) < 1:
          continue
        dist = distance_matrix(coords, refligcrd)
        num = np.sum(np.min(dist, axis=1) < impose_boundary)
        if num > maxnum:
          refid, maxnum = chain.id, num
      if refid is None:
        # Skip this refernece mol if no primary binding polymer chain
        continue
      # Get reference residues and sequence for primary binding polymer chain
      refres = [_ for _ in refmodel[refid] if _.get_id()[0] == ' ']
      refseq = ''.join([aa3to1.get(_.resname, 'X') for _ in refres])
      # Align predicted polymer and ligand onto reference polymer and ligand
      for inpchain in inpmodel.get_chains():
        # Get predicted residues for one predicted chain
        inpres = [_ for _ in inpchain if _.get_id()[0] == ' ']
        if len(inpres) < 1:
          # Skip this predicted chain if no residues
          continue
        inpseq = ''.join([aa3to1.get(_.resname, 'X') for _ in inpres])
        # Mapping current inpseq onto refseq
        ali = make_alignmets_by_biopython(inpseq, refseq)[0]
        numali = sum([a != '-' and b != '-' for a, b in zip(ali[0], ali[1])])
        numequ = sum([a == b for a, b in zip(ali[0], ali[1])])
        if 1. * numequ / numali < 0.8:
          # Skip this predicted chain if <80% residues are identically aligned
          logging.debug('%s and %s are different', refmodel[refid], inpchain)
          continue
        # Get pocket residues from reference residues and predicted residues
        inppkt, refpkt = [], []
        for i in range(ali.length):
          m, n = ali.indices[0][i], ali.indices[1][i]
          if m != -1 and n != -1 and 'CA' in inpres[m] and 'CA' in refres[n]:
            _dist = distance_matrix([refres[n]['CA'].get_coord()], refligcrd)
            if np.min(_dist) < impose_boundary:
              inppkt.append(inpres[m])
              refpkt.append(refres[n])
        inppktcrd = np.array([_['CA'].get_coord() for _ in inppkt])
        refpktcrd = np.array([_['CA'].get_coord() for _ in refpkt])
        # Calculate pocket aligned RMSD for each predicted ligand
        for inpmol, inpconf in inpligs:
          # Get predicted ligand coordinates
          inpligcrd = inpmol.GetConformer().GetPositions()
          # Calculate pocket aligned RMSD
          _rmsd, _r, _t = calc_rmsd(refligcrd, inpligcrd, refpktcrd, inppktcrd)
          # Skip this predicted ligand if pocket aligned RMSD is large
          if _rmsd >= score['PocketAlignedRMSD']:
            continue
          # Update the best aligned RMSD
          best_inpligcrd_t = np.dot(inpligcrd, _r) + _t
          _pocketconf = np.mean(
            np.array([_['CA'].get_bfactor() for _ in inppkt])
          )
          score.update({
            'PocketConf': _pocketconf,
            'PocketRMSD': calc_rmsd(refpktcrd, inppktcrd)[0],
            'LigandConf': inpconf,
            'LigandRMSD': calc_rmsd(refligcrd, inpligcrd)[0],
            'PocketAlignedConf': (inpconf + _pocketconf) / 2,
            'PocketAlignedRMSD': _rmsd,
          })
          with io.StringIO() as sio:
            with Chem.SDWriter(sio) as w:
              _outmol = Chem.Mol(inpmol)
              _conformer = _outmol.GetConformer()
              for i in range(_outmol.GetNumAtoms()):
                _conformer.SetAtomPosition(i, best_inpligcrd_t[i])
              _outmol.SetProp('_Name', f'{pdbid}_{ligid}_Final')
              w.write(_outmol)
            score['OutSDFString'] = sio.getvalue()
    assert score['OutSDFString'], 'Failed to calculate pocket aligned RMSD'
    outsdfstr = score['OutSDFString']

    # Run PoseBusters command to evaluate the model
    result = None
    with (tempfile.NamedTemporaryFile(suffix='.sdf') as mol_pred,
          tempfile.NamedTemporaryFile(suffix='.sdf') as mol_true,
          tempfile.NamedTemporaryFile(suffix='.pdb') as mol_cond):
      mol_pred.write(outsdfstr.encode())
      mol_pred.flush()
      mol_true.write(model['refsdfstr'].encode())
      mol_true.flush()
      mol_cond.write(model['refpdbstr'].encode())
      mol_cond.flush()
      result = PoseBusters().bust(
        mol_pred=mol_pred.name,
        mol_true=mol_true.name,
        mol_cond=mol_cond.name,
        full_report=full_report,
      )
    assert result is not None, 'PoseBusters command failed.'
    score.update({
      'BustRMSD': result['rmsd'].iloc[0],
      'BustKabschRMSD': result['kabsch_rmsd'].iloc[0],
      'BustCentroidDistance': result['centroid_distance'].iloc[0],
    })
  except Exception as e:
    logging.error('Failed to transform model %s, %s', model['name'], e)

  if not save_ligand:
    del score['OutSDFString']

  return score


if __name__ == '__main__':
  if len(sys.argv) != 3 and len(sys.argv) != 4:
    sys.exit(f'Usage: {sys.argv[0]} <posebusters_lmdb> <result_dir> [output_dir]')
  pblmdb, result_dir = sys.argv[1:3]
  output_dir = sys.argv[3] if len(sys.argv) == 4 else None

  assert pathlib.Path(pblmdb).exists(), f'{pblmdb} does not exist'
  assert pathlib.Path(result_dir).exists(), f'{result_dir} does not exist'
  result_dir = str(pathlib.Path(result_dir).resolve())
  output_dir and pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

  logging.info('Collecting models for evaluation...')
  models = collect_models(pblmdb, result_dir)
  assert len(models) > 0, f'No models collected from {result_dir}'
  logging.info('%d models collected from %s', len(models), result_dir)

  # print(evaluate_one_model(models[0]))
  # exit('Debug')
  scores = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(evaluate_one_model)(_, IMPOSE_BOUNDARY, FULL_REPORT)
    for _ in tqdm(models, desc='Evaluating models by PoseBusters...')
  )
  df = pd.DataFrame(scores)
  print(df)
  df.to_csv(f'{result_dir}_Score4EachModel.csv', index=False)
  logging.info('%d records in evaluation result', len(df))

  logging.info('Saving average score for each target to %s.', output_dir)
  if df.isnull().values.any():
    df.fillna(10000., inplace=True)
  if 'Ranking' not in df.columns:
    dfs = []
    for name, gdf in df.groupby('Name'):
      gdf.sort_values(by=['LigandConf'], ascending=False, inplace=True)
      gdf['Ranking'] = range(1, len(gdf) + 1)
      dfs.append(gdf)
    df = pd.concat(dfs).sort_index()
  if 'Orcale' not in df.columns:
    dfs = []
    for name, gdf in df.groupby('Name'):
      gdf.sort_values(by=['BustRMSD'], ascending=True, inplace=True)
      gdf['Oracle'] = range(1, len(gdf) + 1)
      dfs.append(gdf)
    df = pd.concat(dfs).sort_index()
  logging.info('Saving score4model to %s', f'{result_dir}_Score4EachModel.csv')

  logging.info('Calculating average score for each target...')
  records = []
  cols = ['ModelIndex', 'Ranking', 'Oracle',
          'PocketRMSD', 'BustRMSD', 'BustKabschRMSD', 'BustCentroidDistance']
  for name, gdf in df.groupby('Name'):
    record = {'Name': name}
    rnd1 = gdf[gdf['ModelIndex'] == 1].iloc[0]
    for k, v in rnd1[cols].to_dict().items():
      record[f'Rnd1_{k}'] = v
    top1 = gdf[gdf['Ranking'] == 1].iloc[0]
    for k, v in top1[cols].to_dict().items():
      record[f'Top1_{k}'] = v
    best = gdf[gdf['Oracle'] == 1].iloc[0]
    for k, v in best[cols].to_dict().items():
      record[f'Best_{k}'] = v
    records.append(record)
  newdf = pd.DataFrame(records)
  print(newdf)
  newdf.to_csv(f'{result_dir}_Score4Target.csv', index=False)
  logging.info('Saving score4target to %s', f'{result_dir}_Score4Target.csv')

  logging.info('Calculating success rate...')
  meandict = [{
    'Metric': 'SuccessRate',
    'Number': len(newdf),
    'Rnd1(%)': 1. * sum(newdf['Rnd1_BustRMSD'] < 2.0) / len(newdf),
    'Top1(%)': 1. * sum(newdf['Top1_BustRMSD'] < 2.0) / len(newdf),
    'Best(%)': 1. * sum(newdf['Best_BustRMSD'] < 2.0) / len(newdf),
  }]
  meandf = pd.DataFrame(meandict).set_index('Metric')
  with pd.option_context('display.float_format', '{:.2%}'.format):
    print(meandf)
