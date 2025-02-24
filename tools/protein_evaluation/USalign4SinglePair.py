#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import subprocess
import sys
from typing import Any, Mapping


def USalign4SinglePair(
  predicted_file: str,
  reference_file: str,
) -> Mapping[str, Any]:
  """USalign superimpose predicted structure onto reference structure.
   0#|
   1#|Name of Structure_1: 7k0v_vqp.cif:B:C:A:D:: (to be superimposed onto Structure_2)
   2#|Name of Structure_2: 7K0V_VQP_protein.pdb:A:D:::B:C
   3#|Length of Structure_1: 1152 residues
   4#|Length of Structure_2: 1064 residues
   5#
   6#|Aligned length= 273, RMSD=   2.94, Seq_ID=n_identical/n_aligned= 0.974
   7#|TM-score= 0.22841 (normalized by length of Structure_1: L=1152, d0=11.14)
   8#|TM-score= 0.24698 (normalized by length of Structure_2: L=1064, d0=10.80)
   9#|(You should use TM-score normalized by length of the reference structure)
  10#
  11#|(":" denotes residue pairs of d < 5.0 Angstrom, "." denotes other aligned residues)
  12#|SeqB*SeqC*SeqA*SeqD*-*-*
  13#|::::*::::*::::*::::*:*:*
  14#|SeqA*SeqD*-*-*SeqB*SeqC*
  15#
  16#|------ The rotation matrix to rotate Structure_1 to Structure_2 ------
  17#|m               t[m]        u[m][0]        u[m][1]        u[m][2]
  18#|0      35.2577312563  -0.6456319378  -0.2817654486   0.7097659000
  19#|1     -43.2018555219  -0.2709565617  -0.7844415087  -0.5578835551
  20#|2     -58.9685608561   0.7139621436  -0.5525031686   0.4301142942
  21#
  22#|Code for rotating Structure 1 from (x,y,z) to (X,Y,Z):
  23#|for(i=0; i<L; i++)
  24#|{
  25#|   X[i] = t[0] + u[0][0]*x[i] + u[0][1]*y[i] + u[0][2]*z[i];
  26#|   Y[i] = t[1] + u[1][0]*x[i] + u[1][1]*y[i] + u[1][2]*z[i];
  27#|   Z[i] = t[2] + u[2][0]*x[i] + u[2][1]*y[i] + u[2][2]*z[i];
  28#|}
  29#|#Total CPU time is  2.28 seconds
  30#|
  """
  score = {
    'PredictedFile': '',
    'ReferenceFile': '',
    'PredictedLen': 0,
    'ReferenceLen': 0,
    'AlignedLen': 0,
    'PredNormScore': -1.,
    'RefeNormScore': -1.,
    'TMscore': -1.,
    'RMSD': -1.,
    'SeqID': -1.,
    'PredictedChains': [],
    'ReferenceChains': [],
    'RotationMatrix': [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
    'TranslationVector': [0., 0., 0.],
  }
  try:
    exitcode, output = subprocess.getstatusoutput('which USalign')
    assert exitcode == 0, f'Program USalign not installed, {output}.'

    score['PredictedFile'] = predicted_file
    assert os.path.exists(predicted_file), f'{predicted_file} does not exist'

    score['ReferenceFile'] = reference_file
    assert os.path.exists(reference_file), f'{reference_file} does not exist'

    cmd = ['USalign', '-TMscore', '7', '-ter', '1', '-m', '-']
    cmd.extend([predicted_file, reference_file])
    # print(' '.join(cmd))
    lines = subprocess.run(cmd, capture_output=True, text=True).stdout.split('\n')

    ihead, itail = -1, -1
    for i, l in enumerate(lines):
      if l.startswith('Name of Structure_1:'):
        ihead = i
      elif l.startswith('#Total CPU time is'):
        itail = i
    assert ihead > 0 and itail > 0, f'Wrong USalign output'

    for l in lines[ihead:itail]:
      cols = l.split()
      if l.startswith('Name of Structure_1:'):
        score['PredictedChains'] = cols[3].split(':')[1:]
      elif l.startswith('Name of Structure_2:'):
        score['ReferenceChains'] = cols[3].split(':')[1:]
      elif l.startswith('Length of Structure_1:'):
        score['PredictedLen'] = int(cols[3])
      elif l.startswith('Length of Structure_2:'):
        score['ReferenceLen'] = int(cols[3])
      elif l.startswith('Aligned length='):
        score['AlignedLen'] = int(cols[2].strip(','))
        score['RMSD'] = float(cols[4].strip(','))
        score['SeqID'] = float(cols[6])
      elif l.startswith('TM-score=') and cols[6] == 'Structure_1:':
        score['PredNormScore'] = float(cols[1])
      elif l.startswith('TM-score=') and cols[6] == 'Structure_2:':
        score['RefeNormScore'] = float(cols[1])
        score['TMscore'] = float(cols[1])
      elif len(cols) == 5 and cols[0] in ('0', '1', '2'):
        m = int(cols[0])
        score['TranslationVector'][m] = float(cols[1])
        score['RotationMatrix'][m] = [float(_) for _ in cols[2:5]]
      else:
        pass
  except Exception as e:
    logging.error('Failed to run USalign %s, %s', ' '.join(cmd), e)

  return score


if __name__ == '__main__':
  if len(sys.argv) != 3:
    sys.exit(f'Usage: {sys.argv[0]} <predicted_model_pdb> <native_structure_pdb>')
  prepdb, natpdb = sys.argv[1:]

  # USalign between predicted and native pdb
  score = USalign4SinglePair(prepdb, natpdb)
  keys = [
    'PredictedFile', 'ReferenceFile', 'PredictedLen', 'ReferenceLen', 'AlignLen',
    'PredNormScore', 'RefeNormScore', 'TMscore', 'RMSD', 'SeqID',
  ]
  print(' '.join(['%s' % score.get(_, 'XXXXX') for _ in keys[:2]]), end=' ')
  print(' '.join(['%5d' % score.get(_, 0) for _ in keys[2:5]]), end=' ')
  print(' '.join(['%10.3f' % score.get(_, 0.) for _ in keys[5:]]))

  print(sys.argv[0], 'Done.')
