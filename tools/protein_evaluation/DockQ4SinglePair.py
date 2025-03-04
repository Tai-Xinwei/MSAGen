#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import subprocess
import sys
from typing import Any, Mapping


def DockQ4SinglePair(
  predicted_file: str,
  reference_file: str,
) -> Mapping[str, Any]:
  """Evaluate one model by DockQ."""
  score = {
    'PredictedFile': '',
    'ReferenceFile': '',
    'DockQ': -1.,
    'iRMSD': -1.,
    'LRMSD': -1.,
    'fnat': -1.,
    'fnonnat': -1.,
    'F1': -1.,
    'clashes': -1,
  }
  try:
    exitcode, output = subprocess.getstatusoutput('which DockQ')
    assert exitcode == 0, f'Program DockQ not installed, {output}.'

    score['PredictedFile'] = predicted_file
    assert os.path.exists(predicted_file), f'{predicted_file} not found'

    score['ReferenceFile'] = reference_file
    assert os.path.exists(reference_file), f'{reference_file} not found'

    cmd = ['DockQ', predicted_file, reference_file]
    # print(' '.join(cmd))
    lines = subprocess.run(cmd, capture_output=True, text=True).stdout.split('\n')

    for l in lines:
      l = l.strip()
      if l.startswith(
        ('DockQ:', 'iRMSD:', 'LRMSD:', 'fnat:', 'fnonnat:', 'F1:', 'clashes:')
      ):
        cols = l.split(':')
        score[cols[0]] = float(cols[1])
  except Exception as e:
    logging.error('Failed to run USalign %s, %s', ' '.join(cmd), e)

  return score


if __name__ == '__main__':
  if len(sys.argv) != 3:
    sys.exit(f'Usage: {sys.argv[0]} <predicted_model_pdb> <native_structure_pdb>')
  prepdb, natpdb = sys.argv[1:]

  # USalign between predicted and native pdb
  score = DockQ4SinglePair(prepdb, natpdb)
  keys = [
   'PredictedFile', 'ReferenceFile',
   'DockQ', 'iRMSD', 'LRMSD', 'fnat', 'fnonnat', 'F1', 'clashes',
  ]
  print(' '.join(['%s' % score.get(_, 'XXXXX') for _ in keys[:2]]), end=' ')
  print(' '.join(['%10.3f' % score.get(_, 0.) for _ in keys[2:]]))

  print(sys.argv[0], 'Done.')
