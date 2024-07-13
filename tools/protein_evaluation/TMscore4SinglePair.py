# -*- coding: utf-8 -*-
"""TMscore a predicted pdb against a native pdb
"""
import os
import sys

from utils import *


def check_TMscore(name):
    """Check whether `name` is on PATH and marked as executable."""
    from shutil import which
    return which(name) is not None


def TMscore4SinglePair(predicted_pdb, native_pdb):
    """Calculate model score by using TMscore program"""
    # intialization
    score = {'PredictedPDB' : os.path.basename(predicted_pdb),
             'NativePDB' : os.path.basename(native_pdb),
             'PredictedLen' : 0,
             'NativeLen' : 0,
             'AlignLen' : 0,
             'RMSD' : 0.,
             'TMscore' : 0.,
             'MaxSub' : 0.,
             'GDT_TS' : 0.,
             'GDT_HA' : 0.}

    if not check_TMscore('TMscore'):
        print('ERROR: TMscore does not exist in $PATH', file=sys.stderr)
        return score

    if not os.path.exists(predicted_pdb):
        print('ERROR: cannot found predicted model', predicted_pdb, file=sys.stderr)
        return score

    if not os.path.exists(native_pdb):
        print('ERROR: cannot found native structure', native_pdb, file=sys.stderr)
        return score

    # execuate command and get output
    cmds = ['TMscore', predicted_pdb, native_pdb]
    lines = check_output_lines(cmds)

    # parse model score
    for i, l in enumerate(lines):
        cols = l.split()
        if l.startswith('Structure1:') and len(cols) > 3:
            score['PredictedLen'] = int(cols[3])
        elif l.startswith('Structure2:') and len(cols) > 3:
            score['NativeLen'] = int(cols[3])
        elif l.startswith('Number') and len(cols) > 5:
            score['AlignLen'] = int(cols[5])
        elif l.startswith('RMSD') and len(cols) > 5:
            score['RMSD'] = float(cols[5])
        elif l.startswith('TM-score') and len(cols) > 2:
            score['TMscore'] = float(cols[2])
        elif l.startswith('MaxSub') and len(cols) > 1:
            score['MaxSub'] = float(cols[1])
        elif l.startswith('GDT-TS') and len(cols) > 1:
            score['GDT_TS'] = float(cols[1]) * 100
        elif l.startswith('GDT-HA') and len(cols) > 1:
            score['GDT_HA'] = float(cols[1]) * 100
        elif l.startswith('(":"'):
            i += 1
            break
        else:
            continue

    # check data format and TMscore output
    if len(lines) != i + 5 or lines[-1] != '\n':
        print('ERROR: wrong TMscore results', file=sys.stderr)

    return score


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Usage: %s <predicted_model_pdb> <native_structure_pdb>' % sys.argv[0])
    prepdb, natpdb = sys.argv[1:]

     # TMscore between predicted and native pdb
    score = TMscore4SinglePair(prepdb, natpdb)
    keys = ['PredictedPDB', 'NativePDB', 'PredictedLen', 'NativeLen', 'AlignLen',
            'RMSD', 'TMscore', 'MaxSub', 'GDT_TS', 'GDT_HA']
    print(' '.join(['%s' % score.get(_, 'XXXXX') for _ in keys[:2]]), end=' ')
    print(' '.join(['%5d' % score.get(_, 0) for _ in keys[2:5]]), end=' ')
    print(' '.join(['%10.3f' % score.get(_, 0.) for _ in keys[5:]]))

    print(sys.argv[0], 'Done.')
