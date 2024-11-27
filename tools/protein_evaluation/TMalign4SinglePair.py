# -*- coding: utf-8 -*-
"""TMalign a predicted pdb against a native pdb
"""
import os
import sys

from utils import *


def check_TMalign(name):
    """Check whether `name` is on PATH and marked as executable."""
    from shutil import which
    return which(name) is not None


def TMalign4SinglePair(predicted_pdb, native_pdb):
    """Calculate model score by using TMalign program"""
    # intialization
    score = {
        'PredictedPDB' : os.path.basename(predicted_pdb),
        'NativePDB' : os.path.basename(native_pdb),
        'PredictedLen' : 0,
        'NativeLen' : 0,
        'AlignLen' : 0,
        'RMSD' : 0.,
        'SeqID' : 0.,
        'TMscoreNormByPredicted' : 0.,
        'TMscoreNormByNative' : 0.,
    }

    if not check_TMalign('TMalign'):
        print('ERROR: TMalign does not exist in $PATH', file=sys.stderr)
        return score

    if not os.path.exists(predicted_pdb):
        print('ERROR: cannot found predicted model', predicted_pdb, file=sys.stderr)
        return score

    if not os.path.exists(native_pdb):
        print('ERROR: cannot found native structure', native_pdb, file=sys.stderr)
        return score

    # execuate command and get output
    cmds = ['TMalign', predicted_pdb, native_pdb]
    lines = check_output_lines(cmds)

    # parse model score
    for i, l in enumerate(lines):
        cols = l.split()
        if l.startswith('Length of Chain_1:') and len(cols) > 4:
            score['PredictedLen'] = int(cols[3])
        elif l.startswith('Length of Chain_2:') and len(cols) > 4:
            score['NativeLen'] = int(cols[3])
        elif l.startswith('Aligned length=') and len(cols) > 6:
            score['AlignLen'] = int(cols[2].strip(','))
            score['RMSD'] = float(cols[4].strip(','))
            score['SeqID'] = float(cols[6].strip(','))
        elif l.startswith('TM-score=') and l.endswith('Chain_1)\n'):
            score['TMscoreNormByPredicted'] = float(cols[1])
        elif l.startswith('TM-score=') and l.endswith('Chain_2)\n'):
            score['TMscoreNormByNative'] = float(cols[1])
        elif l.startswith('(":"'):
            i += 1
            break
        else:
            continue

    # check data format and TMalign output
    if len(lines) != i + 4 or lines[-1] != '\n':
        print('ERROR: wrong TMalign results', file=sys.stderr)

    return score


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Usage: %s <predicted_model_pdb> <native_structure_pdb>' % sys.argv[0])
    prepdb, natpdb = sys.argv[1:]

     # TMalign between predicted and native pdb
    score = TMalign4SinglePair(prepdb, natpdb)
    keys = ['PredictedPDB', 'NativePDB', 'PredictedLen', 'NativeLen', 'AlignLen',
            'RMSD', 'SeqID', 'TMscoreNormByPredicted', 'TMscoreNormByNative']
    print(' '.join(['%s' % score.get(_, 'XXXXX') for _ in keys[:2]]), end=' ')
    print(' '.join(['%5d' % score.get(_, 0) for _ in keys[2:5]]), end=' ')
    print(' '.join(['%10.3f' % score.get(_, 0.) for _ in keys[5:]]))

    print(sys.argv[0], 'Done.')
