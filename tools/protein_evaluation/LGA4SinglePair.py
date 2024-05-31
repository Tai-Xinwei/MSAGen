# -*- coding: utf-8 -*-
"""LGA a predicted pdb against a native pdb
"""
import os
import sys

from utils import *




def check_LGA(name):
    """Check whether `name` is on PATH and marked as executable."""
    from shutil import which
    return which(name) is not None


def LGA4SinglePair(predicted_pdb, native_pdb):
    """Calculate model score by using LGA program"""
    # intialization
    score = {'PredictedPDB' : os.path.basename(predicted_pdb),
             'NativePDB' : os.path.basename(native_pdb),
             'GDT_TS' : 0.,
             'GDT_HA' : 0.}
    # check program and files
    if not check_LGA('runlga.mol_mol.pl'):
        print('ERROR: LGA does not exist in $PATH', file=sys.stderr)
        return score
    if not os.path.exists(predicted_pdb):
        print('ERROR: cannot found predicted model', predicted_pdb, file=sys.stderr)
        return score
    if not os.path.exists(native_pdb):
        print('ERROR: cannot found native structure', native_pdb, file=sys.stderr)
        return score
    # get absolute path
    predicted_pdb = os.path.abspath(predicted_pdb)
    native_pdb = os.path.abspath(native_pdb)
    # generated new molecule including two pdbs
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        print('Making temporary directory', tmpdirname)
        # generate results
        current_dir = os.getcwd()
        os.chdir(tmpdirname)
        cmd = 'ulimit -s unlimited; runlga.mol_mol.pl %s %s -3 -sda -o2 -d:4' % (
            predicted_pdb, native_pdb)
        print(cmd)
        os.system(cmd)
        os.chdir(current_dir)
        # check results file
        resfile = '%s/RESULTS/%s.%s.res' % (tmpdirname,
                                            os.path.basename(predicted_pdb),
                                            os.path.basename(native_pdb))
        # extract GDT_TS and GDT_HA
        try:
          with open(resfile, 'r') as fin:
            for line in fin:
              if line.startswith('GDT PERCENT_AT'):
                cols = [float(_) for _ in line.split()[2:]]
                assert len(cols) == 20
                # calculate GDT_TS and GDT_HA
                score['GDT_TS'] = (cols[1] + cols[3] + cols[7] + cols[15]) / 4.0
                score['GDT_HA'] = (cols[0] + cols[1] + cols[3] + cols[7]) / 4.0
                break
            else:
                raise ValueError
        except:
          print('ERROR: wrong GDT results', resfile, file=sys.stderr)

    return score




if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Usage: %s <predicted_model_pdb> <native_structure_pdb>' % sys.argv[0])
    prepdb, natpdb = sys.argv[1:]

     # LGA between predicted and native pdb
    score = LGA4SinglePair(prepdb, natpdb)
    keys = ['PredictedPDB', 'NativePDB', 'GDT_TS', 'GDT_HA']
    print(' '.join(['%s' % score.get(_, 'XXXXX') for _ in keys[:2]]), end=' ')
    print(' '.join(['%10.3f' % score.get(_, 0.) for _ in keys[2:]]))

    print(sys.argv[0], 'Done.')
