# -*- coding: utf-8 -*-
'''
utils.py
Copyright
Author: zhujianwei@ict.ac.cn (Jianwei Zhu)

This module provides utility functions that are used within program
that are also useful for external consumption.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
import os
import pickle
import subprocess
import sys
import tempfile
import zlib


def check_output_file(command, filename):
    '''Exculate a command and return output to a file.'''

    print(' '.join(command))

    # open file and write the output to this file
    with open(filename,'w') as tmp:
        proc = subprocess.Popen(command, stdout=tmp, stderr=tmp)

        return proc.wait()


def check_output_stdout(command):
    '''Exculate a command and return output to stdout.'''

    print(' '.join(command))

    # write the output to stdout
    proc = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stdout)

    return proc.wait()


def check_output_lines(command):
    '''Exculate a command and return output to a list.'''

    # open a temporary file and write the output to this file
    lines = None
    with tempfile.TemporaryFile() as tmp:
        proc = subprocess.Popen(command, stdout=tmp, stderr=tmp)
        proc.wait()

        tmp.seek(0)
        lines = [_.decode('utf-8') for _ in tmp.readlines()]
    return lines


def parse_listfile(listfile, col_list=None):
    '''Parse list file from columns list.'''

    lines = []
    try:
        with open(listfile, 'r') as fin:
            if col_list:
                for line in fin:
                    cols = line.split()
                    lines.append(tuple(cols[i-1] for i in col_list))
            else:
                for line in fin:
                    lines.append(tuple(line.split()))

    except Exception as e:
        print('ERROR: wrong list file "%s"\n      ' % listfile, e, file=sys.stderr)

    return lines


def parse_fastafile(fastafile):
    '''Parse fasta file.'''

    seqs = []
    try:
        with open(fastafile, 'r') as fin:
            header, seq = '', []
            for line in fin:
                if line[0] == '>':
                    seqs.append( (header, ''.join(seq)) )
                    header, seq = line.strip(), []
                else:
                    seq.append( line.strip() )
            seqs.append( (header, ''.join(seq)) )
            del seqs[0]

    except Exception as e:
        print('ERROR: wrong fasta file "%s"\n      ' % fastafile, e, file=sys.stderr)

    return seqs


def parse_protein_id(filename):
    '''
    Parse protein name from path name
    filename = "/tmp/d1a3aa_" --> d1a3aa_
    filename = "/tmp/d1a3aa_.fasta" --> d1a3aa_
    '''

    base = os.path.basename(filename)
    protein_id = os.path.splitext(base)[0]
    #name = filename.split('/')[-1]
    #protein_id = '.'.join(name.split('.')[:-1]) or name

    return protein_id


def check_outdir(outdir):
    '''Check output directory. If it is not exist, create it'''

    if not os.path.exists(outdir):
        print('Output directory create %s' % outdir)
        os.makedirs(outdir)
    else:
        print('Output directory exists %s' % outdir)


def accumulate(iterable, func=operator.add):
    '''
    Return running totals
    accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    '''

    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total


def obj2bstr(obj):
    return zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def bstr2obj(bstr: bytes):
    return pickle.loads(zlib.decompress(bstr))
