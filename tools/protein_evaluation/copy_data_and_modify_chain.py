#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path


def find_source_pdb(target, srcdir, num):
    srcdir = Path(srcdir)

    srcpdb = srcdir / target / f'ranked_{num-1}.pdb'
    if srcpdb.exists():
        return srcpdb

    srcpdb = srcdir / f'{target}-{num}.pdb'
    if srcpdb.exists():
        return srcpdb

    return None


def get_destnation_pdb(target, dstdir, server, num):
    dstdir = Path(dstdir)

    if target[0] == 'T':
        outdir = dstdir / target
        dstpdb = outdir / f'{target}TS{server}_{num}'
    else:
        outdir = dstdir / target / 'servers' / f'server{server}'
        dstpdb = outdir / f'model-{num}' / f'model-{num}.pdb'

    return dstpdb


def copy_and_modify(srcpdb, dstpdb, chain):
    lines = []
    with open(srcpdb, 'r') as fp:
        lines = fp.readlines()

    for i, line in enumerate(lines):
        if len(line) != 81:
            continue
        if line.startswith('ATOM  ') or line.startswith('HETATM'):
            lines[i] = line[:21] + chain + line[22:]

    os.makedirs(dstpdb.parent, exist_ok=True)
    with open(dstpdb, 'w') as fp:
        fp.writelines(lines)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(f'Usage: {sys.argv[0]} <source_directory> <output_root_directory> <server_number>')
    srcdir, rootdir, server = sys.argv[1:4]

    # check output directory
    caspdir = Path(rootdir) / 'casp-official-targets.prediction'
    cameodir = Path(rootdir) / 'cameo-official-targets.prediction'
    if not caspdir.exists() or not cameodir.exists():
        raise FileNotFoundError(f"Directory {caspdir} or {cameodir} not found.")

    # get target names
    targets = [_.name for _ in caspdir.iterdir() if _.is_dir()]
    targets += [_.name for _ in cameodir.iterdir() if _.is_dir()]
    print(f'Number of targets: {len(targets)}')

    # processing target prediction one by one
    for target in sorted(targets):
        if len(target) == 6 and target[4] == '_':
            tarname, chain = target[:4], target[5]
        elif target[0] == 'T':
            tarname, chain = target, ' '
        else:
            raise ValueError(f'ERROR: wrong target name {target}.')
        print(f"{target} name={tarname} chain='{chain}'.")

        for idx in range(1):
            model_num = idx + 1

            srcdir = Path(srcdir)
            srcpdb = find_source_pdb(target, srcdir, model_num)
            if srcpdb is None:
                print(f"ERROR: {target}-{model_num} does not exist in {srcdir}.")
                continue
            #print(srcpdb)

            dstdir = caspdir if target[0] == 'T' else cameodir
            dstpdb = get_destnation_pdb(target, dstdir, server, model_num)
            #print(dstpdb)
            print(f'cp {srcpdb} {dstpdb}')

            copy_and_modify(srcpdb, dstpdb, chain)
