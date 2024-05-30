# -*- coding: utf-8 -*-
import os
import sys
import pickle
import lmdb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])
from sfm.logging import logger
from sfm.data.prot_data.util import bstr2obj, obj2bstr

def main():
    write_file = '/home/peiranjin/output/sample_result/cameo-from-20220401-to-20220625.lmdb'
    write_env = lmdb.open(write_file, map_size=1536 ** 4)
    write_txn = write_env.begin(write=True)
    keys = []
    sizes = []
    with open('/home/peiranjin/output/sample_result/cameo-from-20220401-to-20220625.fasta', 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            if line[0] == '>':
                id = line[1:].split("_")[0]
            elif line[0] != '>':
                seq = line.strip()
                write_txn.put(f"{id}".encode(), obj2bstr(seq))

                keys.append(id)
                sizes.append(len(seq))
            else:
                raise ValueError(f"Invalid line: {line}")

    metadata = {}
    metadata['keys'] = keys
    metadata['sizes'] = sizes

    write_txn.put("__metadata__".encode(), obj2bstr(metadata))
    write_txn.commit()


if __name__ == "__main__":
    main()
