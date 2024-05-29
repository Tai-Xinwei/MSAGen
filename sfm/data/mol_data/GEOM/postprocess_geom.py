# -*- coding: utf-8 -*-
import sys

import lmdb
from ogb.utils import smiles2graph
from tqdm import tqdm

from sfm.data.mol_data.utils.lmdb import bstr2obj, obj2bstr


def postprocess_dataset(input_path, output_path):
    graphs = []

    env = lmdb.open(input_path, map_size=8 * 1024**4)
    with env.begin() as txn:
        try:
            metadata = bstr2obj(txn.get(b"__metadata__"))
            lmdb_keys = metadata["keys"]
        except TypeError:
            lmdb_keys = list(txn.cursor().iternext(values=False))

        keys = []
        size = []

        for key in tqdm(lmdb_keys):
            data = bstr2obj(txn.get(key))

            try:
                graph = smiles2graph(data["smiles"])
            except AttributeError:
                continue
            graph.update({"mol": data["mol"]})

            keys.append(key.decode())
            size.append(graph["num_nodes"])

            graphs.append(graph)

    env.close()

    metadata = {"keys": keys, "size": size}

    env = lmdb.open(output_path, map_size=8 * 1024**4)
    with env.begin(write=True) as txn:
        txn.put("metadata".encode(), obj2bstr(metadata))

        for key, graph in zip(keys, graphs):
            txn.put(key.encode(), obj2bstr(graph))

    env.close()


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    postprocess_dataset(input_path, output_path)
