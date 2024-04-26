# -*- coding: utf-8 -*-
import argparse
import json
import os
from multiprocessing import Pool

from invcryrep.invcryrep import InvCryRep
from pymatgen.core.structure import Structure
from tqdm import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--data_path",
    type=str,
    default="/hai1/SFM/threedimargen/data/materials_data/mp_nomad_qmdb_train.jsonl",
)
argparser.add_argument("--output_path", type=str, default=None)
args = argparser.parse_args()


def process_data(data_path, output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(data_path), "data_processed.jsonl")

    backend = InvCryRep(graph_method="crystalnn")
    with open(data_path, "r") as fr:
        lines = fr.readlines()
    with open(output_path, "w") as fw:
        for line in tqdm(lines):
            line = json.loads(line)
            species = [site["element"] for site in line["sites"]]
            coords = [site["fractional_coordinates"] for site in line["sites"]]
            structure = Structure(
                lattice=line["lattice"],
                species=species,
                coords=coords,
            )
            # with Pool(1) as p:
            #    res = p.apply_async(backend.structure2SLICES, (structure,))
            #    try:
            #        slices = res.get(timeout=10)
            #    except:
            #        print(f"{structure.formula} cannot be converted")
            #        continue
            try:
                slices = backend.structure2SLICES(structure)
            except:
                print(f"{structure.formula} cannot be converted")
                continue
            slices = slices.split()
            slices.insert(structure.num_sites, "<gen>")
            data = " ".join(slices)
            fw.write(data + "\n")
            fw.flush()
    return


def main():
    data_path = args.data_path
    data_output_path = args.output_path
    process_data(data_path, data_output_path)


if __name__ == "__main__":
    main()
