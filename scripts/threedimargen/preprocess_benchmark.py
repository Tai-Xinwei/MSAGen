# -*- coding: utf-8 -*-
import json
import os

import numpy as np
import pandas as pd
from mp_time_split.core import MPTimeSplit
from pymatgen.io.cif import CifParser


def process_csv(data_path, output_path, source):
    if output_path is None:
        output_path = data_path.replace(".csv", ".jsonl")

    with open(output_path, "w") as fw:
        df = pd.read_csv(data_path)
        for i, row in df.iterrows():
            id = row[0] + 1
            material_id = row["material_id"]
            parser = CifParser.from_str(row["cif"])
            structure = parser.get_structures()[0]
            lattice = structure.lattice.matrix.tolist()
            space_group = structure.get_space_group_info()
            formula = structure.formula
            sites = []
            for site in structure.sites:
                sites.append(
                    {
                        "element": site.species_string,
                        "fractional_coordinates": site.frac_coords.tolist(),
                        "cartesian_coordinates": site.coords.tolist(),
                    }
                )
            data = {
                "source": source,
                "id": id,
                "material_id": material_id,
                "formula": formula,
                "lattice": lattice,
                "sites": sites,
                "space_group": {
                    "no": space_group[1],
                    "symbol": space_group[0],
                },
            }
            fw.write(json.dumps(data) + "\n")


def process_mpts52(output_path, source="mpts-52"):
    mpt = MPTimeSplit(target="energy_above_hull")
    mpt.load(dummy=False)

    train_split, test_split = mpt.test_split

    def process_split(split, split_name):
        with open(output_path.replace(".jsonl", f"_{split_name}.jsonl"), "w") as fw:
            for i in split:
                material_id = mpt.data.iloc[i].material_id
                structure = mpt.data.iloc[i].structure
                lattice = structure.lattice.matrix.tolist()
                space_group = structure.get_space_group_info()
                formula = structure.formula
                sites = []
                for site in structure.sites:
                    sites.append(
                        {
                            "element": site.species_string,
                            "fractional_coordinates": site.frac_coords.tolist(),
                            "cartesian_coordinates": site.coords.tolist(),
                        }
                    )
                data = {
                    "source": source,
                    "id": int(i + 1),
                    "material_id": material_id,
                    "formula": formula,
                    "lattice": lattice,
                    "sites": sites,
                    "space_group": {
                        "no": space_group[1],
                        "symbol": space_group[0],
                    },
                }
                fw.write(json.dumps(data) + "\n")

    process_split(train_split, "train")
    process_split(test_split, "test")


def main():
    # for dataset in ["carbon_24", "perov_5", "mp_20"]:
    #     for split in ['train', 'val', 'test']:
    #         data_path = f"/hai1/SFM/threedimargen/data/materials_data/{dataset}/{split}.csv"
    #         data_output_path = f"/hai1/SFM/threedimargen/data/materials_data/{dataset}_{split}.jsonl"
    #         process_csv(data_path, data_output_path, dataset)
    process_mpts52("/hai1/SFM/threedimargen/data/materials_data/mpts-52.jsonl")


if __name__ == "__main__":
    main()
