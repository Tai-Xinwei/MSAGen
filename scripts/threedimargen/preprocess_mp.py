# -*- coding: utf-8 -*-
import json
import os

import numpy as np


def normalize_coordinates(coordinates):
    ret = []
    for x in coordinates:
        if x < 0:
            x = x + abs(int(x)) + 1
        if x > 1:
            x = x - int(x)
        ret.append(round(x, 6))
    return ret


def process_mp(data_path, output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(data_path), "mp_processed.jsonl")

    with open(output_path, "w") as fw:
        with open(data_path, "r") as fr:
            for i, line in enumerate(fr):
                line = json.loads(line)
                formula = line["formula_pretty"]
                material_id = line["material_id"]
                lattice = line["structure"]["lattice"]["matrix"]
                space_group = line["structure"]["space_group"]
                sites = [
                    {
                        "element": site["species"][0]["element"],
                        "fractional_coordinates": normalize_coordinates(site["abc"]),
                        "cartesian_coordinates": site["xyz"],
                    }
                    for site in line["structure"]["sites"]
                ]
                # flatten_sites(sites)
                data = {
                    "source": "mp",
                    "id": i + 1,
                    "material_id": material_id,
                    "formula": formula,
                    "lattice": lattice,
                    "sites": sites,
                    "space_group": space_group,
                }
                fw.write(json.dumps(data) + "\n")
    return


def main():
    mp_data_path = "/blob/v-yihangzhai/materials_data/mp/materialProject_all_20230717_props_sharedByTian_sg.jsonl"
    mp_data_output_path = "/hai1/SFM/threedimargen/data/materials_data/mp.jsonl"
    process_mp(mp_data_path, mp_data_output_path)


if __name__ == "__main__":
    main()
