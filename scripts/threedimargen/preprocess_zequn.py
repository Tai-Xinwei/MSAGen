# -*- coding: utf-8 -*-
import json
import os
import re

import numpy as np

cache = {}


def process_mp(data_path, output_path=None):
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(data_path), "mtgen_hit_correct_full.jsonl"
        )

    with open(output_path, "w") as fw:
        with open(data_path, "r") as fr:
            for i, line in enumerate(fr):
                line = json.loads(line)
                output = line["output"]
                matches = re.findall(r"<i>(\w+)", output)
                atoms = []
                formula = []
                # Iterate over the matches in pairs (element, number)
                for j in range(0, len(matches), 2):
                    element = matches[j]
                    number = int(matches[j + 1])
                    # Multiply the element by its number and add the result to the list
                    formula.append(element)
                    if number > 1:
                        formula.append(str(number))
                    atoms.extend([element] * number)
                formula = "".join(formula)
                lattice = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

                match = re.findall(r"<sg(\d+)>", output)
                space_group = int(match[0])
                cache_key = formula + "<sg" + str(space_group) + ">"
                if cache_key in cache:
                    continue
                cache[cache_key] = 1
                sites = [
                    {
                        "element": atom,
                        "fractional_coordinates": [0, 0, 0],
                        "cartesian_coordinates": [0, 0, 0],
                    }
                    for atom in atoms
                ]
                data = {
                    "source": "mtgen_hit_correct_full",
                    "id": i + 1,
                    "formula": formula,
                    "lattice": lattice,
                    "sites": sites,
                    "space_group": {"no": space_group},
                }
                fw.write(json.dumps(data) + "\n")
    return


def main():
    data_path = "/hai1/SFM/threedimargen/data/materials_data/mtgen_hit_correct_full.txt"
    data_output_path = (
        "/hai1/SFM/threedimargen/data/materials_data/mtgen_hit_correct_full.jsonl"
    )
    process_mp(data_path, data_output_path)


if __name__ == "__main__":
    main()
