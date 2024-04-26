# -*- coding: utf-8 -*-
import json
import os

import numpy as np


def convert_meter_to_angstrom(coordinates):
    return [[x * 1e10 for x in coordinate] for coordinate in coordinates]


def normalize_coordinates(coordinates):
    ret = []
    for x in coordinates:
        if x < 0:
            x = x + abs(int(x)) + 1
        if x > 1:
            x = x - int(x)
        ret.append(round(x, 6))
    return ret


def calculate_fractional_coordinates(lattice, cartesian_coordinates):
    """
    The lattice matrix is in the form [[a1, a2, a3], [b1, b2, b3], [c1, c2,c3]], each row represents one vector of the lattice
    So we need to transpose it to get the form [[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]], each row represents one dimension of all the lattice vectors
    Calculate:
        lattcie.T * fractional_coordinates = cartesian_coordinates
        fractional_coordinates = inverse(lattice.T) * cartesian_coordinates
    """
    lattice = np.array(lattice)
    transposed_lattice = np.transpose(lattice)
    inverse_lattice_matrix = np.linalg.inv(transposed_lattice)
    cartesian_coordinates = np.array(cartesian_coordinates)
    fractional_coordinates = np.dot(inverse_lattice_matrix, cartesian_coordinates)
    return fractional_coordinates.tolist()


def process_nomad(data_path, output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(data_path), "nomad_processed.jsonl")

    with open(output_path, "w") as fw:
        with open(data_path, "r") as fr:
            for i, line in enumerate(fr):
                line = json.loads(line)
                formula = line["archive"]["results"]["material"][
                    "chemical_formula_reduced"
                ]
                lattice = line["archive"]["results"]["properties"]["structures"][
                    "structure_original"
                ]["lattice_vectors"]
                lattice = convert_meter_to_angstrom(lattice)
                flattend_sites = line["archive"]["results"]["properties"]["structures"][
                    "structure_original"
                ]["species_at_sites"]
                cartesian_site_positions = line["archive"]["results"]["properties"][
                    "structures"
                ]["structure_original"]["cartesian_site_positions"]
                cartesian_site_positions = convert_meter_to_angstrom(
                    cartesian_site_positions
                )
                space_group = line["space_group"]
                sites = []
                for j, site in enumerate(flattend_sites):
                    sites.append(
                        {
                            "element": site,
                            "fractional_coordinates": normalize_coordinates(
                                calculate_fractional_coordinates(
                                    lattice, cartesian_site_positions[j]
                                )
                            ),
                            "cartesian_coordinates": cartesian_site_positions[j],
                        }
                    )

                data = {
                    "source": "nomad",
                    "id": i + 1,
                    "formula": formula,
                    "lattice": lattice,
                    "sites": sites,
                    "space_group": space_group,
                }
                fw.write(json.dumps(data) + "\n")
    return


def main():
    nomad_data_path = "/blob/v-yihangzhai/materials_data/nomad/materials/all_sg.jsonl"
    nomad_data_output_path = "/hai1/SFM/threedimargen/data/materials_data/nomad.jsonl"
    process_nomad(nomad_data_path, nomad_data_output_path)
    return


if __name__ == "__main__":
    main()
