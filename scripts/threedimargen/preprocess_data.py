# -*- coding: utf-8 -*-
import json
import os

import numpy as np


def process_mp(data_path, output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(data_path), "mp_processed.jsonl")

    def flatten_sites(sites):
        return [site["element"] for site in sites]

    with open(output_path, "w") as fw:
        with open(data_path, "r") as fr:
            for i, line in enumerate(fr):
                line = json.loads(line)
                formula = line["formula_pretty"]
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
                flatten_sites(sites)
                data = {
                    "source": "mp",
                    "id": i + 1,
                    "formula": formula,
                    # "composition": flattend_sites,
                    "lattice": lattice,
                    "sites": sites,
                    "space_group": space_group,
                }
                fw.write(json.dumps(data) + "\n")
    return


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


def process_nomad(data_path, output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(data_path), "nomad_processed.jsonl")

    with open(output_path, "w") as fw:
        with open(data_path, "r") as fr:
            for line in fr:
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
                for i, site in enumerate(flattend_sites):
                    sites.append(
                        {
                            "element": site,
                            "fractional_coordinates": normalize_coordinates(
                                calculate_fractional_coordinates(
                                    lattice, cartesian_site_positions[i]
                                )
                            ),
                            "cartesian_coordinates": cartesian_site_positions[i],
                        }
                    )

                data = {
                    "source": "nomad",
                    "id": i + 1,
                    "formula": formula,
                    # "composition": flattend_sites,
                    "lattice": lattice,
                    "sites": sites,
                    "space_group": space_group,
                }
                fw.write(json.dumps(data) + "\n")

    return


def deduplicate(data_path, output_path=None):
    cache = dict()
    with open(output_path, "w") as fw:
        with open(data_path, "r") as fr:
            for line in fr:
                d = json.loads(line)
                key = (
                    "".join([site["element"] for site in d["sites"]])
                    + "\t"
                    + str(d["space_group"]["no"])
                )
                if key not in cache:
                    cache[key] = True
                    fw.write(json.dumps(d) + "\n")


def main():
    # mp_data_path = "/blob/v-yihangzhai/materials_data/mp/materialProject_all_20230717_props_sharedByTian_sg.jsonl"
    # mp_data_output_path = "/hai1/SFM/threedimargen/data/materials_data/mp.jsonl"
    # nomad_data_path = "/blob/v-yihangzhai/materials_data/nomad/materials/all_sg.jsonl"
    # nomad_data_output_path = "/hai1/SFM/threedimargen/data/materials_data/nomad.jsonl"
    # process_mp(mp_data_path, mp_data_output_path)
    # process_nomad(nomad_data_path, nomad_data_output_path)
    # deduplicate(
    #    mp_data_output_path,
    #    "/hai1/SFM/threedimargen/data/materials_data/mp_dedup.jsonl",
    # )
    # deduplicate(
    #    nomad_data_output_path,
    #    "/hai1/SFM/threedimargen/data/materials_data/nomad_dedup.jsonl",
    # )
    deduplicate(
        "/hai1/SFM/threedimargen/data/materials_data/mp_nomad.jsonl",
        "/hai1/SFM/threedimargen/data/materials_data/mp_nomad_dedup.jsonl",
    )
    return


if __name__ == "__main__":
    main()
