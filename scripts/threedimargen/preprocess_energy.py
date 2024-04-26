# -*- coding: utf-8 -*-
import json
import os
import re

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


def process_data(filename, output_filename):
    with open(filename, "r") as fr:
        lines = fr.readlines()

    source = filename.split("/")[-1].split(".")[0]
    with open(output_filename, "w") as fw:
        i = 0
        count = 1
        while i < len(lines):
            item = {}
            item["source"] = source
            item["id"] = count
            count += 1

            num_atoms = int(lines[i].strip())
            item["num_atoms"] = num_atoms

            i += 1
            properties = lines[i].strip()
            lattice = re.search('Lattice="(.*?)"', properties)[1]
            lattice = list(map(float, lattice.split(" ")))
            lattice = [lattice[:3], lattice[3:6], lattice[6:]]
            item["lattice"] = lattice

            energy = re.search(" energy=(.*?) ", properties)[1]
            energy = float(energy)
            item["energy"] = energy

            i += 1
            item["sites"] = []
            for _ in range(num_atoms):
                site_data = re.split("\s+", lines[i].strip())
                element = site_data[0]
                cart_coords = list(map(float, site_data[1:4]))
                site = {
                    "element": element,
                    "cartesian_coordinates": cart_coords,
                    "fractional_coordinates": normalize_coordinates(
                        calculate_fractional_coordinates(lattice, cart_coords)
                    ),
                }
                item["sites"].append(site)
                i += 1

            fw.write(json.dumps(item) + "\n")
    return


def process_all_data(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".xyz"):
            process_data(
                os.path.join(folder, filename),
                os.path.join(folder, filename.replace(".xyz", ".jsonl")),
            )


if __name__ == "__main__":
    process_all_data("/hai1/SFM/threedimargen/data/materials_data/energy")
