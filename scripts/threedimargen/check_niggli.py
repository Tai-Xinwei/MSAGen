# -*- coding: utf-8 -*-
import json
from copy import deepcopy

from pymatgen.core.structure import Structure

fname = "/hai1/SFM/threedimargen/data/materials_data/mp_nomad_qmdb_train.jsonl"


def get_niggli(item):
    new_item = deepcopy(item)
    species = [site["element"] for site in item["sites"]]
    coords = [site["fractional_coordinates"] for site in item["sites"]]
    lattice = item["lattice"]
    crystal = Structure(
        species=species, coords=coords, lattice=lattice, coords_are_cartesian=False
    )
    new_crystal = Structure(
        species=crystal.species,
        coords=crystal.cart_coords,
        lattice=crystal.lattice.get_niggli_reduced_lattice(),
        coords_are_cartesian=True,
    )
    new_lattice = new_crystal.lattice.matrix.tolist()
    new_coords = new_crystal.frac_coords.tolist()
    new_item["lattice"] = new_lattice
    for i in range(len(item["sites"])):
        new_item["sites"][i]["fractional_coordinates"] = new_coords[i]
    return new_item


def check_niggli(fname):
    with open(fname, "r") as f:
        for line in f:
            item = json.loads(line)
            new_item = get_niggli(item)
            if (
                new_item["lattice"] != item["lattice"]
                or new_item["sites"] != item["sites"]
            ):
                print(f"{item}\n{new_item}\n\n")


if __name__ == "__main__":
    check_niggli(fname)
