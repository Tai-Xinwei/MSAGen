# -*- coding: utf-8 -*-
import os
import json
from tqdm import tqdm
import fire
import openai


def table_to_sequence(data):
    formula = data["formula_pretty"]
    structure = data["structure"]
    space_group_no = structure["space_group"]["no"]
    space_group_symbol = structure["space_group"]["symbol"]
    formation_energy_per_atom = data["formation_energy_per_atom"]
    energy_above_hull = data["energy_above_hull"]
    band_gap = data["band_gap"]
    text = (f"The material with the chemical formula {formula} has a crystal structure belonging to the space group {space_group_symbol}."
            f" Its formation energy is {formation_energy_per_atom} eV/atom and its energy above hull is {energy_above_hull} eV/atom. Its band gap is {band_gap} eV.")
    return text

def main(
    data="/blob/v-yihangzhai/materials_data/mp/materialProject_all_20230717_props_sharedByTian_sg.jsonl",
    output="/blob/renqian/sfm_gen/data/mp/sequence.txt",
):
    with open(data) as f:
        dataset = [json.loads(line) for line in f]
    with open(output, "w") as f:
        for item in tqdm(dataset):
            seq = table_to_sequence(item)
            f.write(seq + "\n")

if __name__ == "__main__":
    fire.Fire(main)
