# -*- coding: utf-8 -*-
# %%
raw_data_path = '/blob/v-yihangzhai/materials_data/mp/materialProject_all_20230717_props_sharedByTian_sg.jsonl'

# %%
import json

# %%
raw_data = []
with open(raw_data_path, 'r') as f:
    for line in f:
        raw_data.append(json.loads(line))

# %%
len(raw_data)

# %%
import random
random.seed(0)

random.shuffle(raw_data)


# %%
train_raw_data = raw_data[:154000]
val_raw_data = raw_data[154000:]

# %%
train_raw_data[0]

# %%
space_group_templates = [
    "This material belongs to the space group {symbol}.",
    "The space group of this material is also known as {symbol}.",
    "With its structure pertaining to the space group {symbol}, the material exhibits unique properties.",
    "This material is part of the space group {symbol}.",
    "The material falls under the {symbol} space group.",
    "Belonging to the space group {symbol}, this material has an interesting structure.",
    "The space group {symbol} houses this material's structure.",
    "In the realm of space groups, this material belongs to {symbol}.",
    "The classification of this material's space group is {symbol}.",
    "Falling into the category of space group {symbol}, the material has a unique structure."
]

formation_energy_templates = [
    "It has a formation energy per atom of {formation_energy:.4f}.",
    "The formation energy per atom for this material is {formation_energy:.4f}.",
    "With a formation energy per atom of {formation_energy:.4f}, the material shows unique characteristics.",
    "{formation_energy:.4f} is the formation energy per atom of this material.",
    "This material exhibits a formation energy per atom of {formation_energy:.4f}.",
    "The material's formation energy per atom measures at {formation_energy:.4f}.",
    "A formation energy per atom of {formation_energy:.4f} is a defining characteristic of this material.",
    "This material's formation energy per atom is recorded at {formation_energy:.4f}.",
    "Notably, the formation energy per atom of this material is {formation_energy:.4f}.",
    "The material's hallmark is its formation energy per atom, which is {formation_energy:.4f}."
]

energy_above_hull_templates = [
    "An energy above hull of {energy_above_hull:.4f} is observed in this material.",
    "This material has an energy above hull of {energy_above_hull:.4f}.",
    "The energy above hull for this material is {energy_above_hull:.4f}.",
    "With an energy above hull measuring at {energy_above_hull:.4f}, this material displays unique properties.",
    "{energy_above_hull:.4f} is the recorded energy above hull for this material.",
    "The energy above hull of this material stands at {energy_above_hull:.4f}.",
    "A noteworthy feature of this material is its energy above hull of {energy_above_hull:.4f}.",
    "This material exhibits an energy above hull of {energy_above_hull:.4f}.",
    "The material possesses an energy above hull measuring {energy_above_hull:.4f}.",
    "An energy above hull of {energy_above_hull:.4f} characterizes this material."
]

band_gap_templates = [
    "The material's band gap is {band_gap:.4f}.",
    "A band gap of {band_gap:.4f} is present in this material.",
    "This material exhibits a band gap of {band_gap:.4f}.",
    "With a band gap of {band_gap:.4f}, the material possesses unique properties.",
    "The band gap for this material is recorded at {band_gap:.4f}.",
    "This material has a band gap measuring {band_gap:.4f}.",
    "This material is characterized by its band gap of {band_gap:.4f}.",
    "The band gap of this material measures at {band_gap:.4f}.",
    "{band_gap:.4f} is the band gap of this material.",
    "A defining feature of this material is its band gap of {band_gap:.4f}."
]

total_magnetization_templates = [
    "It has a total normalized magnetization volume of {total_magnetization:.4g}.",
    "This material's total normalized magnetization volume is {total_magnetization:.4g}.",
    "With a total normalized magnetization volume of {total_magnetization:.4g}, this material exhibits unique magnetic properties.",
    "The total normalized magnetization volume of this material is {total_magnetization:.4g}.",
    "The material has a total normalized magnetization volume of {total_magnetization:.4g}.",
    "{total_magnetization:.4f} is the total normalized magnetization volume of this material.",
    "A total normalized magnetization volume of {total_magnetization:.4g} characterizes this material.",
    "This material is notable for its total normalized magnetization volume of {total_magnetization:.4g}.",
    "The material's total normalized magnetization volume measures at {total_magnetization:.4g}."
]

formula_templates = [
    "The formula for this material is {formula}.",
    "{formula} is the chemical formula of this material.",
    "This material can be represented by the formula {formula}.",
    "The chemical formula of this material is {formula}.",
    "The material's formula is given by {formula}.",
    "This material is denoted by the formula {formula}.",
    "The formula {formula} represents this material.",
    "In terms of its chemical formula, this material is expressed as {formula}.",
    "The constituent elements of this material form the compound {formula}.",
    "The formula {formula} denotes the composition of this material."
]

atom_symbols_templates = [
    "This material is composed of the elements {atom_symbols}.",
    "The {atom_symbols} are the elements that make up this material.",
    "The elements in this material are {atom_symbols}.",
    "This material contains the elements {atom_symbols}.",
    "The composition of this material includes the elements {atom_symbols}.",
    "With a composition of {atom_symbols}, this material has unique properties.",
    "The {atom_symbols} are the constituent elements of this material.",
    "The material is comprised of the elements {atom_symbols}.",
    "The atomic makeup of this material includes {atom_symbols}.",
    "This material consists of the elements {atom_symbols}."
]



# %%
from PyAstronomy import pyasl

# %%
def atom_symbol_to_name(atom_symbol):
    an = pyasl.AtomicNo()
    return an.getElementName(an.getAtomicNo(atom_symbol)).lower()
# %%
import re
def tokenize_formula(formula):
    # Define the regular expression pattern for tokenizing
    pattern = r"([A-Z][a-z]*)(\d*)|(\()|(\))(\d*)|([A-Z][a-z]*)"
    # Use regular expression to find all tokens
    tokens = re.findall(pattern, formula)

    # Flatten the list of tokens and remove empty strings
    tokens = [item for sublist in tokens for item in sublist if item]
    return ' '.join(tokens)

print(tokenize_formula('Mo6PbCl14'))
print(tokenize_formula('HCl'))
print(tokenize_formula('H2O'))
print(tokenize_formula('(Al2(O4)2)2SiO4'))


# %%
def describe_material(data):
    formula = data['formula_pretty'] if 'formula_pretty' in data else None
    formation_energy = data['formation_energy_per_atom'] if 'formation_energy_per_atom' in data else None
    energy_above_hull = data['energy_above_hull'] if 'energy_above_hull' in data else None
    band_gap = data['band_gap'] if 'band_gap' in data else None
    total_magnetization = data['total_magnetization_normalized_vol'] if 'total_magnetization_normalized_vol' in data else None
    space_group = data['structure']['space_group'] if 'structure' in data and 'space_group' in data['structure'] else None

    atom_symbols = [site['species'][0]['element'] for site in data['structure']['sites']]
    atom_symbols = list(set(atom_symbols))  # remove duplicates
    atom_names = [str(atom_symbol_to_name(symbol)) for symbol in atom_symbols]
    if len(atom_names) == 1:
        atom_names_str = atom_names[0]
    else:
        atom_names_str = ', '.join(atom_names[:-1]) + ', and ' + atom_names[-1]

    description = []

    if formation_energy:
        description.append(random.choice(formation_energy_templates).format(formation_energy=formation_energy))
    if energy_above_hull:
        description.append(random.choice(energy_above_hull_templates).format(energy_above_hull=energy_above_hull))
    if band_gap:
        description.append(random.choice(band_gap_templates).format(band_gap=band_gap))
    if total_magnetization:
        description.append(random.choice(total_magnetization_templates).format(total_magnetization=total_magnetization))
    if space_group:
        description.append(random.choice(space_group_templates).format(symbol=space_group['symbol']))
    if atom_names_str:
        description.append(random.choice(atom_symbols_templates).format(atom_symbols=atom_names_str))

    random.shuffle(description)

    if formula:
        sg = f'<sg{space_group["no"]}>' if space_group else ''
        formula = f'<material> {tokenize_formula(formula)} {sg} </material>'
        description.append(random.choice(formula_templates).format(formula=formula))

    random.shuffle(description)

    return ' '.join(description)

describe_material(train_raw_data[0])

# %%
output_folder = '/blob/shufxi/data/scigpt/materials_project_data'
# ensure the folder exists
import os
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

from tqdm import tqdm

fold=10
with open(os.path.join(output_folder, f'train_x{fold}.txt'), 'w') as f:
    for i in range(fold):
        for data in tqdm(train_raw_data, desc=f'train_{i}'):
            f.write(describe_material(data) + '\n')
        print(f'Finished writing fold {i}')

with open(os.path.join(output_folder, f'val{fold}.txt'), 'w') as f:
    for data in tqdm(val_raw_data, desc='val'):
        f.write(describe_material(data) + '\n')

# %%
