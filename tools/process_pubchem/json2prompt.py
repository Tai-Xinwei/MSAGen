# -*- coding: utf-8 -*-
import gzip
import os
import json
import multiprocessing

Question_list = {
    "molecular_formula": [
        "The molecular formula for this molecule <<|mol0|>> is <<|string0|>>.",
        "For this specific molecule <<|mol0|>>, the molecular formula is <<|string0|>>.",
        "In the case of molecule <<|mol0|>>, the appropriate molecular formula would be <<|string0|>>.",
        "For the particular molecule <<|mol0|>>, the corresponding molecular formula is <<|string0|>>.",
        "This molecule, <<|mol0|>>, has a molecular formula represented as <<|string0|>>.",
        "Expressing molecule <<|mol0|>> in terms of its molecular formula yields <<|string0|>>.",
        "Molecule <<|mol0|>> has a molecular formula of <<|string0|>>.",
        "The molecular composition of molecule <<|mol0|>> is represented by the formula <<|string0|>>.",
        "Describing molecule <<|mol0|>> in the context of molecular formula, we get <<|string0|>>.",
        "The molecular formula for <<|mol0|>> is <<|string0|>>.",
        "Molecule <<|mol0|>> is associated with the molecular formula <<|string0|>>.",
    ],
    "molecular_weight": [
        "The molecular weight for this molecule <<|mol0|>> is <<|num0|>>.",
        "For this specific molecule <<|mol0|>>, the molecular weight is <<|num0|>>.",
        "In the case of molecule <<|mol0|>>, the appropriate molecular weight would be <<|num0|>>.",
        "For the particular molecule <<|mol0|>>, the corresponding molecular weight is <<|num0|>>.",
        "This molecule, <<|mol0|>>, has a molecular weight represented as <<|num0|>>.",
        "Expressing molecule <<|mol0|>> in terms of its molecular weight yields <<|num0|>>.",
        "Molecule <<|mol0|>> has a molecular weight of <<|num0|>>.",
        "The molecular mass of molecule <<|mol0|>> is represented by the value <<|num0|>>.",
        "Describing molecule <<|mol0|>> in the context of molecular weight, we get <<|num0|>>.",
        "The molecular weight for <<|mol0|>> is <<|num0|>>.",
        "Molecule <<|mol0|>> is associated with the molecular weight <<|num0|>>.",
    ],
    "exact_mass": [
        "The precise mass of this particular molecule <<|mol0|>> is <<|num0|>>.",
        "For this specific molecule <<|mol0|>>, the precise mass is <<|num0|>>.",
        "In the case of molecule <<|mol0|>>, the appropriate precise mass would be <<|num0|>>.",
        "For the particular molecule <<|mol0|>>, the corresponding precise mass is <<|num0|>>.",
        "This molecule, <<|mol0|>>, has a precise mass represented as <<|num0|>>.",
        "Expressing molecule <<|mol0|>> in terms of its precise mass yields <<|num0|>>.",
        "Molecule <<|mol0|>> has a precise mass of <<|num0|>>.",
        "The exact mass of molecule <<|mol0|>> is represented by the value <<|num0|>>.",
        "Describing molecule <<|mol0|>> in the context of precise mass, we get <<|num0|>>.",
        "The precise mass for <<|mol0|>> is <<|num0|>>.",
        "Molecule <<|mol0|>> is associated with the precise mass <<|num0|>>.",
    ],
    "total_charge": [
        "The overall charge of this particular molecule <<|mol0|>> is <<|num0|>>.",
        "For this specific molecule <<|mol0|>>, the overall charge is <<|num0|>>.",
        "In the case of molecule <<|mol0|>>, the appropriate overall charge would be <<|num0|>>.",
        "For the particular molecule <<|mol0|>>, the corresponding overall charge is <<|num0|>>.",
        "This molecule, <<|mol0|>>, has an overall charge represented as <<|num0|>>.",
        "Expressing molecule <<|mol0|>> in terms of its overall charge yields <<|num0|>>.",
        "Molecule <<|mol0|>> has an overall charge of <<|num0|>>.",
        "The total charge of molecule <<|mol0|>> is represented by the value <<|num0|>>.",
        "Describing molecule <<|mol0|>> in the context of overall charge, we get <<|num0|>>.",
        "The overall charge for <<|mol0|>> is <<|num0|>>.",
        "Molecule <<|mol0|>> is associated with the overall charge <<|num0|>>.",
    ],
    "cas_name": [
        "The International Union of Pure and Applied Chemistry (IUPAC) name for this molecule <<|mol0|>> is <<|string0|>>.",
        "For this specific molecule <<|mol0|>>, the IUPAC name is <<|string0|>>.",
        "In the case of molecule <<|mol0|>>, the appropriate IUPAC name would be <<|string0|>>.",
        "For the particular molecule <<|mol0|>>, the corresponding IUPAC name is <<|string0|>>.",
        "This molecule, <<|mol0|>>, has an IUPAC name represented as <<|string0|>>.",
        "Expressing molecule <<|mol0|>> in terms of its IUPAC name yields <<|string0|>>.",
        "Molecule <<|mol0|>> has an IUPAC name of <<|string0|>>.",
        "The IUPAC nomenclature of molecule <<|mol0|>> is represented by the name <<|string0|>>.",
        "Describing molecule <<|mol0|>> in the context of IUPAC name, we get <<|string0|>>.",
        "The IUPAC name for <<|mol0|>> is <<|string0|>>.",
        "Molecule <<|mol0|>> is associated with the IUPAC name <<|string0|>>.",
    ],
    "xlogp3_aa": [
        "xLogP3-AA, a specific algorithm used to calculate the LogP value for a compound, is based on the atom-additive (AA) method, which considers the contributions of individual atoms or functional groups in a molecule to its overall hydrophobicity. The algorithm employs a set of predefined atom-types and their associated hydrophobicity values to estimate the compound's LogP value. This method is widely used in computational chemistry, drug design, and environmental studies to predict the behavior of chemical compounds in various biological and environmental contexts. For molecule <<|mol0|>>, the xLogP3-AA value is <<|num0|>>.",
        "The xLogP3-AA algorithm calculates a compound's LogP value using the atom-additive (AA) method, which accounts for the hydrophobicity contributions of individual atoms or functional groups within a molecule. This widely-used method employs predefined atom-types and their hydrophobicity values to estimate the LogP value, aiding in the prediction of compound behavior in various biological and environmental contexts. The calculated xLogP3-AA value for molecule <<|mol0|>> is <<|num0|>>.",
        "Employing the atom-additive (AA) method, the xLogP3-AA algorithm estimates a compound's LogP value by considering the hydrophobicity contributions of individual atoms or functional groups in a molecule. Utilizing predefined atom-types and their associated hydrophobicity values, this method is commonly used in computational chemistry, drug design, and environmental studies to anticipate the behavior of chemical compounds in a range of biological and environmental scenarios. In the case of molecule <<|mol0|>>, its xLogP3-AA value is <<|num0|>>.",
        "xLogP3-AA is an algorithm that calculates the LogP value of a compound based on the atom-additive (AA) method. This approach takes into account the hydrophobicity contributions of individual atoms or functional groups in a molecule, using predefined atom-types and their associated hydrophobicity values. The xLogP3-AA method is widely applied in fields such as computational chemistry, drug design, and environmental studies to predict the behavior of chemical compounds in various biological and environmental settings. Molecule <<|mol0|>> has an xLogP3-AA value of <<|num0|>>.",
        "Utilizing the atom-additive (AA) method, the xLogP3-AA algorithm calculates a compound's LogP value by considering the individual hydrophobicity contributions of atoms or functional groups in a molecule. The algorithm relies on a set of predefined atom-types and their hydrophobicity values to estimate the LogP value. This method is extensively used in computational chemistry, drug design, and environmental studies for predicting the behavior of chemical compounds in diverse biological and environmental contexts. The xLogP3-AA value for molecule <<|mol0|>> stands at <<|num0|>>.",
        "Based on the atom-additive (AA) method, the xLogP3-AA algorithm calculates a compound's LogP value by taking into account the hydrophobicity contributions of individual atoms or functional groups within a molecule. Using predefined atom-types and their associated hydrophobicity values, this method plays a significant role in computational chemistry, drug design, and environmental studies, predicting the behavior of chemical compounds in various biological and environmental situations. For molecule <<|mol0|>>, the resulting xLogP3-AA value is <<|num0|>>.",
        "The xLogP3-AA algorithm, which relies on the atom-additive (AA) method, calculates a compound's LogP value by considering the hydrophobicity contributions of individual atoms or functional groups present in a molecule. The algorithm uses a set of predefined atom-types and their hydrophobicity values to estimate the LogP value, making it an essential tool in computational chemistry, drug design, and environmental studies for predicting the behavior of chemical compounds in different biological and environmental contexts. The xLogP3-AA value for molecule <<|mol0|>> has been determined to be <<|num0|>>.",
        "Using the atom-additive (AA) method, the xLogP3-AA algorithm calculates a compound's LogP value by accounting for the hydrophobicity contributions of individual atoms or functional groups in a molecule. Employing predefined atom-types and their associated hydrophobicity values, this method has gained widespread use in computational chemistry, drug design, and environmental studies for predicting the behavior of chemical compounds in a variety of biological and environmental contexts. Molecule <<|mol0|>> exhibits an xLogP3-AA value of <<|num0|>>.",
        "The xLogP3-AA algorithm is designed to calculate the LogP value of a compound based on the atom-additive (AA) method, taking into consideration the hydrophobicity contributions of individual atoms or functional groups within a molecule. With its predefined atom-types and associated hydrophobicity values, this method is extensively applied in computational chemistry, drug design, and environmental studies to forecast the behavior of chemical compounds in numerous biological and environmental situations. The xLogP3-AA value for molecule <<|mol0|>> is calculated to be <<|num0|>>.",
        "Estimating a compound's LogP value through the atom-additive (AA) method, the xLogP3-AA algorithm factors in the hydrophobicity contributions of individual atoms or functional groups in a molecule. Utilizing a set of predefined atom-types and their hydrophobicity values, this method is of great importance in fields such as computational chemistry, drug design, and environmental studies, where it helps predict the behavior of chemical compounds in various biological and environmental conditions. For molecule <<|mol0|>>, the xLogP3-AA value is found to be <<|num0|>>.",
        "The xLogP3-AA algorithm computes a compound's LogP value using the atom-additive (AA) method, which considers the hydrophobicity contributions from individual atoms or functional groups in a molecule. Relying on predefined atom-types and their hydrophobicity values, this method is widely adopted in computational chemistry, drug design, and environmental studies to predict the behavior of chemical compounds across a range of biological and environmental settings. In the case of molecule <<|mol0|>>, the determined xLogP3-AA value is <<|num0|>>.",
    ],
    "cactvs_tpsa": [
        "Cactvs Topological Polar Surface Area (TPSA or Cactvs_TPSA) is a calculated molecular property that represents the total polar surface area of a chemical compound. TPSA is a useful parameter in understanding a molecule's absorption, distribution, metabolism, and excretion (ADME) properties, particularly in the context of drug design and development. The Cactvs TPSA value for molecule <<|mol0|>> is <<|num0|>>.",
        "The Cactvs Topological Polar Surface Area (TPSA or Cactvs_TPSA) is a molecular property that calculates the total polar surface area of a chemical compound, playing a crucial role in evaluating a molecule's absorption, distribution, metabolism, and excretion (ADME) properties, especially in drug design and development. For molecule <<|mol0|>>, its Cactvs TPSA value is <<|num0|>>.",
        "Representing the total polar surface area of a chemical compound, the Cactvs Topological Polar Surface Area (TPSA or Cactvs_TPSA) is a calculated molecular property that aids in understanding a molecule's absorption, distribution, metabolism, and excretion (ADME) properties in the context of drug design and development. Molecule <<|mol0|>> has a Cactvs TPSA value of <<|num0|>>.",
        "As a calculated molecular property, the Cactvs Topological Polar Surface Area (TPSA or Cactvs_TPSA) measures the total polar surface area of a chemical compound and serves as a valuable parameter for assessing a molecule's absorption, distribution, metabolism, and excretion (ADME) properties, particularly in drug design and development. The Cactvs TPSA value for molecule <<|mol0|>> stands at <<|num0|>>.",
        "The Cactvs Topological Polar Surface Area (TPSA or Cactvs_TPSA) calculates the total polar surface area of a chemical compound and is a molecular property that is instrumental in comprehending a molecule's absorption, distribution, metabolism, and excretion (ADME) properties, especially within the scope of drug design and development. Molecule <<|mol0|>> exhibits a Cactvs TPSA value of <<|num0|>>.",
        "A calculated molecular property, the Cactvs Topological Polar Surface Area (TPSA or Cactvs_TPSA) represents the total polar surface area of a chemical compound and is useful in determining a molecule's absorption, distribution, metabolism, and excretion (ADME) properties in the realm of drug design and development. The Cactvs TPSA value calculated for molecule <<|mol0|>> is <<|num0|>>.",
        "The Cactvs Topological Polar Surface Area (TPSA or Cactvs_TPSA) is a molecular property that calculates the total polar surface area of a chemical compound, providing insights into a molecule's absorption, distribution, metabolism, and excretion (ADME) properties, particularly with regard to drug design and development. In the case of molecule <<|mol0|>>, the Cactvs TPSA value is <<|num0|>>.",
        "A molecular property used to determine the total polar surface area of a chemical compound, the Cactvs Topological Polar Surface Area (TPSA or Cactvs_TPSA) plays a significant role in evaluating a molecule's absorption, distribution, metabolism, and excretion (ADME) properties, particularly in the field of drug design and development. The Cactvs TPSA value for molecule <<|mol0|>> is <<|num0|>>.",
        "Measuring the total polar surface area of a chemical compound, the Cactvs Topological Polar Surface Area (TPSA or Cactvs_TPSA) is a calculated molecular property that offers valuable insights into a molecule's absorption, distribution, metabolism, and excretion (ADME) properties, specifically in relation to drug design and development. The determined Cactvs TPSA value for molecule <<|mol0|>> is <<|num0|>>.",
        "The Cactvs Topological Polar Surface Area (TPSA or Cactvs_TPSA), a calculated molecular property, quantifies the total polar surface area of a chemical compound and assists in understanding a molecule's absorption, distribution, metabolism, and excretion (ADME) properties, mainly in the context of drug design and development. For molecule <<|mol0|>>, the Cactvs TPSA value is found to be <<|num0|>>.",
        "As a molecular property that calculates the total polar surface area of a chemical compound, the Cactvs Topological Polar Surface Area (TPSA or Cactvs_TPSA) contributes to the analysis of a molecule's absorption, distribution, metabolism, and excretion (ADME) properties, particularly when it comes to drug design and development. Molecule <<|mol0|>> has a Cactvs TPSA value of <<|num0|>>.",
    ]
}

def item2prompt(idx, item):
    items = []
    for k, v in Question_list.items():
        if k not in item:
            continue
        sample = {}
        sample["text"] = v[idx % len(v)]
        sample["entity"] = {}
        sample["entity"]["<<|mol0|>>"] = {}
        sample["entity"]["<<|mol0|>>"]["smiles"] = item["openeye_can_smiles"]
        if k in ["molecular_formula", "cas_name"]:
            sample["entity"]["<<|string0|>>"] = {}
            sample["entity"]["<<|string0|>>"]["value"] = item[k]
        else:
            sample["entity"]["<<|num0|>>"] = {}
            sample["entity"]["<<|num0|>>"]["value"] = item[k]

        items.append(sample)
    return items

def dict2prompt(path):
    data = json.load(open(path, 'r', encoding='utf-8'))
    prompt = []
    for idx, item in enumerate(data):
        items = item2prompt(idx, item)
        prompt.extend(items)
    return prompt

def savejson(data, path):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)


def process_file(file):
    jsonpath = os.path.join('/mnt/pubchem/json/', file.replace('.sdf.gz', '.json'))
    print(f"Processing {jsonpath}")
    prompt = dict2prompt(jsonpath)
    savepath = os.path.join('/mnt/pubchem/prompt/', file.replace('.json', 'prompt.json'))
    print(f"Saving to {savepath}")
    savejson(prompt, savepath)


# Get the list of files
files = []
for file in os.listdir('/mnt/pubchem/json/'):
    if file.endswith('.json'):
        files.append(file)
print(files)

# Create a process pool and start processing the files
with multiprocessing.Pool(12) as pool:
    pool.map(process_file, files)
