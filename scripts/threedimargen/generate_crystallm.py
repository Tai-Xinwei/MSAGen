# -*- coding: utf-8 -*-
import json
import math

import requests
from pymatgen.symmetry.groups import SpaceGroup
from tqdm import tqdm

# Get your API key from the CrystaLLM team
api_key = "a37e315c660561e2fd5abf03"
max_sites = 100

SPACE_GROUPS = [
    "Aea2",
    "Aem2",
    "Ama2",
    "Amm2",
    "C2",
    "C2/c",
    "C2/m",
    "C222",
    "C222_1",
    "Cc",
    "Ccc2",
    "Ccce",
    "Cccm",
    "Cm",
    "Cmc2_1",
    "Cmce",
    "Cmcm",
    "Cmm2",
    "Cmme",
    "Cmmm",
    "F-43c",
    "F-43m",
    "F222",
    "F23",
    "F432",
    "F4_132",
    "Fd-3",
    "Fd-3c",
    "Fd-3m",
    "Fdd2",
    "Fddd",
    "Fm-3",
    "Fm-3c",
    "Fm-3m",
    "Fmm2",
    "Fmmm",
    "I-4",
    "I-42d",
    "I-42m",
    "I-43d",
    "I-43m",
    "I-4c2",
    "I-4m2",
    "I222",
    "I23",
    "I2_12_12_1",
    "I2_13",
    "I4",
    "I4/m",
    "I4/mcm",
    "I4/mmm",
    "I422",
    "I432",
    "I4_1",
    "I4_1/a",
    "I4_1/acd",
    "I4_1/amd",
    "I4_122",
    "I4_132",
    "I4_1cd",
    "I4_1md",
    "I4cm",
    "I4mm",
    "Ia-3",
    "Ia-3d",
    "Iba2",
    "Ibam",
    "Ibca",
    "Im-3",
    "Im-3m",
    "Ima2",
    "Imm2",
    "Imma",
    "Immm",
    "P-1",
    "P-3",
    "P-31c",
    "P-31m",
    "P-3c1",
    "P-3m1",
    "P-4",
    "P-42_1c",
    "P-42_1m",
    "P-42c",
    "P-42m",
    "P-43m",
    "P-43n",
    "P-4b2",
    "P-4c2",
    "P-4m2",
    "P-4n2",
    "P-6",
    "P-62c",
    "P-62m",
    "P-6c2",
    "P-6m2",
    "P1",
    "P2",
    "P2/c",
    "P2/m",
    "P222",
    "P222_1",
    "P23",
    "P2_1",
    "P2_1/c",
    "P2_1/m",
    "P2_12_12",
    "P2_12_12_1",
    "P2_13",
    "P3",
    "P312",
    "P31c",
    "P31m",
    "P321",
    "P3_1",
    "P3_112",
    "P3_121",
    "P3_2",
    "P3_212",
    "P3_221",
    "P3c1",
    "P3m1",
    "P4",
    "P4/m",
    "P4/mbm",
    "P4/mcc",
    "P4/mmm",
    "P4/mnc",
    "P4/n",
    "P4/nbm",
    "P4/ncc",
    "P4/nmm",
    "P4/nnc",
    "P422",
    "P42_12",
    "P4_1",
    "P4_122",
    "P4_12_12",
    "P4_132",
    "P4_2",
    "P4_2/m",
    "P4_2/mbc",
    "P4_2/mcm",
    "P4_2/mmc",
    "P4_2/mnm",
    "P4_2/n",
    "P4_2/nbc",
    "P4_2/ncm",
    "P4_2/nmc",
    "P4_2/nnm",
    "P4_22_12",
    "P4_232",
    "P4_2bc",
    "P4_2cm",
    "P4_2mc",
    "P4_2nm",
    "P4_3",
    "P4_322",
    "P4_32_12",
    "P4_332",
    "P4bm",
    "P4cc",
    "P4mm",
    "P4nc",
    "P6/m",
    "P6/mcc",
    "P6/mmm",
    "P622",
    "P6_1",
    "P6_122",
    "P6_2",
    "P6_222",
    "P6_3",
    "P6_3/m",
    "P6_3/mcm",
    "P6_3/mmc",
    "P6_322",
    "P6_3cm",
    "P6_3mc",
    "P6_4",
    "P6_422",
    "P6_5",
    "P6_522",
    "P6cc",
    "P6mm",
    "Pa-3",
    "Pba2",
    "Pbam",
    "Pban",
    "Pbca",
    "Pbcm",
    "Pbcn",
    "Pc",
    "Pca2_1",
    "Pcc2",
    "Pcca",
    "Pccm",
    "Pccn",
    "Pm",
    "Pm-3",
    "Pm-3m",
    "Pm-3n",
    "Pma2",
    "Pmc2_1",
    "Pmm2",
    "Pmma",
    "Pmmm",
    "Pmmn",
    "Pmn2_1",
    "Pmna",
    "Pn-3",
    "Pn-3m",
    "Pn-3n",
    "Pna2_1",
    "Pnc2",
    "Pnma",
    "Pnn2",
    "Pnna",
    "Pnnm",
    "Pnnn",
    "R-3",
    "R-3c",
    "R-3m",
    "R3",
    "R32",
    "R3c",
    "R3m",
]
SPACE_GROUPS_DICT = {SpaceGroup(symbol).int_number: symbol for symbol in SPACE_GROUPS}
for i in range(1, 231):
    if i not in SPACE_GROUPS_DICT:
        SPACE_GROUPS_DICT[i] = SpaceGroup.from_int_number(i).symbol


def get_response(comp, z, space_group, model="small"):
    # Specify the model and the message
    print(comp, z, space_group)
    model = "small"
    message = {"comp": comp, "sg": space_group}
    if z is not None:
        message["z"] = z

    # Make a POST request to the API endpoint
    endpoint = "https://api.crystallm.com/v1/generate"
    headers = {"Content-Type": "application/json", "x-api-key": api_key}
    data = {"model": model, "message": message}
    response = requests.post(endpoint, headers=headers, json=data)

    # Parse the response
    if response.status_code == 200:
        result = response.json()
        cif = result["cifs"][0]["generated"]
        valid = result["cifs"][0]["valid"]
        messages = result["cifs"][0]["messages"]
        # fe = result["cifs"][0]["fe"]
        print("Generated CIF file:")
        print(cif)
        print("Validity status:", valid)
        print("Messages:", messages)
        # print("Formation energy per atom:", fe)
    else:
        print(f"Error: {response.status_code} - {response.text}")
        result = None
    return result


def get_z(formula, sites):
    sites_count = {}
    for site in sites:
        if site["element"] not in sites_count:
            sites_count[site["element"]] = 0
        sites_count[site["element"]] += 1

    # get the first element from the formula
    i = 1
    while i < len(formula) and formula[i].islower():
        i += 1
    elem = formula[:i]

    if i < len(formula) and formula[i].isdigit():
        num_start = i
        while i < len(formula) and formula[i].isdigit():
            i += 1
        num = int(formula[num_start:i])
    else:
        num = 1
    z = math.ceil(sites_count[elem] / num)
    return z


def infer_from_file(input_file, output_file):
    with open(output_file, "w") as fw:
        with open(input_file, "r") as fr:
            for i, line in tqdm(enumerate(fr)):
                data = json.loads(line)
                space_group_symbol = SPACE_GROUPS_DICT[data["space_group"]["no"]]
                formula = data["formula"]
                data["sites"]
                # z = get_z(formula, sites)
                result = get_response(formula, z=None, space_group=space_group_symbol)
                # cif = result["cifs"][0]["generated"]
                # valid = result["cifs"][0]["valid"]
                # messages = result["cifs"][0]["messages"]
                # fe = result["cifs"][0]["fe"]
                data["response"] = result

                fw.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    infer_from_file(
        "/hai1/SFM/threedimargen/data/materials_data/mp_nomad_dedup_valid.jsonl",
        "/hai1/SFM/threedimargen/data/materials_data/mp_nomad_dedup_valid_infer_crystallm.jsonl",
    )
