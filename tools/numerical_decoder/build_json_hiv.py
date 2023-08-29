# -*- coding: utf-8 -*-
import json
import re

import deepchem as dc
import pandas as pd
from deepchem.feat.molecule_featurizers import RawFeaturizer
from rdkit import Chem


def rm_map_number(smiles):
    t = re.sub(":\d*", "", smiles)
    return t


def canonicalize(smiles):
    try:
        smiles = rm_map_number(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        else:
            return Chem.MolToSmiles(mol)
    except:
        return smiles


featurizer = RawFeaturizer(smiles=True)

tasks, dataset, _ = dc.molnet.load_hiv(featurizer=featurizer, splitter="scaffold")

train, valid, test = dataset
df_train = train.to_dataframe()
df_valid = valid.to_dataframe()
df_test = test.to_dataframe()

print(tasks)
print("Total:", df_train.shape[0] + df_valid.shape[0] + df_test.shape[0])
print(df_train.columns)
# import ipdb; ipdb.set_trace()
# {"text": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nDescribe this molecule.\n\n### Input:\n<<|mol0|>>\n\n### Response:\nThe molecule is a nitrile that is acetonitrile where one of the methyl hydrogens is substituted by a 2-methylphenyl group. It derives from an acetonitrile.",
# "entities": {"<<|mol0|>>": {"smiles": "Cc1ccccc1CC#N"}}}
split_token = "\n\n### "

prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
inst = "Instruction:\nPredict whether the given molecule can inhibit HIV replication."
mol_input = "Input:\n<<|mol0|>>"
response = "Response:\n"


def build_json_for_df(df):
    json_data = []
    for i in range(df.shape[0]):
        smiles = df.iloc[i]["X"]
        smiles = canonicalize(smiles)
        label_i = "Yes." if df.iloc[i]["y"] == 1.0 else "No."
        json_data.append(
            {
                "text": prefix
                + split_token
                + inst
                + split_token
                + mol_input
                + split_token
                + response
                + label_i,
                "entities": {
                    "<<|mol0|>>": {"smiles": smiles},
                },
            }
        )
    return json_data


json_data_train = build_json_for_df(df_train)
json_data_valid = build_json_for_df(df_valid)
json_data_test = build_json_for_df(df_test)

result_path = "/sfm/ds_dataset/qizhi_numerical"
dataset_name = 'hiv'

with open(f"{result_path}/json/{dataset_name}_train.json", "w") as f:
    for dictionary in json_data_train:
        json_str = json.dumps(dictionary)
        f.write(json_str)
        f.write("\n")
with open(f"{result_path}/json/{dataset_name}_valid.json", "w") as f:
    for dictionary in json_data_valid:
        json_str = json.dumps(dictionary)
        f.write(json_str)
        f.write("\n")
with open(f"{result_path}/json/{dataset_name}_test.json", "w") as f:
    for dictionary in json_data_test:
        json_str = json.dumps(dictionary)
        f.write(json_str)
        f.write("\n")
