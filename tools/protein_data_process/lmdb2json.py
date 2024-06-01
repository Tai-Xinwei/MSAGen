# -*- coding: utf-8 -*-
import re
import ast
import json
import lmdb
import spacy
import pickle as pkl
from sfm.data.prot_data.util import bstr2obj


nlp = spacy.load("en_core_web_sm")


def is_noun(text):
    doc = nlp(text)
    tags = [token.pos_ for token in doc[:1]]

    if all(tag in ["VERB", "ADP"] for tag in tags):
        return "phrase"
    else:
        return "noun"


def get_ec_class(ec_class_path=None, save_path=None):
    ec_class_path = "/home/v-zekunguo/zekun_data/protein/EC_class.txt"
    ec_class = {}
    with open(ec_class_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            s = line.split("  ")  # split the string into two parts
            if len(s) > 2:
                s = line.split("   ")
            # further split the first part and assign to variables
            ec1, ec2, ec3, ec4 = s[0].replace(". ", ".").split(".")

            # the second part is the description
            description = s[1].strip()
            if description == "":
                print(line)
            if ec2 == "-":
                ec_class[ec1] = {}
                ec_class[ec1]["description"] = description[0].lower() + description[1:]
                ec_class[ec1]["type"] = is_noun(description)
                continue
            if ec3 == "-":
                ec_class[ec1][ec2] = {}
                ec_class[ec1][ec2]["description"] = (
                    description[0].lower() + description[1:]
                )
                ec_class[ec1][ec2]["type"] = is_noun(description)
                continue
            if ec4 == "-":
                ec_class[ec1][ec2][ec3] = {}
                ec_class[ec1][ec2][ec3]["description"] = (
                    description[0].lower() + description[1:]
                )
                ec_class[ec1][ec2][ec3]["type"] = is_noun(description)
            # print(ec1, ec2, ec3, ec4, description)
            # break
    # print(ec_class)
    save_path = "/home/v-zekunguo/zekun_data/protein/EC_class.json"
    with open(save_path, "w") as f:
        json.dump(ec_class, f)


def EC_lmdb2_json(lmdb_path, save_path):
    ec_class_path = "/home/v-zekunguo/zekun_data/protein/EC_class.json"
    result = []
    with open(ec_class_path, "r") as f:
        ec_class = json.load(f)
    new_env = lmdb.open(
        str(lmdb_path),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
    )
    txn = new_env.begin(write=False)

    metadata = bstr2obj(txn.get("__metadata__".encode()))
    keys = metadata["keys"]

    s = metadata["comment"]

    # Use regular expressions to find content within {}
    matches = re.findall(r"{(.*?)}", s)

    for match in matches:
        # print(match)  # Output: content to extract
        d = ast.literal_eval("{" + match + "}")
    conversion_dict = {}
    count = 0
    for key, value in d.items():
        conversion_dict[value] = key
    for key in keys:
        value = txn.get(str(key).encode())
        value = bstr2obj(value)
        tmp = {}
        target_list = []
        tmp["aa"] = [value["aa"]]
        # if len(value["target"]) > 2:
        #     print(key)
        #     print(value.keys())
        #     for target in list(set(value["target"])):

        #         print(conversion_dict[target])
        #     break
        tmp_target_value_list = set()
        for target_value in list(set(value["target"])):
            tmp_target_value_list.add(conversion_dict[target_value][:-1])
        # if len(tmp_target_value_list) > 1:
        #     count += 1
        for target_value in list(tmp_target_value_list):
            # print(target_value)
            ec1, ec2, ec3, _ = target_value.split(".")
            tmp_target_list = []
            tmp_target_list.append(ec_class[ec1]["description"][:-1])
            tmp_target_list.append(ec_class[ec1][ec2]["description"][:-1])
            tmp_target_list.append(ec_class[ec1][ec2][ec3]["description"][:-1])
            # ec2 ec3 type
            tmp_target_list.append(
                ", ".join([ec_class[ec1][ec2]["type"], ec_class[ec1][ec2][ec3]["type"]])
            )
            target_list.append(tmp_target_list)
        tmp["target"] = target_list
        tmp["ec_class"] = True
        result.append(tmp)
    # print(count)
    with open(save_path, "w") as f:
        json.dump(result, f)


def ppi_affinity_lmdb2_json(lmdb_path, save_path):
    result = []
    new_env = lmdb.open(
        str(lmdb_path),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
    )
    txn = new_env.begin(write=False)

    metadata = bstr2obj(txn.get("__metadata__".encode()))
    keys = metadata["keys"]
    for key in keys:
        value = txn.get(str(key).encode())
        value = bstr2obj(value)
        tmp = {}
        target_list = []
        tmp["aa"] = value["aa"]
        if len(value["target"]) > 1:
            print(value["target"])
        tmp["target"] = value["target"][0]
        tmp["ppi_affinity"] = True
        result.append(tmp)
    with open(save_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    get_ec_class()
    EC_lmdb2_json(
        "/home/v-zekunguo/blob/pfm/data/bfm_benchmark/EnzymeCommission/EnzymeCommission_valid.lmdb",
        "/home/v-zekunguo/zekun_data/protein/EnzymeCommission_valid.json",
    )
    ppi_affinity_lmdb2_json(
        "/home/v-zekunguo/blob/pfm/data/bfm_benchmark/ppi_affinity/ppi_affinity_valid.lmdb",
        "/home/v-zekunguo/zekun_data/protein/ppi_affinity_valid.json",
    )
