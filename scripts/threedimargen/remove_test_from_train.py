# -*- coding: utf-8 -*-
import json
import os


def get_formula(data):
    formula = ""
    for site in data["sites"]:
        formula += site["element"]
    formula = "".join(sorted(formula))
    return formula


def remove_test_from_train(train_paths, test_paths, output_path):
    train_data = []
    for train_path in train_paths:
        with open(train_path) as fr:
            for line in fr:
                train_data.append(json.loads(line))
    test_data = []
    for test_path in test_paths:
        with open(test_path) as fr:
            for line in fr:
                test_data.append(json.loads(line))

    test_formulas = [get_formula(data) for data in test_data]
    test_formulas = set(test_formulas)
    print(f"test num: {len(test_formulas)}")
    print(f"unique test formulas: {len(test_formulas)}")
    with open(output_path, "w") as fw:
        for data in train_data:
            formula = get_formula(data)
            if formula not in test_formulas:
                fw.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    remove_test_from_train(
        [
            "/hai1/SFM/threedimargen/data/materials_data/mp.jsonl",
            "/hai1/SFM/threedimargen/data/materials_data/nomad.jsonl",
            "/hai1/SFM/threedimargen/data/materials_data/qmdb.jsonl",
            "/hai1/SFM/threedimargen/data/materials_data/carbon_24_train.jsonl",
            "/hai1/SFM/threedimargen/data/materials_data/perov_5_train.jsonl",
        ],
        [
            "/hai1/SFM/threedimargen/data/materials_data/carbon_24_val.jsonl",
            "/hai1/SFM/threedimargen/data/materials_data/carbon_24_test.jsonl",
            "/hai1/SFM/threedimargen/data/materials_data/perov_5_val.jsonl",
            "/hai1/SFM/threedimargen/data/materials_data/perov_5_test.jsonl",
            "/hai1/SFM/threedimargen/data/materials_data/mp_20_val.jsonl",
            "/hai1/SFM/threedimargen/data/materials_data/mp_20_test.jsonl",
            "/hai1/SFM/threedimargen/data/materials_data/mpts-52_test.jsonl",
        ],
        "/hai1/SFM/threedimargen/data/materials_data/mp_nomad_qmdb_train.jsonl",
    )
