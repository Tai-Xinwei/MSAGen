# -*- coding: utf-8 -*-
import json
import os
import re
import subprocess
from argparse import ArgumentParser
from glob import glob

from rdkit import Chem
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def test_aromatic(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            return True
    return False


def get_label(smiles, method, functional_groups):
    if method == "nitro":
        return int(smiles.find("n") != -1 or smiles.find("N") != -1)
    elif method == "sulf":
        return int(smiles.find("s") != -1 or smiles.find("S") != -1)
    elif method == "aromatic":
        return int(test_aromatic(smiles))
    elif method == "carboxyl":
        for func_group in functional_groups:
            if func_group.find("carbox") != -1:
                return 1
        return 0
    elif method == "funcg":
        return 0
    else:
        raise ValueError(f"Unknown test method {method}.")


def test_score(response, method):
    if method == "nitro":
        if (
            response.find("molecule contains no") != -1
            or response.find("molecule has no") != -1
            or response.find("molecule does not") != -1
            or (
                response.find("nitro") != -1
                and (response.find(" no ") != -1 or response.find(" not ") != -1)
            )
            or response.find("false") != -1
            or response.find("False") != -1
            or (response.find("No, ") != -1 and response.find("nitr") != -1)
            or response.find("No, it does not.") != -1
        ):
            return -1
        elif (
            (
                (
                    response.find("molecule contains") != -1
                    or response.find("molecule has") != -1
                    or response.find("molecule does contain") != -1
                )
                and response.find("nitr") != -1
            )
            or response.find("is a nitrogen") != -1
            or response.find("is a nitrile") != -1
            or (
                (
                    (response.find("Yes") != -1 or response.find("yes") != -1)
                    and ((response.find("nitro") != -1))
                )
                or (response.find("Yes, it does.") != -1)
            )
            or response.find("organonitrogen") != -1
            or response.find("is a nitroso") != -1
            or (
                response.find("nitr") != -1
                and response.find(" no ") == -1
                and response.find(" not ") == -1
            )
            or response.find("true") != -1
            or response.find("True") != -1
            or response.find(" N-") != -1
        ):
            return 1
        return 0
    elif method == "aromatic":
        if response.find("The molecule is a natural product found in ") == -1:
            if (
                (response.find(" not ") != -1 or response.find(" no ") != -1)
                and response.find("aroma") != -1
            ) or response.find("No") != -1:
                return -1
            elif (
                response.find("aroma") != -1
                or response.find("Yes, it does.") != -1
                or response.find("Yes.") != -1
            ):
                return 1
        return 0
    elif method == "sulf":
        if response.find("The molecule is a natural product found in ") == -1:
            if (
                (response.find(" not ") != -1 or response.find(" no ") != -1)
                and (response.find("sulf") != -1 or response.find("sulph") != -1)
            ) or response.find("No") != -1:
                return -1
            elif (
                response.find("sulf") != -1
                or response.find("sulph") != -1
                or response.find("Yes, it does.") != -1
                or response.find("Yes.") != -1
            ):
                return 1
        return 0
    elif method == "carboxyl":
        if (
            (response.find(" not ") != -1 or response.find(" no ") != -1)
            and response.find("carbox") != -1
        ) or response.find("No") != -1:
            return -1
        elif (
            response.find("carbox") != -1
            or response.find("Yes, it does.") != -1
            or response.find("Yes.") != -1
        ):
            return 1
        return 0
    elif method == "funcg":
        return 0
    else:
        raise ValueError(f"Unknown test method {method}.")


def calc_precision_and_recall(response, functional_groups):
    pred_functional_groups = list(
        [
            x.strip().strip(".")
            for x in filter(
                lambda x: x != "" and x.find("one can") == -1,
                re.split(" a |,|and", response),
            )
        ]
    )[1:]
    print("pred_functional_groups", pred_functional_groups)
    precision = 0.0
    recall = 0.0
    for pred_func_group in pred_functional_groups:
        if pred_func_group in functional_groups:
            precision += 1.0
    for func_group in functional_groups:
        if func_group in pred_functional_groups:
            recall += 1.0
    precision = (
        precision / len(pred_functional_groups)
        if len(pred_functional_groups) > 0
        else 0.0
    )
    recall = recall / len(functional_groups) if len(functional_groups) > 0 else 0.0
    return precision, recall


def eval(in_fnames, method):
    smiles_to_responses = {}
    for in_fname in glob(in_fnames):
        with open(in_fname, "r") as in_file:
            json_lines = ""
            for line in in_file:
                json_lines += line
                if line.strip() == "}" or (
                    line.find("{") != -1 and line.find("}") != -1
                ):
                    json_obj = json.loads(json_lines)
                    json_lines = ""
                    smiles = json_obj.get("smiles", list(json_obj.keys())[0])
                    response = (
                        json_obj.get("response", list(json_obj.values())[0][0])
                        .split("Response:\n")[-1]
                        .split("</s>")[0]
                    )
                    if smiles not in smiles_to_responses:
                        smiles_to_responses[smiles] = []
                    smiles_to_responses[smiles].append(response)
    scores = []
    labels = []
    total_cnt = 0
    total_aligned_cnt = 0
    if method == "funcg":
        precision = 0.0
        recall = 0.0
    for smiles in tqdm(smiles_to_responses):
        score = 0.0
        cnt = 0
        aligned_cnt = 0
        os.system(f'obabel -:"{smiles}" -omol > 0.mol')
        func_groups = list(
            filter(
                lambda x: x != "" and x != "cation" and x != "anion",
                subprocess.check_output(
                    [
                        "/home/shiyu/git/stanford_alpaca_mfm/datasets-new/functional_group/checkmol",
                        "0.mol",
                    ]
                )
                .decode()
                .split("\n"),
            )
        )
        print(smiles)
        for response in smiles_to_responses[smiles]:
            response_score = 0
            if test_score(response, method) == -1:
                score -= 1
                aligned_cnt += 1
                response_score = -1
            elif test_score(response, method) == 1:
                score += 1
                aligned_cnt += 1
                response_score = 1
            if method == "funcg":
                pr, rc = calc_precision_and_recall(response, func_groups)
                print("true func groups", func_groups)
                precision += pr
                recall += rc
                print(f"{response.strip()}")
                print(f"precision: {pr}, recall: {rc}")
            else:
                print(f"{response.strip()} {response_score}")
                print(func_groups)
            cnt += 1
        total_cnt += cnt
        if method != "funcg":
            total_aligned_cnt += aligned_cnt
            score /= cnt
            # score = (score / aligned_cnt) if aligned_cnt > 0 else 0.0
            label = get_label(smiles, method, func_groups)
            print(f"label {label} score {score} aligned_cnt {aligned_cnt} cnt {cnt}")
            scores.append(score)
            labels.append(label)
        print()
    if method != "funcg":
        print(f"labels: {labels}")
        print(f"scores: {scores}")
        print(
            f"aligned_cnt: {total_aligned_cnt}, total_cnt: {total_cnt}, aligned_rate: {total_aligned_cnt / total_cnt}"
        )
        print(f"auc: {roc_auc_score(labels, scores)}")
    else:
        print(
            f"total_cnt: {total_cnt}, precision: {precision / total_cnt}, recall: {recall / total_cnt}"
        )


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("in_fname", type=str, default=None, help="input file name")
    arg_parser.add_argument("method", type=str, help="test method")
    args = arg_parser.parse_args()
    eval(args.in_fname, args.method)
