# -*- coding: utf-8 -*-
import gzip
import os
import json
import multiprocessing
import random
from string import Template

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
    "ec_class_base": [
        "The amino acid sequence of this protein is <<|protein0|>>, and the enzymatic description is as follows:",
        "The protein has an amino acid sequence of <<|protein0|>>, with the following enzyme description:",
        "With an amino acid sequence of <<|protein0|>>, the enzyme description of this protein is as follows:",
        "This protein, having an amino acid sequence of <<|protein0|>>, is described enzymatically as follows:",
        "The amino acid sequence for this protein is <<|protein0|>>, and the subsequent enzyme description is given:",
        "The enzyme description of this protein, with an amino acid sequence of <<|protein0|>>, is as follows:",
        "This protein is characterized by an amino acid sequence of <<|protein0|>>, and its enzyme description is as follows:",
        "The protein, with an amino acid sequence of <<|protein0|>>, has the following enzymatic characteristics:",
        "The amino acid sequence <<|protein0|>> identifies this protein, and the enzyme description is provided below:",
        "The enzyme description for this protein, which has an amino acid sequence of <<|protein0|>>, is as follows:",
        "Featuring an amino acid sequence of <<|protein0|>>, this protein has the following enzyme description:",
    ],
    "ec2_noun": [
        Template("Specifically, it belongs to the $ec2. "),
        Template("Specifically, it is part of the $ec2. "),
        Template("It is further organized into the $ec2. "),
        Template("More precisely, it is included in the $ec2. "),
        Template("It further belongs to the $ec2. "),
        Template("On a more specific level, it falls into the $ec2. "),
        Template("More specifically, it is in the $ec2. "),
        Template("More specifically, it falls under the $ec2. "),
        Template(" It also falls within the more specific $ec2. "),
    ],
    "ec2_phrase": [
        Template("It is $ec2, "),
    ],
    "ec3_noun": [
        Template("More precisely, it's part of the $ec3."),
        Template(" Most accurately, it is organized into the $ec3."),
        Template("In the most specific terms, it is in the $ec3."),
        Template("Most accurately, it falls into the $ec3."),
        Template("More specifically, it belongs to the $ec3."),
        Template("At the most detailed level, it is part of the $ec3."),
        Template("At the narrowest level, it is in the $ec3."),
        Template("In the most detailed categorization, it is in the $ec3."),
        Template("Most precisely, it is within the $ec3."),
    ],
    "ec3_phrase": [
        Template("$ec3."),
    ],
    "ec_class_list_item": [
        Template(" $index. This protein enzyme, identified as a $ec1 type. $ec2ec3"),
        Template(" $index. This protein belongs to the $ec1 category. $ec2ec3"),
        Template(" $index. The protein, a member of the $ec1 category. $ec2ec3"),
        Template(" $index. The protein, falling into the $ec1 category. $ec2ec3"),
        Template(" $index. The protein, a constituent of the $ec1 category. $ec2ec3"),
        Template(" $index. The protein belongs to the $ec1 category. $ec2ec3"),
        Template(" $index. This protein is part of the $ec1 category. $ec2ec3"),
        Template(" $index. This protein is classified as $ec1. $ec2ec3"),
        Template(" $index. The protein falls under the $ec1 category. $ec2ec3"),
    ],
    # 这条蛋白质其序列为<<|protein0|>>，属于$ec1，具体来说，其$ec2,并且$ec3
    "ec_class": [
        Template(
            "This protein, identified by the amino acid sequence <<|protein0|>>, belongs to the $ec1 category. $ec2ec3"
        ),
        Template(
            "The protein, recognized by its amino acid sequence <<|protein0|>>, falls under the $ec1 category. $ec2ec3"
        ),
        Template(
            "The protein with the sequence <<|protein0|>> is categorized under the $ec1. $ec2ec3"
        ),
        Template(
            "Recognized by the sequence <<|protein0|>>, this protein falls under the $ec1 category. $ec2ec3"
        ),
        Template(
            "The protein, identified by the sequence <<|protein0|>>, belongs to the $ec1 category. $ec2ec3"
        ),
        Template(
            "This protein, known by its amino acid sequence <<|protein0|>>, is classified under the $ec1 category. $ec2ec3"
        ),
        Template(
            "Identified by the sequence <<|protein0|>>, this protein is part of the $ec1 category. $ec2ec3"
        ),
        Template(
            "Characterized by the amino acid sequence <<|protein0|>>, this protein is a member of the $ec1 category. $ec2ec3"
        ),
        Template(
            "This protein, having the sequence <<|protein0|>>, is part of the $ec1 category. $ec2ec3"
        ),
        Template(
            "The protein, known by its amino acid sequence <<|protein0|>>, is part of the $ec1 class. $ec2ec3"
        ),
        Template(
            "With the amino acid sequence <<|protein0|>>, this protein is classified as $ec1. $ec2ec3"
        ),
    ],
    "ppi_affinity": [
        Template(
            "The affinity between the amino acid sequence <<|protein0|>> and the amino acid sequence <<|protein1|>> is quantified as $num."
        ),
        Template(
            "The binding strength of the amino acid sequence <<|protein0|>> with the amino acid sequence <<|protein1|>> is measured as $num."
        ),
        Template(
            "The score quantifying the affinity between the amino acid sequence <<|protein0|>> and the amino acid sequence <<|protein1|>> is $num."
        ),
        Template(
            "The affinity calculation between the amino acid sequence <<|protein0|>> and the amino acid sequence <<|protein1|>> gives a score of $num."
        ),
        Template(
            "The quantitative measure of the interaction between the protein sequence <<|protein0|>> and the protein sequence <<|protein1|>> is $num."
        ),
        Template(
            "The interaction value between the amino acid sequence <<|protein0|>> and the amino acid sequence <<|protein1|>> is registered at $num."
        ),
        Template(
            "The affinity measure for the interaction of the protein sequence <<|protein0|>> with the protein sequence <<|protein1|>> is $num."
        ),
        Template(
            "The compatibility of the amino acid sequence <<|protein0|>> with the amino acid sequence <<|protein1|>> is quantified by a score of $num."
        ),
        Template(
            "The affinity score between the protein sequence <<|protein0|>> and the protein sequence <<|protein1|>> is $num."
        ),
        Template(
            "The interaction between the amino acid sequence <<|protein0|>> and the amino acid sequence <<|protein1|>> has a value of $num."
        ),
        Template(
            "The binding strength of the protein sequence <<|protein0|>> with the protein sequence <<|protein1|>> is measured as $num."
        ),
    ],
}


def item2prompt(item):
    items = []
    for k, v in Question_list.items():
        if k not in item:
            continue
        sample = {}
        if "ec_class" == k:
            if len(item["target"]) > 1:
                cur_base_text = random.choice(Question_list[k + "_base"])
                for index, target in enumerate(item["target"]):
                    cur_template = random.choice(Question_list[k + "_list_item"])
                    ec2_type, ec3_type = target[-1].split(", ")
                    ec2_template = random.choice(Question_list["ec2_" + ec2_type])
                    ec3_template = random.choice(Question_list["ec3_" + ec3_type])
                    ec2ec3 = ec2_template.substitute(
                        ec2=target[1]
                    ) + ec3_template.substitute(ec3=target[2])
                    cur_index_text = cur_template.substitute(
                        index=(index + 1), ec1=target[0], ec2ec3=ec2ec3
                    )
                    cur_base_text += cur_index_text
            else:
                target = item["target"][0]
                ec2_type, ec3_type = target[-1].split(", ")
                ec2_template = random.choice(Question_list["ec2_" + ec2_type])
                ec3_template = random.choice(Question_list["ec3_" + ec3_type])
                ec2ec3 = ec2_template.substitute(
                    ec2=target[1]
                ) + ec3_template.substitute(ec3=target[2])
                cur_base_text = random.choice(v).substitute(
                    ec1=target[0], ec2ec3=ec2ec3
                )
        if "ppi_affinity" == k:
            cur_base_text = random.choice(v).substitute(num=item["target"])
        sample["text"] = cur_base_text
        sample["entity"] = {}
        sample["entity"]["<<|protein|>>"] = item["aa"]

        items.append(sample)
    return items


def dict2prompt(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    prompt = []
    for idx, item in enumerate(data):
        items = item2prompt(item)
        prompt.extend(items)
    return prompt


def savejson(data, path):
    with open(path, "w") as outfile:
        json.dump(data, outfile)


base_prompt_dir = "/home/v-zekunguo/zekun_data/protein/prompt"
base_json_dit = "/home/v-zekunguo/zekun_data/protein/json"


def process_file(file):
    jsonpath = os.path.join(base_json_dit, file)
    print(f"Processing {jsonpath}")
    prompt = dict2prompt(jsonpath)
    savepath = os.path.join(base_prompt_dir, file.replace(".json", "_prompt.json"))
    print(f"Saving to {savepath}")
    savejson(prompt, savepath)


# Get the list of files
files = []
for file in os.listdir(base_json_dit):
    if file.endswith(".json"):
        files.append(file)
print(files)

# Create a process pool and start processing the files
with multiprocessing.Pool(12) as pool:
    pool.map(process_file, files)
