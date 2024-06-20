# -*- coding: utf-8 -*-
import csv
from string import Template

tsv_path = "/home/v-zekunguo/zekun_data/gene_anno/geneannot_031824_raw.tsv"
instruct_templete = [
    Template(
        "Can you illustrate the genetic function of the <protein>$protein</protein> protein?"
    ),
    Template(
        "Craft the gene function of the protein sequence <protein>$protein</protein>."
    ),
    Template(
        "What is the biological role of the <protein>$protein</protein> protein's gene?"
    ),
    Template(
        "What is the role of the gene in the <protein>$protein</protein> protein?"
    ),
    Template("Formulate the gene function for <protein>$protein</protein>."),
    Template("What is the gene function of <protein>$protein</protein>?"),
    Template("What's the genetic purpose of the <protein>$protein</protein> protein?"),
    Template("How does the gene of the <protein>$protein</protein> protein function?"),
    Template(
        "How would you define the gene function of the <protein>$protein</protein> protein?"
    ),
    Template(
        "Could you describe the function that the <protein>$protein</protein> protein's gene performs?"
    ),
]
import random

save_path = "/home/v-zekunguo/zekun_data/gene_anno/instruct_gene_annot_031824_test_no_predict.tsv"
result = []
count = 0
seq_dic = {}
use_count = 0
repeat_count = 0
with open(tsv_path, "r") as tsvfile:
    reader = csv.reader(tsvfile, delimiter="\t")
    is_first = True
    for row in reader:
        if is_first:
            is_first = False
            continue
        cur_templete = random.choice(instruct_templete)
        if "predicted" in row[6] or "Predicted" in row[6]:
            count += 1
            continue
        if row[-1].strip() in seq_dic:
            repeat_count += 1
            continue
        else:
            seq_dic[row[-1].strip()] = row[6].strip().capitalize()
            result.append(
                [
                    cur_templete.substitute(protein=row[-1].strip()),
                    row[6].strip().capitalize(),
                ]
            )
print(repeat_count)
# with open(save_path, "w") as file:
#     writer = csv.writer(file, delimiter="\t")
#     writer.writerows(result)
print(count)
print(len(result))
