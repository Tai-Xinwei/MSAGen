# -*- coding: utf-8 -*-
import os

import numpy as np
from pymatgen.core.periodic_table import Element


def read_lines(data_path):
    with open(data_path, "r") as fr:
        lines = fr.readlines()
    lines = [line.strip() for line in lines]
    return lines


def read_slices(data_path):
    lines = read_lines(data_path)
    slices = [line.strip().split() for line in lines]
    return slices


def get_vocab_slices(slices):
    vocab_slices = {}
    for slice in slices:
        for i, token in enumerate(slice):
            if token not in vocab_slices:
                vocab_slices[token] = 1
            else:
                vocab_slices[token] += 1
    return vocab_slices


def get_vocab_slices_from_data_path(data_path):
    slices = read_slices(data_path)
    vocab_slices = get_vocab_slices(slices)
    return vocab_slices


def sort_vocab_slices(vocab_slices):
    vocab_slices = dict(
        sorted(vocab_slices.items(), key=lambda item: item[1], reverse=True)
    )
    return vocab_slices


def save_vocab_slices(vocab_slices, output_path):
    with open(output_path, "w") as fw:
        for key, value in vocab_slices.items():
            fw.write(f"{key} {value}\n")
    return


def stat_and_save_vocab_slices():
    data_path = (
        "/hai1/SFM/threedimargen/data/materials_data/mp_nomad_qmdb_train_slices.txt"
    )
    output_path = "/hai1/SFM/threedimargen/data/materials_data/mp_nomad_qmdb_train_slices_vocab.txt"
    vocab_slices = get_vocab_slices_from_data_path(data_path)
    vocab_slices = sort_vocab_slices(vocab_slices)
    save_vocab_slices(vocab_slices, output_path)
    return


def get_vocab():
    all_tok = []
    # add special tokens
    all_tok.append("o")
    all_tok.append("-")
    all_tok.append("+")

    # add elements
    all_tok += [e.symbol for e in Element]

    # add edge tokens
    for i in range(1000):
        all_tok.append(str(i))
    return all_tok


def save_vocab(output_path):
    all_tok = get_vocab()
    with open(output_path, "w") as fw:
        for tok in all_tok:
            fw.write(f"{tok}\n")
    return


def stat_length():
    data_path = (
        "/hai1/SFM/threedimargen/data/materials_data/mp_nomad_qmdb_train_slices.txt"
    )
    slices = read_slices(data_path)
    lengths = [len(slice) for slice in slices]
    lengths = np.array(lengths)
    print(f"count of length > 2048: {np.sum(lengths > 2048-2)}")
    print(f"max length: {np.max(lengths)}")
    print(f"min length: {np.min(lengths)}")
    print(f"mean length: {np.mean(lengths)}")
    print(f"median length: {np.median(lengths)}")
    return


def check_vocab():
    all_tok = get_vocab()
    all_tok_vocab = {tok: 1 for tok in all_tok}
    all_tok_vocab["<gen>"] = 1
    data_path = (
        "/hai1/SFM/threedimargen/data/materials_data/mp_nomad_qmdb_train_slices.txt"
    )
    slices = read_slices(data_path)
    for slice in slices:
        for token in slice:
            if token not in all_tok_vocab:
                print(f"{token} not in all_tok")
    return


def main():
    # stat_and_save_vocab_slices()
    # stat_length()
    # check_vocab()
    get_vocab()
    save_vocab("dict_slices.txt")


if __name__ == "__main__":
    main()
