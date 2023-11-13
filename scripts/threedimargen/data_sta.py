# -*- coding: utf-8 -*-
import json
import statistics

import matplotlib.pyplot as plt
import numpy as np
import tqdm


# Read in the data from the file
def read_data(data_path):
    abc_nums = []
    xyz_nums = []
    atoms = []
    with open(data_path, "r") as f:
        for line in tqdm.tqdm(f):
            data = json.loads(line)
            for site in data["sites"]:
                abc = list(site["fractional_coordinates"])
                xyz = list(site["cartesian_coordinates"])
                abc_nums.extend(abc)
                xyz_nums.extend(xyz)
            atoms.append(len(data["sites"]))
    return abc_nums, xyz_nums, atoms


def draw_coordinates_distribution(
    numbers, name="histogram.png", title="Distribution of Numbers in the Range [-1, 1]"
):
    # Calculate the histogram using numpy
    hist, bin_edges = np.histogram(numbers, bins=20, range=(-1, 1))

    # Print the histogram values
    print("Histogram values:", hist)

    # Visualize the histogram using matplotlib
    plt.hist(numbers, bins=20, range=(-1, 1), edgecolor="black")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.savefig(name)
    plt.clf()


def draw_lengths_distribution(
    numbers, name="histogram.png", title="Distribution of number of lengths"
):
    # Calculate the histogram using numpy
    hist, bin_edges = np.histogram(numbers, bins=20, range=(0, 10000))

    # Print the histogram values
    print("Histogram values:", hist)

    # Visualize the histogram using matplotlib
    plt.hist(numbers, range=(0, 10000), edgecolor="black")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.savefig(name)
    plt.clf()


def distribution():
    abc_nums, xyz_nums, atoms = read_data(
        "/blob/renqian/data/sfm/gen/materials/mp.jsonl"
    )
    # draw_coordinates_distribution(abc_nums, "mp_fractional_coordinates_histogram.png")
    # draw_coordinates_distribution(xyz_nums, "mp_cartesian_coordinates_histogram.png")
    draw_lengths_distribution(atoms, "mp_lengths_histogram.png")
    print(
        f"max length: {max(atoms)}\tmean: {statistics.mean(atoms)}\tmedian: {statistics.median(atoms)}"
    )

    abc_nums, xyz_nums, atoms = read_data(
        "/blob/renqian/data/sfm/gen/materials/nomad.jsonl"
    )
    # draw_coordinates_distribution(abc_nums, "nomad_fractional_coordinates_histogram.png")
    # draw_coordinates_distribution(xyz_nums, "nomad_cartesian_coordinates_histogram.png")
    draw_lengths_distribution(atoms, "nomad_lengths_histogram.png")
    print(
        f"max length: {max(atoms)}\tmean: {statistics.mean(atoms)}\tmedian: {statistics.median(atoms)}"
    )


def duplicate():
    return
