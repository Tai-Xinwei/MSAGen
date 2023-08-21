# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import List

import numpy as np


def no_noise(
    item: dict,
    args: dict,
    seed: int,
    coord_noise: bool,
    angle_noise: bool,
):
    pos, ang = item["pos"], item["ang"]
    pos_shape, ang_shape = pos.shape, ang.shape
    return np.zeros(pos_shape), np.zeros(ang_shape)


def normal_noise(
    item: dict,
    args: dict,
    seed: int,
    coord_noise: bool,
    angle_noise: bool,
):
    pos, ang = item["pos"], item["ang"]
    pos_shape, ang_shape = pos.shape, ang.shape
    rng = np.random.default_rng(seed)
    if coord_noise:
        pos_noise = rng.normal(
            args.coord_noise_mean, args.coord_noise_stdev, size=pos_shape
        ).astype(np.float32)
    else:
        pos_noise = np.zeros(pos_shape, dtype=np.float32)
    if angle_noise:
        ang_noise = rng.normal(
            args.angle_noise_mean, args.angle_noise_stdev, size=ang_shape
        ).astype(np.float32)
    else:
        ang_noise = np.zeros(ang_shape, dtype=np.float32)
    return pos_noise, ang_noise


noise_registry = {
    "no": no_noise,
    "normal": normal_noise,
    # TODO: add more
}
