# -*- coding: utf-8 -*-
import os
import sys
from typing import Dict

import deepspeed
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])
from sfm.pipeline.generalist.graphormerllama_3Dtrainer import Trainer3D


def main() -> None:
    freeze_list = []
    unfreeze_list = ["adaptor", "dummy", "0.layers.22", "0.layers.23"]

    trainer = Trainer3D(
        freeze_list=freeze_list,
        unfreeze_list=unfreeze_list,
    )

    trainer.train_tensor_pipeline()


if __name__ == "__main__":
    main()
