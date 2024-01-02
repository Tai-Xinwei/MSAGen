# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass

from sfm.models.graphormer.graphormer_config import GraphormerConfig


@dataclass
class PFMConfig(GraphormerConfig):
    model_type = "pfm"

    task_name: str = ""
    data_basepath: str = ""
    output_dim: int = 1024

    add_rope: bool = True
    stack_seq: bool = False
    num_residues: int = 32
    max_num_aa: int = 1024
    task: str = "mae"

    train_data_path: str = ""
    valid_data_path: str = ""

    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(args)
        for k, v in asdict(self).items():
            if hasattr(args, k):
                setattr(self, k, getattr(args, k))
