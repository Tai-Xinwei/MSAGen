# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass

from sfm.models.graphormer.graphormer_config import GraphormerConfig


@dataclass
class TOXConfig(GraphormerConfig):
    model_type: str = "pfm"
    seq_masking_method: str = "transformerM"
    add_rope: bool = True
    num_residues: int = 32
    max_num_aa: int = 1024
    task: str = "mae"
    sample_mode: bool = False

    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(args)
        for k, v in asdict(self).items():
            if hasattr(args, k):
                setattr(self, k, getattr(args, k))
