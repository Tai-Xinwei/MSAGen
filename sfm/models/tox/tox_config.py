# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass

from sfm.models.graphormer.graphormer_config import GraphormerConfig


@dataclass
class TOXConfig(GraphormerConfig):
    model_type: str = "tox"
    seq_masking_method: str = "transformerM"

    add_rope: bool = True
    num_residues: int = 32
    max_num_aa: int = 1024
    encoder_pair_embed_dim: int = 64
    task: str = "mae"
    sample_mode: bool = False

    train_data_path: str = ""
    valid_data_path: str = ""

    lamb_pde: float = 0.01

    mode: str = "score"

    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(args)
        for k, v in asdict(self).items():
            if hasattr(args, k):
                setattr(self, k, getattr(args, k))
