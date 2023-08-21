# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass

from sfm.models.graphormer.graphormer_config import GraphormerConfig


@dataclass
class PFMConfig(GraphormerConfig):
    model_type = "pfm"

    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(args)
