# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Optional

from transformers.models.mixtral.configuration_mixtral import MixtralConfig

from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig


@dataclass
class ScigptMoeConfig(MixtralConfig, DistributedTrainConfig):
    model_type: str = "scigpt_moe"

    learnable_cutoff: int = 0
