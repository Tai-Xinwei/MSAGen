# -*- coding: utf-8 -*-
from dataclasses import asdict, dataclass


@dataclass
class TutConfig:
    max_length: int = 1024

    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__(args)
        for k, v in asdict(self).items():
            if hasattr(args, k):
                setattr(self, k, getattr(args, k))
