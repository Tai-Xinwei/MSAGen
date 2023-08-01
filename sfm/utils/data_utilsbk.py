# -*- coding: utf-8 -*-
import torch


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    elif hasattr(batch, "__dataclass_fields__"):
        return type(batch)(
            **{k: move_to_device(v, device) for k, v in batch.__dict__.items()}
        )
    else:
        return batch
