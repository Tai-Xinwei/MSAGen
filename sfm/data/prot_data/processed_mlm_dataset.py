# -*- coding: utf-8 -*-
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Batch:
    x: torch.Tensor  # input to model, (B, T)
    y: torch.Tensor  # target of model, (B, T)
    mask: torch.Tensor  # mask for the selected tokens, (B, T)
    pad_mask: torch.Tensor  # mask for the padded tokens, (B, T)


class ProcessedMlmDataset(Dataset):
    def __init__(
        self,
        path: str,
        bos_idx: int,
        eos_idx: int,
        pad_idx: int,
        mask_idx: int,
        vocab_size: int,
        mask_prob: float,
        leave_unmasked_prob: float,
        random_token_prob: float,
    ):
        super().__init__()
        self.path = path
        self.data = np.load(path, mmap_mode="r")
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.vocab_size = vocab_size

        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob

    def __len__(self):
        return len(self.data)

    def __str__(self) -> str:
        text = "ProcessedMlmDataset:\n"
        text += f"  path: {self.path}\n"
        text += f"  len: {len(self)}\n"
        text += f"  bos_idx: {self.bos_idx}\n"
        text += f"  eos_idx: {self.eos_idx}\n"
        text += f"  pad_idx: {self.pad_idx}\n"
        text += f"  mask_idx: {self.mask_idx}\n"
        text += f"  vocab_size: {self.vocab_size}\n"
        text += f"  mask_prob: {self.mask_prob}\n"
        text += f"  leave_unmasked_prob: {self.leave_unmasked_prob}\n"
        text += f"  random_token_prob: {self.random_token_prob}\n"
        return text

    def __getitem__(self, index) -> Batch:
        x = np.copy(self.data[index])
        y = np.copy(x)

        # mask tokens
        can_be_masked = (x != self.bos_idx) & (x != self.eos_idx) & (x != self.pad_idx)
        mask = np.zeros_like(x, dtype=bool)
        mask[np.random.rand(*x.shape) < self.mask_prob] = True
        mask[~can_be_masked] = False

        x[mask] = self.mask_idx

        # leave unmasked or random token
        special_mask = np.random.rand(*y.shape)

        leave_unmasked = special_mask < self.leave_unmasked_prob
        x[mask & leave_unmasked] = y[mask & leave_unmasked]

        random_token = (self.leave_unmasked_prob < special_mask) & (
            special_mask < (self.leave_unmasked_prob + self.random_token_prob)
        )

        valid_values = [
            i
            for i in range(self.vocab_size)
            if i not in [self.bos_idx, self.eos_idx, self.pad_idx, self.mask_idx]
        ]
        x[mask & random_token] = np.random.choice(
            valid_values, size=(mask & random_token).sum()
        )

        return Batch(
            x=torch.from_numpy(x.astype(int)),
            y=torch.from_numpy(y.astype(int)),
            mask=torch.from_numpy(mask.astype(bool)),
            pad_mask=torch.from_numpy((x != self.pad_idx).astype(bool)),
        )

    def collate(self, samples):
        x = torch.stack([s.x for s in samples], dim=0)
        y = torch.stack([s.y for s in samples], dim=0)
        mask = torch.stack([s.mask for s in samples], dim=0)
        pad_mask = torch.stack([s.pad_mask for s in samples], dim=0)
        return Batch(x=x, y=y, mask=mask, pad_mask=pad_mask)
