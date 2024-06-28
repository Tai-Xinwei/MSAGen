# -*- coding: utf-8 -*-
import torch
from transformers import GPT2Tokenizer

from sfm.logging import logger


# Custom dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts,
        args=None,
    ):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.inputs = [
            self.tokenizer.encode(text, return_tensors="pt").flatten() for text in texts
        ]
        self.args = args
        self.collate_fn = self.collate
        self.sequence_length = args.max_length

    def __len__(self) -> list:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.inputs[idx]

    # added for dynamic batch sampler to work
    def num_tokens(self, idx: int) -> int:
        return self.inputs[idx].size().numel()

    def collate(self, samples: list) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(
            samples, batch_first=True, padding_value=0
        )
