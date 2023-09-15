# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from sfm.data.dec_data.datasets import TokenType


@dataclass
class HiddenState:
    """
    The x_dict reprensent a jadged tensor because the each token can have differernt dim
    """

    # Each type holds a tensor of shape [N, *] where N is the number of this type of tokens
    x_dict: Dict[TokenType, torch.Tensor]

    token_type_mask: torch.Tensor

    @property
    def batch_size(self):
        return self.token_type_mask.shape[0]

    @property
    def seq_len(self):
        return self.token_type_mask.shape[1]

    @property
    def dtype(self):
        return self.x_dict[TokenType.Text].dtype

    @property
    def device(self):
        return self.x_dict[TokenType.Text].device

    @property
    def logits(self):
        # For generation, the generator expecta a logit tensor of shape [B, T, V]
        # However, here the V could be different for each token type
        # Thus, we need to extend to the max V and fill the rest with -inf

        max_vocab_size = max([v.shape[-1] for v in self.x_dict.values()])
        ret = torch.empty(
            (
                self.batch_size,
                self.seq_len,
                max_vocab_size,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        torch.fill_(ret, torch.finfo(self.dtype).min)

        for k, v in self.x_dict.items():
            ret[self.token_type_mask == k.value][:, : v.shape[-1]] = v

        return ret

    def __add__(self, other):
        return HiddenState(
            x_dict={k: v + other.x_dict[k] for k, v in self.x_dict.items()},
            token_type_mask=self.token_type_mask,
        )

    # Create a copy of this object and update one item of x_dict
    def update_x_dict(self, token_type: TokenType, x: torch.Tensor):
        x_dict = {k: v for k, v in self.x_dict.items()}
        x_dict[token_type] = x
        return HiddenState(x_dict, self.token_type_mask)

    def apply_all_types_mapping(self, fn: nn.ModuleDict, **kwargs):
        x_dict_new = {k: fn[k.name](x, **kwargs) for k, x in self.x_dict.items()}
        return HiddenState(x_dict_new, self.token_type_mask)

    def to_dense(self) -> torch.Tensor:
        # All tensor must have the dim [N, *, ...]
        # If we want to concat them, we need to make sure that the last dims is the same
        for k, v in self.x_dict.items():
            if v.numel() != 0:
                extra_dims = v.shape[1:]
                break
        else:
            return torch.empty(0, device=self.device, dtype=self.dtype)

        # Check all dim are the same
        for k, v in self.x_dict.items():
            # check empty_first:
            if v.numel() == 0:
                continue

            assert v.shape[1:] == extra_dims

        ret = torch.zeros(
            self.batch_size,
            self.seq_len,
            *extra_dims,
            device=self.x_dict[TokenType.Text].device,
            dtype=self.x_dict[TokenType.Text].dtype,
        )
        for k, v in self.x_dict.items():
            ret[self.token_type_mask == k.value] = v

        return ret

    @classmethod
    def from_dense(cls, x: torch.Tensor, token_type_mask: torch.Tensor):
        x_dict = {}
        for k in TokenType:
            x_dict[k] = x[token_type_mask == k.value]
        return cls(x_dict, token_type_mask)

    def to_single_type_batch(self, token_type: TokenType, attention_mask: torch.Tensor):
        """
        Convert this ragged batch to a batch (BxTxC) with only one token type.
        Other kinds of tokens are masked out.
        This is useful when we want to apply a layer to only one token type.
        """
        embed_dim = self.x_dict[token_type].shape[-1]

        batch = torch.zeros(
            (self.batch_size, self.seq_len, embed_dim),
            device=self.x_dict[token_type].device,
            dtype=self.x_dict[token_type].dtype,
        )

        batch[self.token_type_mask == token_type.value] = self.x_dict[token_type]

        # [B, 1, SRC_LEN, TGT_LEN], the 2nd dim head
        attention_mask_new = attention_mask.clone()

        not_to_attend_mask = (self.token_type_mask != token_type.value)[
            :, None, None, :
        ].expand_as(attention_mask_new)

        if attention_mask.dtype == torch.bool:
            # see `mask_to_float`.
            attention_mask_new[not_to_attend_mask] = True
        else:
            val = torch.finfo(attention_mask.dtype).min
            attention_mask_new[not_to_attend_mask] = val

        return batch, attention_mask_new

    def to_tuple(self):
        ret = []
        for k in TokenType:
            ret.append(self.x_dict[k])
        ret.append(self.token_type_mask)
        return tuple(ret)

    @classmethod
    def from_tuple(cls, x):
        x_dict = {}
        for i, k in enumerate(TokenType):
            x_dict[k] = x[i]
        return cls(x_dict, x[-1])

    def update_single_type_batch(self, token_type: TokenType, x: torch.Tensor):
        """
        Update the x_dict with the new batch, after applying a layer to only one type.
        """
        x = x[self.token_type_mask == token_type.value]
        return self.update_x_dict(token_type, x)
