from typing import List, Union

import numpy as np
from dataset import Data
from molecule import Molecule
from torch import LongTensor
from transformers import PreTrainedTokenizer

TOKEN_SEQ_T = Union[LongTensor, np.array]
MOL_POS_T = Union[LongTensor, List[int], np.array]


class Text(Data):
    def __init__(self, token_seq: TOKEN_SEQ_T, **kwargs) -> None:
        super().__init__()
        self.token_seq = token_seq
        for key, value in kwargs:
            self.__dict__[key] = value

    def from_raw_text(cls, text: str, tokenizer: PreTrainedTokenizer) -> "Text":
        tokenized = tokenizer.tokenize(text=text)
        token_seq = tokenized.input_ids[0]
        return cls(token_seq=token_seq)


class MixedText(Text):
    def __init__(
        self,
        token_seq: TOKEN_SEQ_T,
        mol_poses: MOL_POS_T = None,
        molecules: List[Molecule] = None,
        **kwargs,
    ) -> None:
        super().__init__(token_seq, **kwargs)
        assert len(mol_poses) == len(
            molecules
        ), "Numbers of molecules in mol_poses and molecules do not match."

    def from_json(cls, json_text: str, tokenizer: PreTrainedTokenizer) -> "MixedText":
        raise NotImplementedError
