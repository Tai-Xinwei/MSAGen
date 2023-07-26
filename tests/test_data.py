# -*- coding: utf-8 -*-
from sfm.data.molecule import Molecule
from sfm.data.gen_data.dataset import TextMixedWithEntityData, EntityType, TokenIdRange, TextMixedWithEntityDataset

import torch

def test_create_molecule():
    Molecule()

def test_text_mixed_with_entity_data():
    token_seqs = [
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 101, 102]),
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 100]),
    ]
    
    entity_id_rage = {
        EntityType.SMILES: TokenIdRange(100, 103),
    }
    
    texts = [TextMixedWithEntityData(token_seq, entity_id_rage) for token_seq in token_seqs]
    
    dataset = TextMixedWithEntityDataset(texts, pad_idx=0)
    
    batch = dataset.collate(texts)
    
    expected_batched_tokens = torch.LongTensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 101, 102],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 100, 0],
    ])
    
    assert torch.equal(batch.token_seq, expected_batched_tokens)
    
    expected_entity_mask = torch.BoolTensor([
        [False, False, False, False, False, False, False, False, False, True, True, True],
        [False, False, False, False, False, False, False, False, False, True, True, False],
    ])
    
    assert torch.equal(batch.entity_mask(EntityType.SMILES), expected_entity_mask)
