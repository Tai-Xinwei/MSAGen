# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class PartialGradEmbedding(nn.Module):
    def __init__(self, embedding_fn, new_embedding_cutoff=-1):
        """
        This module is used to only update the gradient of the new embeddings
        i.e. the embeddings with index >= new_embedding_cutoff
        Args:
            embedding_fn: a nn.Embedding object
            new_embedding_cutoff: the cutoff index for the new embedding
        """
        super().__init__()
        self.embedding_fn = embedding_fn
        self.cutoff = new_embedding_cutoff

    def forward(self, x):
        w = (x > self.cutoff) * 1.0
        w = w.unsqueeze(-1)
        h = w * self.embedding_fn(x) + (1 - w) * self.embedding_fn(x).detach()
        return h


if __name__ == "__main__":
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 7, 8]])
    emb_fn = nn.Embedding(10, 3)
    loss = PartialGradEmbedding(emb_fn, new_embedding_cutoff=5)(input_ids).sum()
    loss.backward()
    print(emb_fn.weight.grad)

    """
    tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [0., 0., 0.]])
    """
