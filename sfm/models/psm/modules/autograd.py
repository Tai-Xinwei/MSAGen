# -*- coding: utf-8 -*-
from abc import ABC, ABCMeta, abstractmethod

import torch


class GradientHead(torch.nn.Module):
    def __init__(self, force_std=1.0):
        super(GradientHead, self).__init__()
        self.force_std = force_std

    def forward(self, energy, pos):
        grad_outputs = [torch.ones_like(energy)]

        grad = torch.autograd.grad(
            outputs=energy,
            inputs=pos,
            grad_outputs=grad_outputs,
            create_graph=self.training,
            allow_unused=True,
        )
        print(grad)
        exit()
        force_grad = grad[0] / self.force_std

        if force_grad is not None:
            forces = torch.neg(force_grad)

        return forces
