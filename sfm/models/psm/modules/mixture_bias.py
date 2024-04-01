# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn


class PSMBias(nn.Module):
    """
    Class for the invariant encoder bias in the PSM model.
    """

    def __init__(self):
        """
        Initialize the PSMBias class.
        """
        super(PSMBias, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PSMBias class.
        """

        return x
