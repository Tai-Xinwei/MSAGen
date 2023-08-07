#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo "pip install start"
# # # python=3.9, cuda 11.7

# For Megatron Tensor Parallel
python -m pip install pybind11
pip install regex
pip install einops

cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
