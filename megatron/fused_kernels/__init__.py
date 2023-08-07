# -*- coding: utf-8 -*-
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
import pathlib
import subprocess

import torch
from torch.utils import cpp_extension
from sfm.logging import logger
# Setting this param to a list has a problem of generating different
# compilation commands (with diferent order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicity in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def load(args):
    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []
    if torch.version.hip is None:
        _, bare_metal_major, bare_metal_minor = _get_cuda_bare_metal_version(
            cpp_extension.CUDA_HOME
        )
        if int(bare_metal_major) >= 11:
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_80,code=sm_80")
        elif int(bare_metal_major) >= 12:
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    _create_build_dir(buildpath)
    logger.info("Building fused kernels in {}".format(buildpath))
    # Helper function to build the kernels.
    def _cpp_extention_load_helper(
        name, sources, extra_cuda_flags, extra_include_paths
    ):
        if torch.version.hip is not None:
            extra_cuda_cflags = ["-O3"] + extra_cuda_flags + cc_flag
        else:
            extra_cuda_cflags = (
                ["-O3", "-gencode", "arch=compute_70,code=sm_70", "--use_fast_math"]
                + extra_cuda_flags
                + cc_flag
            )

        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=[
                "-O3",
            ],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths,
            verbose=(args.rank == 0),
        )

    # ==============
    # Fused softmax.
    # ==============

    if torch.version.hip is not None:
        extra_include_paths = [os.path.abspath(srcpath)]
    else:
        extra_include_paths = []

    if args.masked_softmax_fusion:
        if torch.version.hip is not None:
            extra_cuda_flags = [
                "-D__HIP_NO_HALF_OPERATORS__=1",
                "-D__HIP_NO_HALF_CONVERSIONS__=1",
            ]
        else:
            extra_cuda_flags = [
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ]

        # Upper triangular softmax.
        sources = [
            srcpath / "scaled_upper_triang_masked_softmax.cpp",
            srcpath / "scaled_upper_triang_masked_softmax_cuda.cu",
        ]
        _cpp_extention_load_helper(
            "scaled_upper_triang_masked_softmax_cuda",
            sources,
            extra_cuda_flags,
            extra_include_paths,
        )
        logger.info("Built Upper triangular softmax")

        # Masked softmax.
        sources = [
            srcpath / "scaled_masked_softmax.cpp",
            srcpath / "scaled_masked_softmax_cuda.cu",
        ]
        _cpp_extention_load_helper(
            "scaled_masked_softmax_cuda", sources, extra_cuda_flags, extra_include_paths
        )
        logger.info("Built Masked softmax")

        # Softmax
        sources = [srcpath / "scaled_softmax.cpp", srcpath / "scaled_softmax_cuda.cu"]
        _cpp_extention_load_helper(
            "scaled_softmax_cuda", sources, extra_cuda_flags, extra_include_paths
        )
        logger.info("Built Softmax")


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
