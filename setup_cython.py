# -*- coding: utf-8 -*-

import numpy
import subprocess

# import order matters!
# see https://stackoverflow.com/questions/21594925/error-each-element-of-ext-modules-option-must-be-an-extension-instance-or-2-t

# isort:skip_file
from setuptools import setup
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    rocm_version = subprocess.run(
        ["apt", "show", "rocm-libs", "-a"], capture_output=True, text=True
    ).stdout
    if "Version: 6.3" in rocm_version:
        with open(
            "sfm/data/preprocess/graphormer_preprocess_cuda_kernel.cu", "r+"
        ) as f:
            content = f.read().replace(".type()", ".scalar_type()")
            f.seek(0)
            f.write(content)
            f.truncate()
        print("Modifications for ROCm 6.3.x is done")
except:
    pass

extensions = [
    "sfm/data/mol_data/algos.pyx",
    "sfm/data/data_utils_fast.pyx",
    "sfm/data/prot_data/token_block_utils_fast.pyx",
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
    include_dirs=[numpy.get_include()],
)

setup(
    name="graphormer_preprocess_cuda",
    ext_modules=[
        CUDAExtension(
            "graphormer_preprocess_cuda",
            [
                "sfm/data/preprocess/graphormer_preprocess_cuda.cpp",
                "sfm/data/preprocess/graphormer_preprocess_cuda_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
