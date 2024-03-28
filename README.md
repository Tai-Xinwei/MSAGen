[![Build Status](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_apis/build/status%2FPython%20Unit%20Tests?branchName=main)](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_build/latest?definitionId=1&branchName=main)

# SFM Repository


## Pre-commit Hooks
Run pre-commit hooks to ensure that the code is formatted correctly and passes all tests before committing.

To install pre-commit, run the following command:
```
pip install pre-commit
```

Run the following command from the root of this repository:
```
pre-commit install
```

To run pre-commit manually on all files in your repository, use the following command:
```
pre-commit run --all-files
```

## Hai1 Cluster Blocklist

When submit multi-node job in Hai1 Cluster, avoid using following nodes due to IB port error:
```
Severe: GCRHYP3C103, GCRHYP3C112, GCRHYP3C142, GCRHYP3C149, GCRHYP3C224, GCRHYP3C321, GCRHYP3C324, GCRHYP3C336 (Not use),
Medium: GCRHYP3C257, GCRHYP3C314, GCRHYP3C342, GCRHYP3C346 (OK to use with short job like 1-3 days, do not use it with pretraining),
Mild: GCRHYP3C108, GCRHYP3C218, GCRHYP3C225, GCRHYP3C253 (Good to use, port error happens rarely)
```

## Documentation

[Check the documentation of the framework](https://aka.ms/A4SFramework)


## Installation

To install the dependencies, run the following command:
```
eval "$(conda shell.bash hook)" && conda create -n sfm python=3.9 && conda activate sfm
bash ./install/install.sh
```

To install the dependencies for the Tensor parallel, run the following command:
```
git clone https://github.com/NVIDIA/apex
```

Replace the following line in `apex/setup.py`:
```if (bare_metal_version != torch_binary_version):```
with
```if (0 and bare_metal_version != torch_binary_version):```

Then run
```
bash ./install/install_megatron.sh
```
