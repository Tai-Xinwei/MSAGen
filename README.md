[![Build Status](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_apis/build/status%2FPython%20Unit%20Tests?branchName=main)](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_build/latest?definitionId=1&branchName=main)

# SFM Repository


## Pre-commit Hooks
Run pre-commit hooks to ensure that the code is formatted correctly and passes all tests before committing.

To install pre-commit, run the following command:

```bash
pip install pre-commit
```

Run the following command from the root of this repository:

```bash
pre-commit install
```

To run pre-commit manually on all files in your repository, use the following command:

```bash
pre-commit run --all-files
```

## Installation

For submitting jobs on clusters, there is no additional installation steps since the `sfm` conda envrionment is already in the docker image. In amlt yaml file, an exemplary job command section can be:

```yaml
# ......

environment:
  image: yaosen/sfm-cuda:py39-torch2.2.2-cuda12.1
  registry: msroctocr.azurecr.io
  username: msroctocr

# ......

# activate sfm environment
- eval "$$(conda shell.bash hook)" && conda activate sfm
# install packages (check tools/docker_image/conda_install.sh for pre-installed packages)
- pip install torcheval
# model running script, refer to Usage section.
- bash ./scripts/xxx/xxxx.sh
```


For development / debug, create conda environment from the YAML files in `./install` folder:

```bash
conda env revmoe -n sfm
conda env create -f ./install/py39-torch2.2.2-cuda12.1.yaml -n sfm
# conda env create -f ./install/py310-torch2.2.2-cuda12.1.yaml -n sfm

# build cython extention
python setup_cython.py build_ext --inplace

# if you want to install NVIDIA apex, run the following scripts
# NOTE: it can take over 20 min for compilation
cwd=$(pwd)
git clone https://github.com/NVIDIA/apex /tmp/apex && cd /tmp/apex
MAX_JOBS=0 pip install -v --disable-pip-version-check \
    --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
cd $cwd
rm -rf /tmp/apex
```

## Usage

To run models on your local machine, try bash scripts in `./scripts` folder, e.g.:

```bash
bash ./scripts/pfm/pretrain_pfm.sh
```

To run jobs on cluster, use yaml file in the ./amlt folder, e.g.:

```bash
amlt run ./amlt/pfm/BFM3B.yaml BFM3B
```

More details can be found in the documentation page.

## Docker Image
CUDA 11.7 docker image:
- `itpeus4cr.azurecr.io/pj/mfmds:20230207_b`

> Call for volunteers: help submit tasks to test the compatibility of the CUDA 12.1 images, especially for python 3.10 image.

CUDA 12.1 docker image:

 - `msroctocr.azurecr.io/yaosen/sfm-cuda:py39-torch2.2.2-cuda12.1`
 - `msroctocr.azurecr.io/yaosen/sfm-cuda:py310-torch2.2.2-cuda12.1`

For CUDA 12.1 docker images, they are Singularity compatible and have built-in `sfm` conda environment with pre-installed packages (check `tools/docker_image/conda_install.sh` for details). For special needs, refer to `tools/docker_image/build.sh` to build another image. Note: currently, Python 3.11 is not supported because `apex` failed to compile.


## Data

TBA, all data of SFM will be moved to one place

## Documentation

[ For more details, check the documentation of the framework](https://aka.ms/A4SFramework)
