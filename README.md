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

For submitting jobs on clusters, there is no additional installation steps since the `sfm` conda envrionment is already in the docker image. In amlt yaml file, an exemplary job YAML can be:

```yaml
# ...... target information .....

environment:
  image: ai4s-sfm:20240429.081857
  registry: msroctocr.azurecr.io  # or msrresrchcr.azurecr.io
  username: msroctocr  # or msrresrchcr

# ...... storage & code section ......
jobs:
  tags:
  - 'ProjectID: xxxxxx'
  # Singularity specific parameters
  priority: high      # may spend more time in the queue, default: medium
  sla_tier: premium   # may be paused any time, but more capacity available
  mpi: true
  # Since Amulet v9.4, Amulet no longer tells AML to use mpirun to spawn processes by default
  # Setting "mpi: True" will ask AML to spawn processes with mpirun. Requires openmpi installed in the image.
  process_count_per_node: 1
  command:
  # activate sfm environment and build cython extention
  - eval "$$(conda shell.bash hook)" && conda activate sfm
  - python setup_cython.py build_ext --inplace

  # Optional: install custom packages that not in the docker image
  # - pip install torcheval

  # model running script, refer to Usage section.
  - bash ./scripts/xxx/xxxx.sh
  submit_args:
    env:
      {SHARED_MEMORY_PERCENT: 1.0}
      # singularity version shm_size
```


For **local** development, create a conda environment from the YAML files in `./install` folder and install other packages that need compilation:

```bash
# create and activate conda environment
conda env remove -n sfm
conda env create -f ./install/environment.yaml -n sfm
conda activate sfm

# build cython extention
python setup_cython.py build_ext --inplace

# optional: install NVIDIA apex locally, which may take 20 minutes
bash install/install_third_party.sh
# If your machine has a different version of CUDA, you may get the apex compile error.
# In such case, run
conda install nvidia/label/cuda-12.1.0::cuda-toolkit
conda install nvidia/label/cuda-12.1.0::cuda-cudart
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
CUDA 11.7:
- `itpeus4cr.azurecr.io/pj/mfmds:20230207_b`

CUDA 12.1, Python 3.11.9:
 - `msroctocr.azurecr.io/ai4s-sfm:20240429.081857`

For CUDA 12.1 docker images, they are Singularity compatible and have built-in `sfm` conda environment with pre-installed packages.


## Data

TBA, all data of SFM will be moved to one place

## Documentation

[ For more details, check the documentation of the framework](https://aka.ms/A4SFramework)
