# MSAGen Repository


## Pre-commit hooks
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

## Running and installation

For submitting jobs on clusters, there is no additional installation steps since the `sfm` conda envrionment is already in the docker image. In amlt yaml file, an exemplary job YAML can be:

```yaml
# ...... environmental variables .....
env_defaults:
  WANDB_API_KEY: ${WANDB_API_KEY}  # access key(s)

# ...... target information .....
environment:
  image: ai4s-sfm:20240531.170731
  registry: msrmoldyn.azurecr.io
  username: msrmoldyn
  setup:
  # activate sfm environment and build cython extention
  - eval "$$(conda shell.bash hook)" && conda activate sfm
  - python setup_cython.py build_ext --inplace
  # Optional: install custom packages that not in the docker image
  # - pip install torcheval

# ...... storage section ......
storage:
  blob:
    storage_account_name: sfmdatawestus # Storage account, select appropriate one
    container_name: <container_name> # Container name on the storage account
    mount_dir: /data  # mount to here

# ...... code section ......
jobs:
  tags:
  - 'ProjectID: xxxxxx'
  # Singularity specific parameters
  sku: 2x192G8  # for example, 2 Nodes each with 192 GB RAM and 8 GPUs, modify according to the system
  sla_tier: premium   # premium/standard/basic - basic could be preempted, but with more capacity
  mpi: true
  # Since Amulet v9.4, Amulet no longer tells AML to use mpirun to spawn processes by default
  # Setting "mpi: True" will ask AML to spawn processes with mpirun. Requires openmpi installed in the image.
  process_count_per_node: 1
  command:
  # model running script, refer to Usage section.
  - eval "$$(conda shell.bash hook)" && conda activate sfm
  - bash ./scripts/xxx/xxxx.sh
  submit_args:
    env:
      SHARED_MEMORY_PERCENT: 1.0
      # skip Singularity verification
      AMLT_DOCKERFILE_TEMPLATE: "none"
      # Singularity User Access Identity
      _AZUREML_SINGULARITY_JOB_UAI: ${AZUREML_SINGULARITY_JOB_UAI}
```

> **NOTE:** In order to run the trainings in a Singularity cluster with AMD GPUs, e.g., `huashanvc4`, `huashanvc5`, `msroctovc` , `msroctobasicvc` and `whitney02`, please use the images specific to AMD by replacing environment/image, as:
> ```yaml
> environment:
>   image: ai4s-sfm/amd:20241111.140607-rocm624
> ```

> **NOTE:** Refer to the [Amulet's documentation](https://amulet-docs.azurewebsites.net/main/setup.html#using-identity-based-access-recommended) for more information on `_AZUREML_SINGULARITY_JOB_UAI` environmental variable.

For **local** development, create a conda environment from the YAML files in [`./install`](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_git/SFM_framework?path=/install/) folder and install other packages that need compilation:

```bash
# create and activate conda environment for NVIDIA
conda env create -f ./install/environment.yaml -n sfm -y
conda activate sfm

# build cython extention
python setup_cython.py build_ext --inplace

# optional for NVIDIA GPUs: install NVIDIA apex locally, which may take 20 minutes
bash install/install_third_party.sh
# If your machine has a different version of CUDA, you may get the apex compile error.
# In such case, run
conda install nvidia/label/cuda-12.1.0::cuda-toolkit
conda install nvidia/label/cuda-12.1.0::cuda-cudart
```
> **NOTE:** For AMD use [`./install/helper_AMD/environment.yaml`](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_git/SFM_framework?path=/install/helper_AMD/environment.yaml) instead when creating the conda environment.


## Usage

To run models on your local machine, try bash scripts in [`./scripts`](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_git/SFM_framework?path=/scripts) folder, e.g.:

```bash
bash ./scripts/pfm/pretrain_pfm.sh
```

To run jobs on a cluster, use a yaml file in the [`./amlt`](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_git/SFM_framework?path=/amlt) folder, e.g.:

```bash
amlt run ./amlt/pfm/BFM3B.yaml BFM3B
```

> More details can be found in the [documentation page](https://aka.ms/A4SFramework).

## Docker Image

> **These images are Singularity compatible and have built-in `sfm` conda environment with pre-installed packages.**

CUDA 11.7:
- `msrmoldyn.azurecr.io/pj/sfm:cu117`

CUDA 12.1, Python 3.11:
 - `msrmoldyn.azurecr.io/ai4s-sfm:20240531.170731`

CUDA 12.4, Python 3.11:
 - `msrmoldyn.azurecr.io/ai4s-sfm:20241112.142059-cu124`

ROCm 6.2.4, Python 3.11:
 - `msrmoldyn.azurecr.io/ai4s-sfm/amd:20241111.140607-rocm624`
> **NOTE:** AMD images are updated regularly for best performance and compatibility, please check the latest tag from [msrmoldyn](https://ms.portal.azure.com/#view/Microsoft_Azure_ContainerRegistries/RepositoryBlade/id/%2Fsubscriptions%2F3eaeebff-de6e-4e20-9473-24de9ca067dc%2FresourceGroups%2Fshared_infrastructure%2Fproviders%2FMicrosoft.ContainerRegistry%2Fregistries%2Fmsrmoldyn/repository/ai4s-sfm%2Famd).

## Data

Refer to [ms.portal.azure.com](https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/c5b6f974-9372-41db-b1e7-86608c3a6afd/resourceGroups/SFM/resourcevisualizer) with your SC account to get the current SFM storage accounts.


## Documentation

[For more details, check the documentation of the framework](https://aka.ms/A4SFramework)
