# VLLM NLM Inference Engine

## Create an Isolated Env for VLLM

### Step 1: Create the Python 3.10 Environment
```bash
conda create -n vllm python=3.10
```
This command creates a new environment called `vllm` with Python 3.10.

### Step 2: Activate the Environment
```bash
conda activate vllm
```

### Step 3: Install `vllm`
You can install `vllm` using `pip` inside the Conda environment:

```bash
pip install vllm
```

### Step 4: Verify Installation
To ensure everything is set up correctly, run the following to check the Python version and confirm `vllm` is installed:

```bash
python --version
pip show vllm
```


## Prepare data
You need to have your input data in this structure:

```
sfm/tasks/nlm/
└── eval/
    └── inputs/
        └── nlm_data/
            ├── molecules_test/
            ├── molecules_test_mini/
            └── science_test/
                ├── test.bace.instruct.tsv
                ├── test.bbbp.instruct.tsv
                ├── test.desc2mol.tsv
                ├── test.hERG.tsv
                ├── test.mol2desc.tsv
                ├── test.molinstruct.reaction.tsv
                ├── test.raw.i2s_i.txt
                ├── test.raw.i2s_s.txt
                ├── test.raw.s2i_i.txt
                ├── test.raw.s2i_s.txt
                ├── test.uspto50k.reaction.osmi.tsv
                └── test.uspto50k.retro.osmi.tsv
```

## Prepare HuggingFace Format Checkpoint
You will need to generate HF-format checkpoint with `moe_inference_module.py`. Store the converted checkpoint in a local path `${NLM_LOCAL_PATH}`, do not utilize `blobfuse` location.

## Inference with VLLM-NLM Engine
Execute this in NLM home, do note that `${NLM_LOCAL_PATH}` is the path to your local HF-format NLM ckpt:
```
# In vllm env
bash ./scripts/nlm/eval/run_nlm_moe_vllm_inference.sh
```

## Inference on AMD

VLLM-NLM inference with HuggingFace Format Checkpoint conversion can be run on [`huashanvc4`](https://ml.azure.com/virtualClusters/22da88f6-1210-4de2-a5a3-da4c7c2a1213/gcr-singularity/huashanvc4/Overview?tid=72f988bf-86f1-41af-91ab-2d7cd011db47) VC from Singularity with AMD MI250x.

For starters, please refer to this [**loop page**](https://loop.cloud.microsoft/p/eyJ3Ijp7InUiOiJodHRwczovL21pY3Jvc29mdC5zaGFyZXBvaW50LmNvbS8%2FbmF2PWN6MGxNa1ltWkQxaUlWOTRNazVPUVRCYVFqQjVVRWRSUVRsTmFreHRiQzFvVFZST1pXZGtPR3hDZEZkbFVVaDNkRVF0TjJWdmVEZFNlblo1UlcxVWNsSmhhRE5DVmpoVGVrY21aajB3TVZnMldrOUZOalkwUTBnMlZraFBWRmxaVGtKWlFqTTNXVnBNTWtzM1ZVUlFKbU05Sm1ac2RXbGtQVEUlM0QiLCJyIjpmYWxzZX0sInAiOnsidSI6Imh0dHBzOi8vbWljcm9zb2Z0LnNoYXJlcG9pbnQuY29tL3NpdGVzLzlkOWE4MzE0LTZlYjEtNGNiOC1hNTFlLTkxYmNiOWJmNWEwMT9uYXY9Y3owbE1rWnphWFJsY3lVeVJqbGtPV0U0TXpFMExUWmxZakV0TkdOaU9DMWhOVEZsTFRreFltTmlPV0ptTldFd01TWmtQV0loWDNneVRrNUJNRnBDTUhsUVIxRkJPVTFxVEcxc0xXaE5WRTVsWjJRNGJFSjBWMlZSU0hkMFJDMDNaVzk0TjFKNmRubEZiVlJ5VW1Gb00wSldPRk42UnlabVBUQXhXRFphVDBVMldVWktXVVJUUjBsR04wazFRMXBaVWt4R1dWSlBXVFpYTWtjbVl6MG1abXgxYVdROU1RJTNEJTNEIiwiciI6ZmFsc2V9LCJpIjp7ImkiOiJmZmRkODAzYS1lZjYxLTRmYWYtODM0OS00ZTk5ZjRhNWZhODkifX0%3D), which also includes installing `vllm` on AMD.

We have already created a seperate image for this purpose in [`msrmoldyn CR`](https://ms.portal.azure.com/#view/Microsoft_Azure_ContainerRegistries/ImageMetadataBlade/registryId/%2Fsubscriptions%2F3eaeebff-de6e-4e20-9473-24de9ca067dc%2FresourceGroups%2Fshared_infrastructure%2Fproviders%2FMicrosoft.ContainerRegistry%2Fregistries%2Fmsrmoldyn/repositoryName/ai4s-sfm%2Fvllm%2Famd/tag/20241001.115052):

```bash
docker pull msrmoldyn.azurecr.io/ai4s-sfm/vllm/amd:20241001.115052
```

The complete inference is automated, please use this Amulet configuration [/amlt/nlm/eval/run_nlm_moe_vllm_inference_mi250x.yaml](https://dev.azure.com/AI4ScienceSFM/SFM_framework/_git/SFM_framework?path=/amlt/nlm/eval/run_nlm_moe_vllm_inference.yaml). You can submit this inference job to the AMD VC after you follow the ***Access*** subsection in [**loop page**](https://loop.cloud.microsoft/p/eyJ3Ijp7InUiOiJodHRwczovL21pY3Jvc29mdC5zaGFyZXBvaW50LmNvbS8%2FbmF2PWN6MGxNa1ltWkQxaUlWOTRNazVPUVRCYVFqQjVVRWRSUVRsTmFreHRiQzFvVFZST1pXZGtPR3hDZEZkbFVVaDNkRVF0TjJWdmVEZFNlblo1UlcxVWNsSmhhRE5DVmpoVGVrY21aajB3TVZnMldrOUZOalkwUTBnMlZraFBWRmxaVGtKWlFqTTNXVnBNTWtzM1ZVUlFKbU05Sm1ac2RXbGtQVEUlM0QiLCJyIjpmYWxzZX0sInAiOnsidSI6Imh0dHBzOi8vbWljcm9zb2Z0LnNoYXJlcG9pbnQuY29tL3NpdGVzLzlkOWE4MzE0LTZlYjEtNGNiOC1hNTFlLTkxYmNiOWJmNWEwMT9uYXY9Y3owbE1rWnphWFJsY3lVeVJqbGtPV0U0TXpFMExUWmxZakV0TkdOaU9DMWhOVEZsTFRreFltTmlPV0ptTldFd01TWmtQV0loWDNneVRrNUJNRnBDTUhsUVIxRkJPVTFxVEcxc0xXaE5WRTVsWjJRNGJFSjBWMlZSU0hkMFJDMDNaVzk0TjFKNmRubEZiVlJ5VW1Gb00wSldPRk42UnlabVBUQXhXRFphVDBVMldVWktXVVJUUjBsR04wazFRMXBaVWt4R1dWSlBXVFpYTWtjbVl6MG1abXgxYVdROU1RJTNEJTNEIiwiciI6ZmFsc2V9LCJpIjp7ImkiOiJmZmRkODAzYS1lZjYxLTRmYWYtODM0OS00ZTk5ZjRhNWZhODkifX0%3D).

### Dockerfile specific for AMD

```Dockerfile
FROM msrmoldyn.azurecr.io/ai4s-sfm/amd:20240917.094627

ENV PYTORCH_ROCM_ARCH=gfx90a

USER root

RUN apt update && apt upgrade -y

RUN --mount=type=bind,source=install,target=/tmp/install \
    eval "$(conda shell.bash hook)" && \
    conda activate sfm && \
    pip install 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl' && \
    cd $(pip show bitsandbytes | grep Location | awk -F': ' '{print $2}')/bitsandbytes && \
    ln -s libbitsandbytes_rocm62.so libbitsandbytes_rocm60.so && \
    ln -s libbitsandbytes_rocm62.so libbitsandbytes_rocm60_nohipblaslt.so && \
    ln -s libbitsandbytes_rocm62.so libbitsandbytes_rocm61_nohipblaslt.so

RUN --mount=type=bind,source=install,target=/tmp/install \
    eval "$(conda shell.bash hook)" && \
    conda create -n vllm python=3.11 -yq && \
    conda activate vllm && \
    pip install /opt/rocm/share/amd_smi && \
    bash /tmp/install/helper_AMD/install_vllm.sh $PYTORCH_ROCM_ARCH

RUN rm -rf /var/lib/apt/lists/* && \
    apt purge --auto-remove -y && \
    apt clean -y && \
    conda clean --all --yes && \
    pip cache purge
```
#### vLLM for AMD

Installation script `install_vllm.sh` for vLLM for AMD:

```bash
# get latest cmake
conda install cmake -y

# get vllm for rocm
git clone --recursive https://github.com/vllm-project/vllm.git vllm
cd vllm

# Install PyTorch>2.5.0 (required for vLLM)
pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

# Install rest of the dependencies
pip install --upgrade numba scipy huggingface-hub[cli]
pip install "numpy<2"
pip install setuptools_scm
pip install -r requirements-rocm.txt

# Apply the patch to ROCM 6.1 (requires root permission)
wget -qN https://github.com/ROCm/vllm/raw/fa78403/rocm_patch/libamdhip64.so.6 -P /opt/rocm/lib
rm -f "$(python3 -c 'import torch; print(torch.__path__[0])')"/lib/libamdhip64.so*

# fix  `GLIBCXX_3.4.30' not found error
rm -rf ${CONDA_PREFIX}/lib/libstdc++.so.6
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX}/lib/libstdc++.so.6

# Build vLLM for MI210/MI250
export PYTORCH_ROCM_ARCH=$1
python3 setup.py develop

# Test the installation
python -c "import vllm; print('vllm test: ',vllm.__version__)"
echo "done"
```

Run with `bash install_vllm.sh gfx90a`.

###
