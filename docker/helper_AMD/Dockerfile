FROM rocm/dev-ubuntu-22.04:latest AS base
FROM rocm/dev-ubuntu-22.04:5.7 AS rocm5
FROM singularitybase.azurecr.io/installer/base/singularity-installer:20240723T094352909 AS installer
FROM singularitybase.azurecr.io/validations/base/singularity-tests:20240916T145736532 AS validator

FROM base

ENV SINGULARITY_IMAGE_ACCELERATOR=AMD
ENV SINGULARITY_IMAGE_FRAMEWORK=PYTORCH
ENV SINGULARITY_IMAGE_ACCELERATOR_SKU=
ENV GPU_ARCHS=gfx942
ENV PYTORCH_ROCM_ARCH=gfx942

USER root

RUN apt update && \
    apt install --no-install-recommends --no-install-suggests -yq \
        wget make curl git unzip cmake ninja-build libaio1 libaio-dev openssh-server \
        openmpi-bin openmpi-doc libopenmpi-dev libjpeg-dev python3-dev python3-pip \
        build-essential zlib1g-dev libncurses5-dev libgdbm-dev ccache tmux \
        ca-certificates sudo bzip2 libx11-6 rocthrust

ENV CONDA_DIR=/opt/conda
RUN wget --no-hsts -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && rm -f ~/miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN --mount=type=bind,source=install,target=/tmp/install \
    eval "$(conda shell.bash hook)" && \
    conda env create -f /tmp/install/helper_AMD/environment.yaml -n sfm && \
    conda activate sfm && \
    pip install /opt/rocm/share/amd_smi

# default PyT
RUN --mount=type=bind,source=install,target=/tmp/install \
    eval "$(conda shell.bash hook)" && conda activate sfm && \
    bash /tmp/install/helper_AMD/torch_geo_rocm.sh torch-2.3 && \
    bash /tmp/install/helper_AMD/fast_attn_rocm.sh $GPU_ARCHS && \
    bash /tmp/install/helper_AMD/kdb.sh $GPU_ARCHS

# nightly PyT -- performance boost for MI300x
# RUN --mount=type=bind,source=install,target=/tmp/install \
#     eval "$(conda shell.bash hook)" && conda activate sfm && \
#     pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2 && \
#     pip install deepspeed==0.15.4 && \
#     bash /tmp/install/helper_AMD/torch_geo_rocm.sh torch-2.6 && \
#     bash /tmp/install/helper_AMD/fast_attn_rocm.sh $GPU_ARCHS && \
#     bash /tmp/install/helper_AMD/kdb.sh $GPU_ARCHS

RUN rm -rf /var/lib/apt/lists/* && \
    apt purge --auto-remove -y && \
    apt clean -y && \
    conda clean --all --yes && \
    pip cache purge

# get rocm 5 additions
COPY --from=rocm5 /opt/rocm/lib/libamdhip64.so.5 /opt/rocm/lib/libamdhip64.so.5

# install Singularity add-ons
COPY --from=installer /installer /opt/microsoft/_singularity/installations/
RUN /opt/microsoft/_singularity/installations/singularity/installer.sh

# validate with the script provided by Singularity
COPY --from=validator /validations /opt/microsoft/_singularity/validations/
RUN /opt/microsoft/_singularity/validations/validator.sh
