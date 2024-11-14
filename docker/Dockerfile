FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base
FROM singularitybase.azurecr.io/installer/base/singularity-installer:20240723T094352909 AS installer
FROM singularitybase.azurecr.io/validations/base/singularity-tests:20240916T145736532 AS validator

FROM base

ENV TZ=Asia/Shanghai \
    TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"

RUN apt update && \
    apt install --no-install-recommends --no-install-suggests -yq \
    wget make curl git unzip cmake ninja-build libaio1 libaio-dev openssh-server \
    openmpi-bin openmpi-doc libopenmpi-dev libjpeg-dev python3-dev python3-pip \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev ccache tmux \
    ca-certificates sudo bzip2 libx11-6

ENV CONDA_DIR=/opt/conda
RUN wget --no-hsts -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p /opt/conda && rm -f ~/miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN --mount=type=bind,source=install,target=/tmp/install \
    eval "$(conda shell.bash hook)" && \
    conda env create -f /tmp/install/environment.yaml -n sfm && \
    conda activate sfm && \
    bash /tmp/install/install_third_party.sh

RUN rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean && \
    conda clean --all --yes && \
    pip cache purge

COPY --from=installer /installer /opt/microsoft/_singularity/installations/
RUN /opt/microsoft/_singularity/installations/singularity/installer.sh

COPY --from=validator /validations /opt/microsoft/_singularity/validations/
RUN /opt/microsoft/_singularity/validations/validator.sh
