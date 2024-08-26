# FROM rocm/dev-ubuntu-22.04 AS base
FROM rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.4 AS base
FROM singularitybase.azurecr.io/base/job/pytorch/acpt-rocm5.7_ubuntu20.04_py3.10_pytorch_2.0.1:20231222T092720818 AS singularity

ENV SINGULARITY_IMAGE_ACCELERATOR=AMD
ENV SINGULARITY_IMAGE_FRAMEWORK=PYTORCH
ENV SINGULARITY_IMAGE_ACCELERATOR_SKU=

FROM base

RUN apt update && \
    apt install --no-install-recommends --no-install-suggests -yq \
        wget make git unzip libaio1 libaio-dev openssh-server \
        openmpi-bin openmpi-doc libopenmpi-dev

# rocm/dev-ubuntu-22.04 does not have conda installed
# ENV CONDA_DIR /opt/conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#         -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda

# ENV PATH=$CONDA_DIR/bin:$PATH

RUN --mount=type=bind,source=install,target=/tmp/install \
    eval "$(conda shell.bash hook)" && \
    conda env create -f /tmp/install/helper_AMD/environment.yaml -n sfm && \
    conda activate sfm

# fails to detect GPUs when built on non AMD GPU machine
# RUN --mount=type=bind,source=install,target=/tmp/install \
#     eval "$(conda shell.bash hook)" && \
#     conda activate sfm && \
#     bash /tmp/install/helper_AMD/torch_geo_rocm.sh && \
#     bash /tmp/install/helper_AMD/torch_geo_rocm.sh

RUN rm -rf /var/lib/apt/lists/* && \
    apt purge --auto-remove -y && \
    apt clean -y && \
    conda clean --all --yes && \
    pip cache purge

COPY --from=singularity /opt/microsoft/_singularity/installations/ /opt/microsoft/_singularity/installations/
RUN /opt/microsoft/_singularity/installations/singularity/installer.sh

COPY --from=singularity /opt/microsoft/_singularity/validations/ /opt/microsoft/_singularity/validations/
RUN /opt/microsoft/_singularity/validations/validator.sh
