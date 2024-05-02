FROM singularitybase.azurecr.io/base/job/pytorch/acpt-2.2.1-py3.10-cuda12.1:20240312T225111416 as base
FROM singularitybase.azurecr.io/validations/base/singularity-tests:20230602T145041989 as validator

FROM base

ENV TZ=Asia/Shanghai \
    TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"

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

COPY --from=validator /validations /opt/microsoft/_singularity/validations/
RUN /opt/microsoft/_singularity/validations/validator.sh
