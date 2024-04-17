# Run build.sh from the root directory of this repo to build the image
ARG BASE_IMAGE
ARG VALIDATOR_IMAGE

FROM $BASE_IMAGE as base
FROM $VALIDATOR_IMAGE as validator
FROM base

# set this for timezone
ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME="/usr/local/cuda" \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0" \
    RED='\033[0;31m' \
    GREEN='\033[1;32m' \
    NC='\033[0m'

COPY ./install/py39-torch2.2.2-cuda12.1.yaml /tmp/sfm-cuda.yaml
COPY ./docker/conda_install.sh /tmp/conda_install.sh

RUN apt-get update && apt-get install -yq figlet && \
    figlet "SFM Framework"

RUN chmod +x /tmp/conda_install.sh && \
    bash /tmp/conda_install.sh

# try to reduce the image size
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean && \
    conda clean --all --yes

# get the validation scripts and run validations
COPY --from=validator /validations /opt/microsoft/_singularity/validations/
RUN /opt/microsoft/_singularity/validations/validator.sh
