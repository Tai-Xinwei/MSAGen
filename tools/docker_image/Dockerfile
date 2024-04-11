# select base image
ARG BASE_IMAGE
ARG VALIDATOR_IMAGE

FROM $BASE_IMAGE as base
FROM $VALIDATOR_IMAGE as validator

FROM base

# set this for timezone
ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq figlet && \
    figlet "SFM Framework"

COPY conda_install.sh /tmp/conda_install.sh
RUN chmod +x /tmp/conda_install.sh && /tmp/conda_install.sh

# try to reduce the image size
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean && \
    conda clean --all --yes

# get the validation scripts and run validations
COPY --from=validator /validations /opt/microsoft/_singularity/validations/
RUN /opt/microsoft/_singularity/validations/validator.sh
