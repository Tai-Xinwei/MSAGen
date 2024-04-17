#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <image-name> <tag>"
    echo "Note: this script is supposed to run from the root of this repo"
    exit 1
fi
IMAGE_NAME=$1
TAG=$2

RED='\033[0;31m'
GREEN='\033[1;32m'
NC='\033[0m'

registry="singularitybase"
# go to portal.azure.com -> container registries -> singularitybase to select a base image
base_image_repo="base/job/pytorch/acpt-2.2.1-py3.10-cuda12.1"
validator_image_repo="validations/base/singularity-tests"

# print logs if failed
az acr login -n $registry

if [ $? -ne 0 ]; then
    echo -e "Failed to login to ACR, exiting..."
    echo -e "Please use ${RED} az login --use-device-code${NC} to login to Azure first."
    echo -e "If this error persists, please check if you have the correct permissions to access the ACR."
    exit 1
else
    echo "Logged in to ACR"
fi
# TODO: here is a warning, maybe someone will modify this.
# WARNING: This command has been deprecated and will be removed in a future release. Use 'acr manifest list-metadata' instead.
base_image_tag=$(az acr repository show-manifests \
    --name $registry \
    --repository $base_image_repo \
    --orderby time_desc \
    --query '[].{Tag:tags[0]}' \
    --output tsv --top 1)

base_image="$registry.azurecr.io/$base_image_repo:$base_image_tag"
echo "base_image: $base_image"

validator_image_tag=$(az acr repository show-manifests \
    --name $registry \
    --repository $validator_image_repo \
    --orderby time_desc \
    --query '[].{Tag:tags[0]}' \
    --output tsv --top 1)

validator_image="$registry.azurecr.io/$validator_image_repo:$validator_image_tag"
echo "validator_image: $validator_image"

echo "Building image..."
docker build . -f docker/Dockerfile \
    --build-arg BASE_IMAGE=$base_image \
    --build-arg VALIDATOR_IMAGE=$validator_image \
    --progress=plain \
    -t "$IMAGE_NAME:$TAG"

if [ $? -eq 0 ]; then
    echo "Docker image $IMAGE_NAME:$TAG built successfully."
else
    echo "Failed to build Docker image $IMAGE_NAME:$TAG."
    exit 1
fi

# push the image to ACR, change to some other registry as needed
# docker tag "$IMAGE_NAME:$TAG" "$registry.azurecr.io/$IMAGE_NAME:$TAG"
# docker push "$registry.azurecr.io/$IMAGE_NAME:$TAG"
# Note: before pushing, you may need to az acr login -n $registry
