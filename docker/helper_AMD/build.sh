#!/bin/bash
set -e

IMAGE=${IMAGE:-ai4s-sfm/amd}
docker build -t ${IMAGE} -f docker/helper_AMD/Dockerfile .

REGISTRY=${REGISTRY:-msrmoldyn}
TAG=${TAG:-$(date -u +%Y%m%d.%H%M%S)}
az acr login --name ${REGISTRY}
docker tag ${IMAGE} ${REGISTRY}.azurecr.io/${IMAGE}:${TAG}
docker push ${REGISTRY}.azurecr.io/${IMAGE}:${TAG}
