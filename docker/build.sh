#!/bin/bash
set -e

IMAGE=${IMAGE:-ai4s-sfm}
docker build --progress=plain -t ${IMAGE} -f docker/Dockerfile .

REGISTRY=${REGISTRY:-msroctocr}
TAG=${TAG:-$(date -u +%Y%m%d.%H%M%S)}
az acr login --name ${REGISTRY}
docker tag ${IMAGE} ${REGISTRY}.azurecr.io/${IMAGE}:${TAG}
docker push ${REGISTRY}.azurecr.io/${IMAGE}:${TAG}
