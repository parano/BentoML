#!/usr/bin/env bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT"

echo "Building BentoML proto generator docker image.."
docker build -t bentoml-proto-generator - < $GIT_ROOT/protos/Dockerfile

echo "Starting BentoML proto generator docker container.."
docker run --rm -v $GIT_ROOT:/home/bento bentoml-proto-generator \
    bash -c "cd /home/bento/ && ./protos/generate.sh"

echo "DONE"

