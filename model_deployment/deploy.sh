#!/bin/bash
set -e

MODEL_WEIGHTS_NAME=$1

# Build and deploy serving docker
sudo docker build --tag triton_server .
sudo docker run --name $MODEL_WEIGHTS_NAME"_deployment" -d --gpus=all -v $(pwd)/model_repository:/home/model_repository --shm-size 1g --rm -p 8000:8000 -p 8001:8001  -p 8002:8002 -p 50050:50050 -p 50051:50051 triton_server
