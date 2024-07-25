#!/bin/bash
set -e

MODEL_WEIGHTS_NAME=$1

# Build compile docker
sudo docker build --tag triton_compile --build-arg MODEL_WEIGHTS_NAME=$1 --build-arg MODEL_TYPE=yolov9 --build-arg DEVICE=cuda .

# Get onnx weight and config file out
sudo docker create --name my_container triton_compile
sudo docker cp my_container:/home/config_files/config.pbtxt config_files/
sudo docker cp my_container:/home/onnx_compilation/model_weights/$MODEL_WEIGHTS_NAME.onnx model_weights/model.onnx
sudo docker container rm my_container

# Run compilation
sudo docker run --gpus=all -i --rm  -v$(pwd):/workspace/ -v $(pwd)/model_weights:/home/model_weights -v $(pwd)/config_files:/home/config_files --name triton_compile triton_compile
