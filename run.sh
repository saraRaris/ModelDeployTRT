#!/bin/bash
set -e

MODEL_WEIGHTS_NAME=$1

if [[ $MODEL_WEIGHTS_NAME == "yolov9" ]]; then
    # Download model weights
    wget -O model_compilation/model_weights/yolov9.pt https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s-converted.pt
else
    echo "Selected model in not suppored!"
    exit
fi

# Compile model
cd model_compilation
bash compile.sh $MODEL_WEIGHTS_NAME

# Copy the comiled model weights
cd ..
cp model_compilation/model_weights/model.plan model_deployment/model_repository/$MODEL_WEIGHTS_NAME/0
echo "Compiled model moved to model repository: $MODEL_WEIGHTS_NAME"

# Deploy the model
cd model_deployment
bash deploy.sh $MODEL_WEIGHTS_NAME