FROM nvcr.io/nvidia/pytorch:22.10-py3
USER root

# Get external arguments
ARG MODEL_WEIGHTS_NAME
ARG MODEL_TYPE
ARG DIM_Y=384
ARG DIM_X=640
ARG DEVICE

# Define external arguments and env variables
ENV MODEL_WEIGHTS_NAME=$MODEL_WEIGHTS_NAME
ENV MODEL_TYPE=$MODEL_TYPE
ENV DIM_Y=$DIM_Y
ENV DIM_X=$DIM_X
ENV DEVICE=$DEVICE

RUN apt-get update
RUN apt-get install -y libgl1

# Copy needed folders into docker
COPY ./onnx_compilation /home/onnx_compilation/
COPY ./model_weights /home/onnx_compilation/model_weights/

# Create a directory for shared files and set permissions
RUN mkdir -p /home/config_files && \
    chmod -R 777 /home/config_files

# Copy config files into the shared directory
COPY ./config_files /home/config_files/

WORKDIR /home/

RUN cd /home/onnx_compilation/$MODEL_TYPE && \
    echo $MODEL_WEIGHTS_NAME && \
    pip install -r requirements.txt onnx onnx-simplifier onnxruntime-gpu && \
    python export.py --weights /home/onnx_compilation/model_weights/$MODEL_WEIGHTS_NAME.pt --include onnx --opset 12 --imgsz $DIM_Y $DIM_X && \
    cd /home/ && \
    python config_files/create_config.py /home/onnx_compilation/model_weights/$MODEL_WEIGHTS_NAME.onnx $MODEL_TYPE $DEVICE

# For all models: Compile to tensorRT
CMD trtexec --onnx=/home/onnx_compilation/model_weights/$MODEL_WEIGHTS_NAME.onnx --saveEngine=/home/model_weights/model.plan --fp16
