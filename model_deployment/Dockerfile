FROM nvcr.io/nvidia/tritonserver:22.10-py3
USER root

# Copy needed files
COPY ./requirements.txt /home/

# Install dependencies
RUN apt-get update
RUN yes | apt-get install libgl1
RUN pip3 install --upgrade pip
RUN apt-get install -y libturbojpeg
RUN pip3 install torch

RUN pip install -r /home/requirements.txt

EXPOSE 8080 8081 50050 50051
CMD [ "tritonserver", "--model-repository=/home/model_repository", "--model-control-mode=explicit", "--load-model=*" , "--strict-model-config=false","--grpc-port=50050"]
