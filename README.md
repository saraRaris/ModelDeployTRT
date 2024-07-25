# Model Compilation and Deployment with TensorRT and Triton

This repository provides a streamlined process for compiling models to TensorRT and deploying them for inference using NVIDIA Triton Inference Server.

## Introduction

This repository enables the compilation of machine learning models to TensorRT format and subsequent deployment for inference using NVIDIA Triton. The current implementation supports YOLOv9 models, but can be extended for other models.

## Repository Structure

- **run.sh**: Main script to handle the entire process of downloading model weights, compiling the model, and deploying it.
- **model_compilation/**: Contains scripts and files necessary for compiling models. The main files are:
  - **compile.sh**: Script that handles the compilation for a model by initializing a Docker, compiling to ONNX, and then compiling to TensorRT.
  - **onnx_compilation/**: Folder that contains the model code necessary to compile listed models to ONNX (for now only YOLOv9 supported).
  - **model_weights/**: Folder where all converted weights are placed.
  - **config_files/**: Folder that contains the Python scripts to create the model config.pbtxt file (currently unused).
- **model_deployment/**: Contains files and scripts for deploying compiled models. The main files are:
  - **deploy.sh**: Script that handles the deployment of the model repository.
  - **model_repository/**: Folder that contains all models to be deployed. For more info on the model_repository, check out the [Triton Nvidia documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)
    - **yolov9_model/**: Main model - ensemble that defines the whole pipeline.
    - **preprocess/**: Auxiliary model - Python model with preprocessing functionality.
    - **yolov9/**: Deep learning model - YOLOv9 compiled to TensorRT.
    - **postprocess/**: Auxiliary model - Python model with postprocessing functionality.
- **inference/**: Contains files necessary for inferring a deployed model.
- **sample_images/**: Contains sample images that can be used to inference.

## Folder Structure

```markdown
triton_simple_deployment/
│
├── model_compilation/
│   ├── compile.sh
│   └── model_weights/
│       └── yolov9.pt
│
├── model_deployment/
│   ├── deploy.sh
│   └── model_repository/
| 		├── preprocess/
|   	├── yolov9_model/
| 		├── postpocess/
│       └── yolov9/
│
├── inference/
│   ├── infer.py
│   └── triton_handler.py
│
├── sample_images/
│
├── run.sh
└── README.md

```

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Poetry
- Installed Docker
- Access to NVIDIA Docker (in the future this repo will also support cpu only)
- wget installed on your system

## Getting Started

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/ModelDeployTRT.git
cd ModelDeployTRT
```

## Usage

#### Compilation & Deployment

To compile and deploy a YOLOv9 model, run the `run.sh` script with the model name as an argument:

```bash
./run.sh yolov9
```

This script performs the following steps:

1. Downloads the YOLOv9 model weights.
2. Compiles the model to TensorRT format.
3. Copies the compiled model to the model repository.
4. Deploys the model using Triton Inference Server.

#### Inference

To test inferring a deployed model, first activate poetry:

```bash
poetry shell
```

Then, you can use the images in *sample_images* folder or use any custom image and run the following script:

```bash
python infer_model.py --port 50050 --host <host_ip> --model-name yolov9_model --model-input <image_path> --visualization True --conf-thresh <threshold>
```

Where:

- **port**: Docker port used for inference, 50050 is defined by default. Do not change this unless any modifications are made to the code.
- **host**: IP of the machine serving the models.
- **model_name**: Name of the main model, in this case `yolov9_model` by default.
- **model_input**: Path to the image, folder of images or video path to infer.
- **visualization** (optional): Boolean to display the results after inferring. When not defined, set to False.
- **conf-thresh** (optional): Confidence threshold for the detections.

## Sample results

Here are examples of object detection results:

<p float="left">
  <img src="results/image3.jpg" width="47%" /> 
  <img src="results/bus.jpg" width="42%" />
</p>

<p float="left">
  <img src="results/horses.jpg" width="42%" />
  <img src="results/zidane.jpg" width="50%" /> 
</p>

<p float="left">
  <img src="results/image1.jpg" width="42%" />
  <img src="results/image2.jpg" width="47%" /> 
</p>

## To do

- Add support for CPU only devices.
- Include image input dimensions as a parameters for compilation and deployment.

## Acknowledgements

- https://github.com/WongKinYiu/yolov9
- https://github.com/triton-inference-server/server