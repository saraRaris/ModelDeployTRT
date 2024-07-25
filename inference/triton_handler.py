import base64

import cv2
import numpy as np
from tritonclient import grpc as tritongrpcclient


class TritonCallHandler:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.input_name_0: str = "INPUT_IMAGESTR"
        self.input_name_1: str = "CONFIDENCE_TRESH"
        self.output_name: str = "FINAL_OUTPUT"

    def add_image(self, image_str: str):
        image_str = base64.b64encode(cv2.imencode(".jpg", image_str)[1]).decode()
        jpg_original = base64.b64decode(image_str)

        img_data = np.array([jpg_original], dtype=bytes)
        img_data = np.stack([img_data], axis=0)

        self.inputs.append(
            tritongrpcclient.InferInput(self.input_name_0, img_data.shape, "BYTES")
        )
        self.inputs[0].set_data_from_numpy(img_data)

        self.outputs.append(tritongrpcclient.InferRequestedOutput(self.output_name))

    def add_threshold(self, conf_tresh: float):
        conf_tresh = np.array([conf_tresh])
        conf_tresh = np.stack([conf_tresh], axis=0)
        self.inputs.append(
            tritongrpcclient.InferInput(self.input_name_1, conf_tresh.shape, "FP64")
        )
        self.inputs[1].set_data_from_numpy(conf_tresh)
