import argparse
import ast
import os
from typing import Any, Dict, List, Tuple

import cv2
import imutils
import numpy as np
from triton_handler import TritonCallHandler
from tritonclient import grpc as tritongrpcclient


def is_image(file_path: str) -> bool:
    try:
        # Try to read the file as an image
        img = cv2.imread(file_path)
        # If successful, it's an image
        return img.any()
    except Exception:
        # If any error occurs, it's not an image
        return False


def is_video(file_path: str) -> bool:
    try:
        # Try to capture the video
        cap = cv2.VideoCapture(file_path)
        # Check if the capture was successful
        return cap.isOpened()
    except Exception:
        # If any error occurs, it's not a video
        return False


def list_files_in_directory(directory_path: str) -> List[str]:
    try:
        # Get a list of all files in the directory
        files = os.listdir(directory_path)
        # Filter out directories, if any
        files = [f for f in files if os.path.isfile(os.path.join(directory_path, f))]
        return files
    except Exception:
        return []


def display_detections(
    model_input: np.ndarray, response: Dict[str, List[Dict[str, Any]]]
) -> None:
    """
    Display object detections on an image.

    Parameters:
        model_input (numpy.ndarray): Input image data.
        response (dict): Detections information.
    """
    h, w, _ = model_input.shape
    frame = model_input.copy()
    color = (255, 0, 255)

    for det in response["detections"]:
        bb = det["location"]
        bound_b: Tuple[Tuple[int, int], Tuple[int, int]] = (
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
        )
        frame = cv2.rectangle(frame, bound_b[0], bound_b[1], color, 2)

        if "label" in det.keys() and "confidence" in det.keys():
            label = det["label"]
            confidence = det["confidence"]
            label_text = f"{label}: {confidence:.2f}"
            text_position: Tuple[int, int] = (int(bb[0]), int(bb[1]) - 10)

            # Get the width and height of the text box
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2
            )
            # Draw the background rectangle for the text
            frame = cv2.rectangle(
                frame,
                (text_position[0], text_position[1] - text_height - baseline),
                (text_position[0] + text_width, text_position[1] + baseline),
                color,
                -1,
            )
            # Put the text on the frame
            frame = cv2.putText(
                frame,
                label_text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

    cv2.imwrite("image.jpg", frame)
    # Display the image
    cv2.imshow("Image", imutils.resize(frame, height=800))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def infer_triton(
    model_input: str,
    host: str,
    port: int,
    model_name: str,
    conf_thresh: bool,
    visualization: bool = False,
) -> Dict:
    triton_handler = TritonCallHandler()

    triton_client = tritongrpcclient.InferenceServerClient(
        url=f"{host}:{port}",
        verbose=False,
        ssl=False,
        #     root_certificates=cert + "/ca.pem",
        #     private_key=cert + "/server-key.pem",
        #     certificate_chain=cert + "/server.pem",
    )

    triton_handler.add_image(model_input)

    if conf_thresh:
        triton_handler.add_threshold(conf_thresh)

    results = triton_client.infer(
        model_name=model_name,
        inputs=triton_handler.inputs,
        outputs=triton_handler.outputs,
    )

    response = ast.literal_eval(
        results.as_numpy(triton_handler.output_name).tolist().decode("utf-8")
    )

    statistics = triton_client.get_inference_statistics(model_name=model_name)
    response_time = statistics.model_stats[0].inference_stats.success.ns / (
        statistics.model_stats[0].inference_stats.success.count
    )

    if visualization:
        display_detections(model_input, response)

    response["time"] = response_time / 1000000000
    print(response)

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolov9_model",
        help="Model name as it appears in the triton server.",
    )
    parser.add_argument(
        "--model-input", type=str, help="Input to model (image, video or folder path)."
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host type (local or ec2 url)"
    )
    parser.add_argument(
        "--visualization", type=bool, default=False, help="Include bbox visuals."
    )
    parser.add_argument("--port", type=int, default=50050, help="Model port.")
    parser.add_argument("--conf-thresh", type=float, default=None, help="Model port.")

    opt = parser.parse_args()

    # Check if input is a directory, an image or a video
    if os.path.isdir(opt.model_input):
        files = list_files_in_directory(opt.model_input)
        files.sort()
        for file in files:
            img = cv2.imread(f"{opt.model_input}/{file}")
            infer_triton(
                img,
                opt.host,
                opt.port,
                opt.model_name,
                opt.conf_thresh,
                opt.visualization,
            )
    else:
        if is_image(opt.model_input):
            frame = cv2.imread(opt.model_input)
            infer_triton(
                frame,
                opt.host,
                opt.port,
                opt.model_name,
                opt.conf_thresh,
                opt.visualization,
            )
        elif is_video(opt.model_input):
            cap = cv2.VideoCapture(opt.model_input)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    infer_triton(
                        frame,
                        opt.host,
                        opt.port,
                        opt.model_name,
                        opt.conf_thresh,
                        opt.visualization,
                    )
                else:
                    cap.release()
