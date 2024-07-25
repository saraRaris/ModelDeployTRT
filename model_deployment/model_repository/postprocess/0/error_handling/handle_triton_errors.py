class ImageDecodingError(Exception):
    """Exception raised for errors in the image decoding process."""

    pass


class InvalidResponseFormatError(Exception):
    """Exception raised for errors in the response format."""

    pass


def check_output_format(response):
    """
    Validates the format of the output response from the Triton inference server.

    Args:
        response (dict): The response dictionary to validate.

    Raises:
        InvalidResponseFormatError: If the response format is invalid.
    """
    # Check if the output is a dictionary
    if not isinstance(response, dict):
        raise InvalidResponseFormatError("Output must be a dictionary")

    # Check if the "detections" key exists and is a list
    if "detections" not in response or not isinstance(response["detections"], list):
        raise InvalidResponseFormatError(
            'Output must contain a "detections" key with a list of detections'
        )

    # Check if each detection contains the required keys
    required_keys = ["label", "location", "confidence"]
    for detection in response["detections"]:
        for key in required_keys:
            if key not in detection:
                raise InvalidResponseFormatError(
                    f"Each detection must contain the key: {key}"
                )
