from turbojpeg import TurboJPEG


class ImageDecodingError(Exception):
    """Exception raised for errors in the image decoding process."""

    pass


jpeg = TurboJPEG()


def decode_image(img):
    """
    Decodes a JPEG image using TurboJPEG.

    Args:
        img (bytes): The image data to decode.

    Returns:
        numpy.ndarray: The decoded image array.

    Raises:
        ImageDecodingError: If there is an error decoding the image.
    """
    try:
        frame = jpeg.decode(img)
        if len(frame.shape) != 3:
            raise ImageDecodingError("Decoded image does not have 3 dimensions")
        return frame
    except Exception as e:
        raise ImageDecodingError(f"Error decoding image: {e}")
