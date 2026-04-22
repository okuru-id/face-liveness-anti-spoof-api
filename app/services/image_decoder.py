import base64
import io

import cv2
import numpy as np

from app.core.errors import InvalidImageFormatError


def decode_base64_image(data: str) -> np.ndarray:
    try:
        raw = base64.b64decode(data)
    except Exception:
        raise InvalidImageFormatError("Invalid base64 string")

    nparr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise InvalidImageFormatError("Image must be JPEG or PNG, base64 encoded")

    return image
