import cv2
import numpy as np

from app.core.config import settings
from app.core.errors import ImageTooLargeError


def validate_image_size(data: str) -> None:
    raw_bytes = len(data.encode()) * 3 // 4
    if raw_bytes > settings.max_image_size_bytes:
        raise ImageTooLargeError(
            f"Image size {raw_bytes} bytes exceeds {settings.max_image_size_bytes} bytes limit"
        )
