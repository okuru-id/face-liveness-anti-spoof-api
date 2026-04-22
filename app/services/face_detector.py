import cv2
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.core.errors import ModelUnavailableError


@dataclass
class FaceDetectionResult:
    detected: bool
    bbox: tuple[int, int, int, int] | None
    face_size: int | None


class FaceDetector:
    def __init__(self):
        self._detector = None

    def _load(self):
        if self._detector is not None:
            return
        deploy = settings.retinaface_deploy_path
        caffemodel = settings.retinaface_caffemodel_path
        if not Path(deploy).exists() or not Path(caffemodel).exists():
            raise ModelUnavailableError('RetinaFace detection model files not found')
        self._detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)

    def detect(self, image: np.ndarray) -> FaceDetectionResult:
        self._load()
        height, width = image.shape[:2]
        aspect_ratio = width / height
        resized = image
        resize_w, resize_h = width, height
        if width * height >= 192 * 192:
            resize_w = int(192 * math.sqrt(aspect_ratio))
            resize_h = int(192 / math.sqrt(aspect_ratio))
            resized = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(resized, 1, mean=(104, 117, 123))
        self._detector.setInput(blob, 'data')
        out = self._detector.forward('detection_out').squeeze()

        if out.size == 0:
            return FaceDetectionResult(detected=False, bbox=None, face_size=None)
        if out.ndim == 1:
            out = np.expand_dims(out, axis=0)

        max_conf_index = int(np.argmax(out[:, 2]))
        confidence = float(out[max_conf_index, 2])
        if confidence <= 0:
            return FaceDetectionResult(detected=False, bbox=None, face_size=None)

        left = out[max_conf_index, 3] * width
        top = out[max_conf_index, 4] * height
        right = out[max_conf_index, 5] * width
        bottom = out[max_conf_index, 6] * height

        x = max(0, int(left))
        y = max(0, int(top))
        w = max(0, int(right - left + 1))
        h = max(0, int(bottom - top + 1))

        if w < settings.min_face_size or h < settings.min_face_size:
            return FaceDetectionResult(detected=False, bbox=None, face_size=None)

        return FaceDetectionResult(
            detected=True,
            bbox=(x, y, w, h),
            face_size=int(max(w, h)),
        )


face_detector = FaceDetector()
