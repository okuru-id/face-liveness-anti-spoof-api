import cv2
import numpy as np

from app.core.config import settings
from app.schemas.liveness import QualityCheckResult


def check_quality(image: np.ndarray) -> QualityCheckResult:
    issues: list[str] = []

    h, w = image.shape[:2]
    if h < settings.min_resolution or w < settings.min_resolution:
        issues.append(f"Resolution too small: {w}x{h}, minimum {settings.min_resolution}x{settings.min_resolution}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < settings.blur_threshold:
        issues.append(f"Image too blurry: blur_score={blur_score:.2f}, threshold={settings.blur_threshold}")

    mean_brightness = np.mean(gray)
    if mean_brightness < settings.brightness_min or mean_brightness > settings.brightness_max:
        issues.append(
            f"Brightness out of range: mean={mean_brightness:.1f}, "
            f"range=[{settings.brightness_min}, {settings.brightness_max}]"
        )

    return QualityCheckResult(passed=len(issues) == 0, issues=issues)
