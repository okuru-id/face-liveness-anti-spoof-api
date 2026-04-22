import time

from fastapi import APIRouter, Depends, Response

from app.api.dependencies.auth import verify_api_key
from app.api.responses import create_liveness_response
from app.core.errors import InvalidImageFormatError, InternalError, RateLimitExceededError
from app.core.logging import get_logger
from app.core.request_id import generate_request_id
from app.schemas.common import SpoofType
from app.schemas.liveness import LivenessRequest
from app.services.anti_spoof import anti_spoof_service
from app.services.fft_analysis import get_fft_service
from app.services.face_detector import face_detector
from app.services.image_decoder import decode_base64_image
from app.services.image_validation import validate_image_size
from app.services.quality_check import check_quality
from app.services.rate_limiter import rate_limit, record_request
from app.services.verdict_engine import determine_verdict

router = APIRouter()
logger = get_logger(__name__)


@router.post("/liveness/check")
async def check_liveness(
    request_body: LivenessRequest,
    api_key: str = Depends(verify_api_key),
) -> Response:
    rl = rate_limit(api_key)
    if rl.remaining <= 0:
        raise RateLimitExceededError()

    record_request(api_key)
    start_time = time.time()
    request_id = generate_request_id()
    debug_enabled = bool(request_body.options and request_body.options.debug)

    if request_body.mode != "passive":
        raise InvalidImageFormatError("Only 'passive' mode is supported in MVP")

    validate_image_size(request_body.image)
    image = decode_base64_image(request_body.image)

    quality_result = check_quality(image)
    hard_quality_issues = [
        issue
        for issue in quality_result.issues
        if issue.startswith("Resolution too small")
    ]
    quality_passed = len(hard_quality_issues) == 0
    if not quality_passed:
        return create_liveness_response(
            request_id=request_id,
            verdict="POOR_QUALITY",
            confidence=0.0,
            spoof_type=None,
            face_detected=False,
            quality_passed=False,
            quality_issues=quality_result.issues,
            processing_time_ms=int((time.time() - start_time) * 1000),
            face_bbox=None,
            anti_spoof_debug=None,
            rate_limit_info=rl,
        )

    face_result = face_detector.detect(image)
    if not face_result.detected:
        return create_liveness_response(
            request_id=request_id,
            verdict="NO_FACE",
            confidence=0.0,
            spoof_type=None,
            face_detected=False,
            quality_passed=len(quality_result.issues) == 0,
            quality_issues=quality_result.issues,
            processing_time_ms=int((time.time() - start_time) * 1000),
            face_bbox=None,
            anti_spoof_debug=None,
            rate_limit_info=rl,
        )

    x, y, w, h = face_result.bbox
    face_crop = image[y : y + h, x : x + w]

    fft_score = get_fft_service().compute_fft_score(face_crop)

    try:
        anti_spoof_result = anti_spoof_service.predict(
            face_crop=face_crop,
            bbox=face_result.bbox,
            full_image=image,
        )
    except Exception as e:
        raise InternalError(f"Model inference failed: {str(e)}")

    verdict, confidence, spoof_type = determine_verdict(
        anti_spoof_result=anti_spoof_result,
        face_detected=True,
        quality_passed=quality_passed,
        spoof_type=None,
        fft_score=fft_score,
        quality_issues=quality_result.issues,
    )

    anti_spoof_debug = anti_spoof_result.debug if debug_enabled else None
    if debug_enabled and anti_spoof_debug is not None:
        if isinstance(anti_spoof_debug, dict):
            anti_spoof_debug["fft_score"] = fft_score
        else:
            anti_spoof_debug = {"fft_score": fft_score}

    return create_liveness_response(
        request_id=request_id,
        verdict=verdict.value,
        confidence=confidence,
        spoof_type=spoof_type.value if spoof_type else None,
        face_detected=True,
        quality_passed=len(quality_result.issues) == 0,
        quality_issues=quality_result.issues,
        processing_time_ms=int((time.time() - start_time) * 1000),
        face_bbox={"x": x, "y": y, "w": w, "h": h},
        anti_spoof_debug=anti_spoof_result.debug if debug_enabled else None,
        rate_limit_info=rl,
    )
