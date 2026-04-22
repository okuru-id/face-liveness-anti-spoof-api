from fastapi import Response
from fastapi.responses import JSONResponse
from datetime import datetime, timezone

from app.core.errors import AppError
from app.core.request_id import generate_request_id
from app.core.logging import get_logger
from app.services.rate_limiter import RateLimitInfo

logger = get_logger(__name__)


def _add_rate_limit_headers(response: Response, info: RateLimitInfo | None) -> Response:
    if info:
        response.headers["X-RateLimit-Limit"] = str(info.limit)
        response.headers["X-RateLimit-Remaining"] = str(max(info.remaining - 1, 0))
        response.headers["X-RateLimit-Reset"] = str(info.reset)
    return response


def create_response(
    request_id: str,
    data: dict,
    status_code: int = 200,
    rate_limit_info: RateLimitInfo | None = None,
) -> Response:
    response = JSONResponse(content=data, status_code=status_code)
    response.headers["X-Request-ID"] = request_id
    return _add_rate_limit_headers(response, rate_limit_info)


def create_error_response(error: AppError, request_id: str | None = None) -> Response:
    rid = request_id or generate_request_id()
    error_data = {
        "error": {
            "code": error.code.value,
            "message": error.message,
            "request_id": rid,
        }
    }
    logger.error(f"Error [{rid}]: {error.code.value} - {error.message}")
    return JSONResponse(content=error_data, status_code=error.status_code)


def create_liveness_response(
    request_id: str,
    verdict: str,
    confidence: float,
    spoof_type: str | None,
    face_detected: bool,
    quality_passed: bool,
    quality_issues: list[str],
    processing_time_ms: int,
    face_bbox: dict | None = None,
    anti_spoof_debug: dict | None = None,
    rate_limit_info: RateLimitInfo | None = None,
) -> Response:
    timestamp = datetime.now(timezone.utc).isoformat()
    data = {
        "request_id": request_id,
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "spoof_type": spoof_type,
        "face_detected": face_detected,
        "quality_check": {"passed": quality_passed, "issues": quality_issues},
        "processing_time_ms": processing_time_ms,
        "timestamp": timestamp,
        "face_bbox": face_bbox,
        "anti_spoof_debug": anti_spoof_debug,
    }
    logger.info(
        f"Liveness [{request_id}]: verdict={verdict} confidence={confidence:.2f} "
        f"spoof_type={spoof_type} face={face_detected} time={processing_time_ms}ms"
    )
    return create_response(request_id, data, 200, rate_limit_info)
