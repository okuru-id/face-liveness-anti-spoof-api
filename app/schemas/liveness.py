from typing import Optional
from pydantic import BaseModel, Field


class QualityCheckResult(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)


class LivenessOptions(BaseModel):
    return_face_crop: bool = False
    min_face_size: int = 100
    debug: bool = False


class LivenessRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image string (JPEG or PNG)")
    mode: str = Field(default="passive", description="Liveness mode, only 'passive' supported in MVP")
    options: Optional[LivenessOptions] = None


class ChallengeResult(BaseModel):
    type: str
    passed: bool
    confidence: float


class FaceBBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class LivenessResponse(BaseModel):
    request_id: str
    verdict: str
    confidence: float
    spoof_type: Optional[str] = None
    face_detected: bool
    quality_check: QualityCheckResult
    processing_time_ms: int
    timestamp: str
    face_bbox: Optional[FaceBBox] = None
    anti_spoof_debug: Optional[dict] = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
