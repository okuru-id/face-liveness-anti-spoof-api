import time
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.dependencies.auth import verify_api_key
from app.api.responses import create_response
from app.core.config import settings
from app.core.request_id import generate_request_id
from app.core.session_store import FrameMeta, session_store
from app.schemas.liveness import FaceBBox
from app.services.anti_spoof import AntiSpoofResult, SpoofLabel, anti_spoof_service
from app.services.face_detector import face_detector
from app.services.fft_analysis import get_fft_service
from app.services.fusion import fuse
from app.services.rate_limiter import rate_limit, record_request
from app.services.rppg_physnet import rppg_service
from app.services.signal_analysis import SignalAnalyzer
from app.services.verdict_engine import determine_verdict

router = APIRouter()


def _aggregate_mini_fas_results(frames: list[FrameMeta]):
    valid_results = [frame.mini_fas_result for frame in frames if isinstance(frame.mini_fas_result.debug, dict)]
    if not valid_results:
        return None

    avg_probs_list = []
    for result in valid_results:
        probs = result.debug.get("avg_probs")
        if isinstance(probs, list) and len(probs) == 3:
            avg_probs_list.append([float(v) for v in probs])

    if not avg_probs_list:
        return valid_results[-1]

    mean_probs = np.mean(np.asarray(avg_probs_list, dtype=np.float32), axis=0)
    pred_label = int(np.argmax(mean_probs))
    confidence = float(mean_probs[pred_label])

    return AntiSpoofResult(
        label=SpoofLabel.LIVE if pred_label == 1 else SpoofLabel.SPOOF,
        confidence=confidence,
        debug={
            "avg_probs": [round(float(v), 6) for v in mean_probs.tolist()],
            "pred_label": pred_label,
            "pred_label_name": "LIVE" if pred_label == 1 else "SPOOF",
            "aggregation": "mean_over_stream_frames",
            "frame_count": len(avg_probs_list),
        },
    )


@router.post("/liveness/stream/init")
async def init_stream(api_key: str = Depends(verify_api_key)):
    del api_key
    request_id = generate_request_id()
    start_time = time.time()

    try:
        session = session_store.create(
            mode="realtime",
            window_ms=settings.stream_window_ms,
            min_frames=settings.stream_min_frames,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return create_response(
        request_id=request_id,
        data={
            "request_id": request_id,
            "session_id": session.session_id,
            "window_ms": session.window_ms,
            "min_frames": session.min_frames,
            "frame_interval_ms": int(1000 / settings.stream_frame_rate),
            "expires_at": datetime.fromtimestamp(
                session.created_at + settings.stream_session_ttl_ms / 1000,
                tz=timezone.utc,
            ).isoformat(),
            "processing_time_ms": int((time.time() - start_time) * 1000),
        },
    )


@router.post("/liveness/stream/frame")
async def upload_stream_frame(
    session_id: str = Form(...),
    frame: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):
    request_id = generate_request_id()
    start_time = time.time()

    rl = rate_limit(api_key)
    if rl.remaining <= 0:
        return create_response(
            request_id,
            {
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Rate limit exceeded",
                    "request_id": request_id,
                }
            },
            status_code=429,
            rate_limit_info=rl,
        )
    record_request(api_key)

    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    payload = await frame.read()
    image_array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="Invalid frame image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_result = face_detector.detect(image)
    if not face_result.detected or face_result.bbox is None:
        return create_response(
            request_id,
            {
                "request_id": request_id,
                "session_id": session_id,
                "status": "skip",
                "reason": "no_face",
                "processing_time_ms": int((time.time() - start_time) * 1000),
            },
        )

    x, y, w, h = face_result.bbox

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))

    if settings.stream_quality_gate_enabled:
        if blur_score < settings.blur_threshold:
            return create_response(
                request_id,
                {
                    "request_id": request_id,
                    "session_id": session_id,
                    "status": "skip",
                    "reason": "blur",
                    "face_bbox": {"x": x, "y": y, "w": w, "h": h},
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                },
            )

        if mean_brightness < settings.brightness_min or mean_brightness > settings.brightness_max:
            return create_response(
                request_id,
                {
                    "request_id": request_id,
                    "session_id": session_id,
                    "status": "skip",
                    "reason": "brightness",
                    "face_bbox": {"x": x, "y": y, "w": w, "h": h},
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                },
            )

    face_crop = image[y : y + h, x : x + w]
    fft_score = get_fft_service().compute_fft_score(face_crop)
    anti_spoof_result = anti_spoof_service.predict(
        face_crop=face_crop,
        bbox=face_result.bbox,
        full_image=image,
    )

    if (
        anti_spoof_result.label.value == "SPOOF"
        and anti_spoof_result.confidence >= settings.stream_fast_reject_spoof_confidence
    ):
        return create_response(
            request_id,
            {
                "request_id": request_id,
                "session_id": session_id,
                "status": "fast_reject",
                "reason": "spoof_confident",
                "confidence": round(anti_spoof_result.confidence, 4),
                "face_bbox": {"x": x, "y": y, "w": w, "h": h},
                "processing_time_ms": int((time.time() - start_time) * 1000),
            },
        )

    updated_session = session_store.add_frame(
        session_id,
        FrameMeta(
            face_crop=face_crop,
            bbox=FaceBBox(x=x, y=y, w=w, h=h),
            mini_fas_result=anti_spoof_result,
            fft_score=fft_score,
            blur_score=blur_score,
        ),
    )
    if updated_session is None:
        raise HTTPException(status_code=410, detail="Session is no longer collecting")

    return create_response(
        request_id,
        {
            "request_id": request_id,
            "session_id": session_id,
            "status": updated_session.state,
            "frame_count": len(updated_session.frames),
            "face_bbox": {"x": x, "y": y, "w": w, "h": h},
            "processing_time_ms": int((time.time() - start_time) * 1000),
        },
    )


@router.get("/liveness/stream/result")
async def get_stream_result(session_id: str, api_key: str = Depends(verify_api_key)):
    request_id = generate_request_id()
    start_time = time.time()

    rl = rate_limit(api_key)
    if rl.remaining <= 0:
        return create_response(
            request_id,
            {
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Rate limit exceeded",
                    "request_id": request_id,
                }
            },
            status_code=429,
            rate_limit_info=rl,
        )
    record_request(api_key)

    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    if session.cached_result is not None:
        return create_response(request_id, session.cached_result)

    if not session.has_enough_frames:
        return create_response(
            request_id,
            {
                "request_id": request_id,
                "session_id": session_id,
                "status": "waiting",
                "frame_count": len(session.frames),
                "min_frames": session.min_frames,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            },
        )

    resized_frames: list[np.ndarray] = []
    for frame_meta in session.frames:
        crop = frame_meta.face_crop
        if crop is None or crop.size == 0:
            continue
        resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        resized_frames.append(resized)

    if len(resized_frames) < session.min_frames:
        return create_response(
            request_id,
            {
                "request_id": request_id,
                "session_id": session_id,
                "status": "waiting",
                "frame_count": len(resized_frames),
                "min_frames": session.min_frames,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            },
        )

    frames = np.stack(resized_frames, axis=0)
    rppg_result = rppg_service.infer(frames)
    signal_metrics = SignalAnalyzer(fps=settings.stream_frame_rate).analyze(
        rppg_result.signal,
        min_frames=session.min_frames,
    )

    latest_frame = session.frames[-1]
    aggregate_mini_fas_result = _aggregate_mini_fas_results(session.frames) or latest_frame.mini_fas_result
    aggregate_fft_score = float(np.median([frame.fft_score for frame in session.frames]))
    aggregate_blur_score = float(np.median([frame.blur_score for frame in session.frames]))
    fusion_result = fuse(aggregate_mini_fas_result, rppg_result, signal_metrics)

    passive_quality_issues: list[str] = []
    if aggregate_blur_score < settings.blur_threshold:
        passive_quality_issues.append(
            f"Image too blurry: blur_score={aggregate_blur_score:.2f}, threshold={settings.blur_threshold}"
        )

    passive_verdict, passive_confidence, passive_spoof_type = determine_verdict(
        anti_spoof_result=aggregate_mini_fas_result,
        face_detected=True,
        quality_passed=True,
        spoof_type=None,
        fft_score=aggregate_fft_score,
        quality_issues=passive_quality_issues,
    )

    final_verdict = fusion_result.verdict
    final_confidence = fusion_result.confidence
    final_spoof_type = None
    override_reason = None

    if passive_verdict == fusion_result.verdict:
        final_spoof_type = passive_spoof_type
    elif passive_verdict.value == "SPOOF":
        final_verdict = passive_verdict
        final_confidence = passive_confidence
        final_spoof_type = passive_spoof_type
        override_reason = "passive_spoof_override"
    elif passive_verdict.value == "UNCERTAIN" and fusion_result.verdict.value == "LIVE":
        final_verdict = passive_verdict
        final_confidence = min(fusion_result.confidence, passive_confidence)
        override_reason = "passive_uncertain_override"

    response_data = {
        "request_id": request_id,
        "verdict": final_verdict.value,
        "confidence": round(final_confidence, 4),
        "spoof_type": final_spoof_type.value if final_spoof_type else None,
        "face_detected": True,
        "quality_check": {
            "passed": signal_metrics.signal_valid,
            "issues": signal_metrics.flags,
        },
        "processing_time_ms": int((time.time() - start_time) * 1000),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "face_bbox": latest_frame.bbox.model_dump(),
        "anti_spoof_debug": {
            **fusion_result.fusion_debug,
            "signal_confidence": round(fusion_result.signal_confidence, 4),
            "mini_fas_confidence": round(fusion_result.mini_fas_confidence, 4),
            "fft_score": round(aggregate_fft_score, 4),
            "latest_fft_score": round(latest_frame.fft_score, 4),
            "aggregate_blur_score": round(aggregate_blur_score, 4),
            "passive_verdict": passive_verdict.value,
            "passive_confidence": round(passive_confidence, 4),
            "passive_quality_issues": passive_quality_issues,
            "override_reason": override_reason,
            "frames_analyzed": len(session.frames),
        },
    }

    session_store.cache_result(session_id, response_data)
    return create_response(request_id, response_data)


@router.post("/liveness/stream/end")
async def end_stream(
    session_id: str = Form(...),
    api_key: str = Depends(verify_api_key),
):
    del api_key
    request_id = generate_request_id()

    if not session_store.remove(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return create_response(
        request_id,
        {
            "request_id": request_id,
            "session_id": session_id,
            "ended": True,
        },
    )
