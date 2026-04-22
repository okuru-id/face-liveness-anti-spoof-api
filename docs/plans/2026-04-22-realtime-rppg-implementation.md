# Real-Time rPPG Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement real-time video liveness detection pipeline dengan face detection, MiniFASNet ONNX, dan rPPG PhysNet ONNX yang menghasilkan verdictライブ/SPOOF berbasis sinyal fisiologis actionable.

**Architecture:** Stream init → session initialization → frame buffering → MiniFASNet fast rejectchecking → PhysNet inference pada buffer frame → SignalAnalysis untuk metrik sinyal → Fusion engine menggabungkan hasil → decision endpoint.

**Tech Stack:** FastAPI, ONNXRuntime, OpenCV, NumPy, Pydantic, Python 3.13

---

## Setup & Prerequisites

### Task 1: Verifikasi environment dan model

**Files:**
- Verify: `models/` folder
- Check: `requirements.txt`

**Step 1: Check model files**

```bash
ls -lh models/
```
Expected: `4_0_0_80x80_MiniFASNetV1SE.onnx`, `2.7_80x80_MiniFASNetV2.onnx` tersedia

**Step 2: Check ONNX runtime**

```bash
python -c "import onnxruntime; print(onnxruntime.get_device())"
```
Expected: `CPU` (atau `GPU` bila tersedia)

**Step 3: Install dependencies** (bila perlu)

```bash
pip install -r requirements.txt
```

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: verify环境 setup"
```

---

## Services Layer

### Task 2: Buat `app/services/rppg_physnet.py`

**Files:**
- Create: `app/services/rppg_physnet.py`

**Step 1: Tulis service ONNX untuk rPPG**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import onnxruntime as ort

from app.core.config import settings
from app.core.errors import ModelUnavailableError


class RPPGResult(NamedTuple):
    signal: np.ndarray
    confidence: float
    debug: dict | None = None


class PhysNetService:
    def __init__(self, model_path: str | None = None):
        raw_path = model_path or settings.rppg_model_path
        self.model_path = Path(raw_path.strip())
        self.session: ort.InferenceSession | None = None

    def _load(self):
        if self.session is not None:
            return
        if not self.model_path.exists():
            raise ModelUnavailableError(f"rPPG model not found: {self.model_path}")
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
                sess_options=ort.SessionOptions(),
            )
        except Exception as e:
            raise ModelUnavailableError(f"Failed to load rPPG model: {e}")

    def infer(self, frames: np.ndarray) -> RPPGResult:
        """
        Args:
            frames: np.ndarray dengan shape [N, H, W, C] (BGR), dtype uint8
        Returns:
            signal: np.ndarray dengan shape [T] (normalized signal)
        """
        self._load()
        assert self.session is not None

        try:
            # Preprocess: resize to model input dim
            target_h, target_w = 80, 80
            batch = []
            for frame in frames:
                resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                normalized = resized.astype(np.float32) / 255.0
                # [H, W, C] -> [C, H, W]
                transposed = np.transpose(normalized, (2, 0, 1))
                batch.append(transposed)
            
            input_tensor = np.stack(batch).astype(np.float32)
            # Tambah batch dim jika single frame
            if input_tensor.ndim == 3:
                input_tensor = np.expand_dims(input_tensor, axis=0)

            # ONNX inference
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            signal_output = self.session.run([output_name], {input_name: input_tensor})[0]

            # Normalize signal
            signal = signal_output.flatten()
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
            confidence = 0.0  # Will be set by SignalAnalyzer

            return RPPGResult(
                signal=signal,
                confidence=confidence,
                debug={"output_shape": list(signal_output.shape)},
            )
        except Exception as e:
            raise RuntimeError(f"rPPG inference failed: {e}")


rppg_service = PhysNetService()
```

**Step 2: Update `app/services/__init__.py`**

```python
from app.services.rppg_physnet import rppg_service, PhysNetService

__all__ = ["rppg_service", "PhysNetService"]
```

**Step 3: Update `.env.example`**

```env
RPPG_MODEL_PATH=models/rppg_physnet.onnx
```

**Step 4: Commit**

```bash
git add app/services/rppg_physnet.py app/services/__init__.py .env.example
git commit -m "feat: add rPPG PhysNet ONNX service"
```

---

### Task 3: Buat `app/services/signal_analysis.py`

**Files:**
- Create: `app/services/signal_analysis.py`

**Step 1: Implement signal analysis**

```python
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

from app.core.config import settings


class SignalMetrics(NamedTuple):
    signal_valid: bool
    signal_confidence: float
    estimated_hr: float | None
    peak_count: int
    snr_db: float
    flags: list[str]


@dataclass
class SignalAnalyzer:
    fps: float = 6.0  # Frame rate target

    def analyze(self, signal: np.ndarray, min_frames: int = 6) -> SignalMetrics:
        """
        Analyze rPPG signal and compute metrics.
        """
        if len(signal) < min_frames:
            return SignalMetrics(
                signal_valid=False,
                signal_confidence=0.0,
                estimated_hr=None,
                peak_count=0,
                snr_db=0.0,
                flags=["insufficient_frames"],
            )

        # Normalize
        sig = signal - np.mean(signal)
        sig = sig / (np.std(sig) + 1e-6)

        # FFT untuk estimated HR
        n = len(sig)
        fft = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)
        ps = np.abs(fft) ** 2

        # Find dominant frequency (0.5-3.0 Hz = 30-180 bpm)
        valid_idx = (freqs >= 0.5) & (freqs <= 3.0)
        if not np.any(valid_idx):
            return SignalMetrics(
                signal_valid=False,
                signal_confidence=0.0,
                estimated_hr=None,
                peak_count=0,
                snr_db=0.0,
                flags=["no_valid_frequency_range"],
            )

        valid_ps = ps[valid_idx]
        valid_freqs = freqs[valid_idx]
        peak_idx = np.argmax(valid_ps)
        dominant_freq = float(valid_freqs[peak_idx])
        estimated_hr = dominant_freq * 60.0

        # SNR estimation (power di dominant freq vs noise band)
        noise band = ps[(freqs < 0.5) | (freqs > 3.0)]
        signal_power = valid_ps[peak_idx]
        noise_power = np.mean(noise_band) if len(noise_band) > 0 else 1e-6
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-6))

        # Peak counting
        peaks, _ = self._find_peaks(sig, distance=int(0.5 * self.fps))
        peak_count = len(peaks)

        # Confidence score
        confidence = 0.0
        flags = []

        if snr_db < 0:
            confidence = 0.1
            flags.append("low_snr")
        elif snr_db < 5:
            confidence = 0.4
            flags.append("moderate_snr")
        else:
            confidence = min(0.95, 0.5 + snr_db / 20.0)

        if peak_count < 2:
            confidence *= 0.8
            flags.append("few_peaks")
        elif peak_count > 6:
            confidence *= 0.95
            flags.append("many_peaks")

        if not (40 <= estimated_hr <= 180):
            confidence *= 0.5
            flags.append("hr_out_of_range")
            estimated_hr = None

        signal_valid = confidence >= 0.4

        return SignalMetrics(
            signal_valid=signal_valid,
            signal_confidence=confidence,
            estimated_hr=estimated_hr,
            peak_count=peak_count,
            snr_db=snr_db,
            flags=flags,
        )

    def _find_peaks(self, sig: np.ndarray, distance: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Simple peak detection (replace with scipy.signal.find_peaks if available)."""
        peaks = []
        for i in range(1, len(sig) - 1):
            if sig[i] > sig[i - 1] and sig[i] > sig[i + 1]:
                peaks.append(i)
        return np.array(peaks), np.ones(len(peaks))
```

**Step 2: Update `app/services/__init__.py`**

```python
from app.services.signal_analysis import SignalAnalyzer, SignalMetrics

__all__ = ["SignalAnalyzer", "SignalMetrics"]
```

**Step 3: Commit**

```bash
git add app/services/signal_analysis.py app/services/__init__.py
git commit -m "feat: add SignalAnalyzer for rPPG metrics"
```

---

### Task 4: Buat `app/services/fusion.py`

**Files:**
- Create: `app/services/fusion.py`

**Step 1: Implement fusion logic**

```python
from typing import NamedTuple

import numpy as np

from app.services.anti_spoof import AntiSpoofResult, SpoofLabel
from app.services.rppg_physnet import RPPGResult
from app.services.signal_analysis import SignalMetrics
from app.core.config import settings


class FusionResult(NamedTuple):
    verdict: str  # "LIVE", "SPOOF", "UNCERTAIN"
    confidence: float
    signal_confidence: float
    mini_fas_confidence: float
    fusion_debug: dict


def fuse(
    mini_fas: AntiSpoofResult,
    rppg_result: RPPGResult,
    signal: SignalMetrics,
) -> FusionResult:
    """
    Fusion logic untuk kombinasi MiniFASNet dan rPPG.
    - MiniFASNet bobot 0.6
    - rPPG confidence bobot 0.4
    """
    mini_fas_conf = mini_fas.confidence
    rppg_conf = signal.signal_confidence

    # Hitung composite score
    live_score = 0.0
    if mini_fas.label == SpoofLabel.LIVE:
        # Skor higher jika MiniFASNet confident LIVE
        live_score = 0.6 * mini_fas_conf + 0.4 * rppg_conf
    else:
        # Jika MiniFASNet SPOOF, kurangi score
        live_score = 0.6 * (1 - mini_fas_conf) * 0.1 + 0.4 * rppg_conf

    # Ambil confidence final
    confidence = min(live_score, 1.0)
    signal_conf = signal.signal_confidence

    # Decision rule
    if confidence >= settings.fusion_live_threshold:
        verdict = "LIVE"
    elif confidence <= settings.fusion_spoof_threshold:
        verdict = "SPOOF"
    else:
        verdict = "UNCERTAIN"

    return FusionResult(
        verdict=verdict,
        confidence=confidence,
        signal_confidence=signal_conf,
        mini_fas_confidence=mini_fas_conf,
        fusion_debug={
            "live_score": round(float(live_score), 4),
            "mini_fas_label": mini_fas.label.value,
            "rppg_conf": round(signal_conf, 4),
            "snr_db": signal.snr_db,
            "flags": signal.flags,
        },
    )
```

**Step 2: Update config `app/core/config.py`**

```python
    fusion_live_threshold: float = 0.7
    fusion_spoof_threshold: float = 0.3
```

**Step 3: Update `app/services/__init__.py`**

```python
from app.services.fusion import FusionResult, fuse

__all__ = ["FusionResult", "fuse"]
```

**Step 4: Commit**

```bash
git add app/services/fusion.py app/core/config.py app/services/__init__.py
git commit -m "feat: add Fusion engine for combined decision"
```

---

## Core Layer

### Task 5: Buat `app/core/session_store.py`

**Files:**
- Create: `app/core/session_store.py`
- Modify: `app/core/__init__.py`

**Step 1: Implement thread-safe session store**

```python
import threading
import time
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

import numpy as np

from app.services.anti_spoof import AntiSpoofResult, SpoofLabel
from app.schemas.liveness import FaceBBox


@dataclass
class FrameMeta:
    buffer_idx: int
    face_crop: np.ndarray
    mini_fas_result: AntiSpoofResult | None = None
    bbox: FaceBBox | None = None


@dataclass
class LivenessSession:
    session_id: str
    mode: str = "realtime"
    frames: list[FrameMeta] = field(default_factory=list)
    state: Literal["collecting", "ready", "decided", "expired"] = "collecting"
    created_at: float = field(default_factory=time.time)
    window_ms: int = 3000
    min_frames: int = 6

    @property
    def age_ms(self) -> float:
        return (time.time() - self.created_at) * 1000

    @property
    def is_expired(self) -> bool:
        return self.age_ms > 30000  # 30s TTL

    @property
    def has_enough_frames(self) -> bool:
        return len(self.frames) >= self.min_frames


class SessionStore:
    def __init__(self, ttl_ms: int = 30000, max_sessions: int = 100):
        self._sessions: dict[str, LivenessSession] = {}
        self._lock = threading.RLock()
        self._ttl_ms = ttl_ms
        self._max_sessions = max_sessions

    def create(
        self,
        mode: str = "realtime",
        window_ms: int = 3000,
        min_frames: int = 6,
    ) -> LivenessSession:
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                self._expire_old()
            session_id = str(uuid4())[:8]
            session = LivenessSession(
                session_id=session_id,
                mode=mode,
                window_ms=window_ms,
                min_frames=min_frames,
            )
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> LivenessSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.is_expired:
                session.state = "expired"
                del self._sessions[session_id]
                return None
            return session

    def add_frame(
        self,
        session_id: str,
        face_crop: np.ndarray,
        mini_fas_result: AntiSpoofResult | None = None,
        bbox=None,
    ) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.state != "collecting":
                return False
            frame = FrameMeta(
                buffer_idx=len(session.frames),
                face_crop=face_crop,
                mini_fas_result=mini_fas_result,
                bbox=bbox,
            )
            session.frames.append(frame)
            return True

    def set_state(
        self,
        session_id: str,
        new_state: Literal["collecting", "ready", "decided", "expired"],
    ) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            session.state = new_state
            return True

    def expire_old(self) -> list[str]:
        expired_ids = []
        with self._lock:
            now = time.time()
            for sid, session in list(self._sessions.items()):
                if (now - session.created_at) * 1000 > self._ttl_ms:
                    expired_ids.append(sid)
                    del self._sessions[sid]
        return expired_ids

    def remove(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


session_store = SessionStore()
```

**Step 2: Update `app/core/__init__.py`**

```python
from app.core.session_store import session_store, SessionStore, LivenessSession

__all__ = ["session_store", "SessionStore", "LivenessSession"]
```

**Step 3: Commit**

```bash
git add app/core/session_store.py app/core/__init__.py
git commit -m "feat: add thread-safe session store"
```

---

## API Layer

### Task 6: Buat `app/api/routes/stream.py`

**Files:**
- Create: `app/api/routes/stream.py`
- Modify: `app/main.py`

**Step 1: Implementation**

```python
import os
import time
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from app.api.dependencies.auth import verify_api_key
from app.core.config import settings
from app.core.session_store import session_store
from app.services.face_detector import face_detector
from app.services.anti_spoof import anti_spoof_service
from app.services.rppg_physnet import rppg_service
from app.services.signal_analysis import SignalAnalyzer
from app.services.fusion import fuse
from app.schemas.liveness import FaceBBox, LivenessResponse
from app.core.request_id import generate_request_id
from app.api.responses import create_liveness_response

router = APIRouter()


@router.post("/stream/init")
async def init_stream(api_key: str = Depends(verify_api_key)):
    """
    Init session untuk stream liveness. Return session_id dan konfigurasi.
    """
    request_id = generate_request_id()
    START_TIME = time.time()

    session = session_store.create(
        mode="realtime",
        window_ms=settings.stream_window_ms,
        min_frames=settings.stream_min_frames,
    )

    return {
        "request_id": request_id,
        "session_id": session.session_id,
        "window_ms": session.window_ms,
        "min_frames": session.min_frames,
        "frame_interval_ms": int(1000 / settings.stream_frame_rate),
        "processing_time_ms": int((time.time() - START_TIME) * 1000),
    }


@router.post("/stream/frame")
async def upload_frame(
    session_id: str,
    frame: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
):
    """
    Upload satu frame ke session. Proses: decode → quality → face-detect → MiniFASNet.
    Jika spoof kuat → fast_reject. Jika lolos → tambah ke buffer.
    """
    START_TIME = time.time()
    request_id = generate_request_id()

    # Decode frame
    try:
        import base64
        import cv2
        import numpy as np

        content = await frame.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None or image.size == 0:
            raise HTTPException(status_code=400, detail="Invalid image data")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame decode failed: {e}")

    # Quality check
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_brightness = np.mean(gray)

    if blur_score < settings.blur_threshold:
        return {
            "request_id": request_id,
            "session_id": session_id,
            "status": "fast_reject",
            "reason": "blur",
            "processing_time_ms": int((time.time() - START_TIME) * 1000),
        }

    if mean_brightness < settings.brightness_min or mean_brightness > settings.brightness_max:
        return {
            "request_id": request_id,
            "session_id": session_id,
            "status": "fast_reject",
            "reason": "brightness",
            "processing_time_ms": int((time.time() - START_TIME) * 1000),
        }

    # Face detection
    face_result = face_detector.detect(image)
    if not face_result.detected:
        return {
            "request_id": request_id,
            "session_id": session_id,
            "status": "skip",
            "reason": "no_face",
            "processing_time_ms": int((time.time() - START_TIME) * 1000),
        }

    # MiniFASNet inference
    x, y, w, h = face_result.bbox
    face_crop = image[y : y + h, x : x + w]

    try:
        anti_spoof_result = anti_spoof_service.predict(
            face_crop=face_crop,
            bbox=face_result.bbox,
            full_image=image,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MiniFASNet failed: {e}")

    # Fast reject bila spoof kuat
    if anti_spoof_result.label.value == "SPOOF" and anti_spoof_result.confidence >= settings.spoof_threshold:
        return {
            "request_id": request_id,
            "session_id": session_id,
            "status": "fast_reject",
            "reason": "spoof_confident",
            "confidence": anti_spoof_result.confidence,
            "processing_time_ms": int((time.time() - START_TIME) * 1000),
        }

    # Add to buffer
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    success = session_store.add_frame(
        session_id=session_id,
        face_crop=face_crop,
        mini_fas_result=anti_spoof_result,
        bbox=FaceBBox(x=x, y=y, w=w, h=h),
    )

    if not success:
        raise HTTPException(status_code=410, detail="Session ended or invalid")

    return {
        "request_id": request_id,
        "session_id": session_id,
        "status": "collected",
        "frame_idx": len(session.frames),
        "mini_fas_confidence": anti_spoof_result.confidence,
        "processing_time_ms": int((time.time() - START_TIME) * 1000),
    }


@router.get("/stream/result")
async def get_stream_result(session_id: str, api_key: str = Depends(verify_api_key)):
    """
    Get verdict untuk session. Jika buffer ready, jalankan PhysNet → signal analysis → fusion.
    """
    START_TIME = time.time()
    request_id = generate_request_id()

    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    # Check deadline
    if session.state == "expired":
        return create_liveness_response(
            request_id=request_id,
            verdict="UNCERTAIN",
            confidence=0.0,
            spoof_type=None,
            face_detected=False,
            quality_passed=False,
            quality_issues=["session_expired"],
            processing_time_ms=int((time.time() - START_TIME) * 1000),
            face_bbox=None,
            anti_spoof_debug=None,
            rate_limit_info=None,
        )

    if session.state == "decided":
        # Return cached result
        return getattr(session, "_cached_result", None) or {
            "error": "result not cached",
            "request_id": request_id,
        }

    # Check frame count
    if not session.has_enough_frames:
        return {
            "request_id": request_id,
            "session_id": session_id,
            "status": "waiting",
            "frames_collected": len(session.frames),
            "min_frames_required": session.min_frames,
            "processing_time_ms": int((time.time() - START_TIME) * 1000),
        }

    # Collect all face crops
    frames = [fm.face_crop for fm in session.frames]

    # rPPG inference
    try:
        rppg_result = rppg_service.infer(np.stack(frames))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rPPG inference failed: {e}")

    # Signal analysis
    analyzer = SignalAnalyzer(fps=settings.stream_frame_rate)
    signal_metrics = analyzer.analyze(rppg_result.signal, min_frames=session.min_frames)

    # Fusion
    # Ambil MiniFASNet hasil dari frame terakhir
    last_frame = session.frames[-1]
    if last_frame.mini_fas_result is None:
        raise HTTPException(status_code=500, detail="Missing MiniFASNet result")

    fusion_result = fuse(last_frame.mini_fas_result, rppg_result, signal_metrics)

    # Create response
    response = create_liveness_response(
        request_id=request_id,
        verdict=fusion_result.verdict,
        confidence=fusion_result.confidence,
        spoof_type=None,
        face_detected=True,
        quality_passed=True,
        quality_issues=[],
        processing_time_ms=int((time.time() - START_TIME) * 1000),
        face_bbox=last_frame.bbox.dict() if last_frame.bbox else None,
        anti_spoof_debug=None,
        rate_limit_info=None,
    )

    # Cache result di session
    session._cached_result = response
    session.state = "decided"
    session_store.set_state(session_id, "decided")

    return response


@router.post("/stream/end")
async def end_stream(session_id: str, api_key: str = Depends(verify_api_key)):
    """
    End session dan cleanup.
    """
    success = session_store.remove(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "ended": True}
```

**Step 2: Update `app/main.py`**

```python
from app.api.routes import stream

app.include_router(stream.router, prefix="/v1/liveness", tags=["stream"])
```

**Step 3: Commit**

```bash
git add app/api/routes/stream.py app/main.py
git commit -m "feat: add stream API endpoints"
```

---

### Task 7: Update config di `.env.example`

**Files:**
- Modify: `.env.example`

**Step 1: Add stream settings**

```env
# Session/stream settings
STREAM_WINDOW_MS=3000
STREAM_MIN_FRAMES=6
STREAM_FRAME_RATE=6

# Fusion thresholds
FUSION_LIVE_THRESHOLD=0.7
FUSION_SPOOF_THRESHOLD=0.3
```

**Step 2: Copy ke `.env` (bila belum ada)**

```bash
cp .env.example .env
# edit .env untuk set RPPG_MODEL_PATH
```

**Step 3: Commit**

```bash
git add .env.example .env
git commit -m "chore: add stream config"
```

---

## Documentation

### Task 8: Buat `docs/api/stream_endpoints.md`

**Files:**
- Create: `docs/api/stream_endpoints.md`

**Step 1: Document API**

```markdown
# Stream API Endpoints

## Init Session

**POST** `/v1/liveness/stream/init`

Request body: None (api_key via header)

Response:
```json
{
  "request_id": "uuid",
  "session_id": "a1b2c3d4",
  "window_ms": 3000,
  "min_frames": 6,
  "frame_interval_ms": 166,
  "processing_time_ms": 5
}
```

---

## Upload Frame

**POST** `/v1/liveness/stream/frame`

Form-data:
- `session_id`: string
- `frame`: file (JPEG/PNG)

Response (success):
```json
{
  "request_id": "uuid",
  "session_id": "a1b2c3d4",
  "status": "collected",
  "frame_idx": 5,
  "mini_fas_confidence": 0.87,
  "processing_time_ms": 24
}
```

Response (fast reject):
```json
{
  "request_id": "uuid",
  "session_id": "a1b2c3d4",
  "status": "fast_reject",
  "reason": "spoof_confident",
  "confidence": 0.92,
  "processing_time_ms": 18
}
```

---

## Get Result

**GET** `/v1/liveness/stream/result?session_id=a1b2c3d4`

Response:
```json
{
  "request_id": "uuid",
  "verdict": "LIVE",
  "confidence": 0.78,
  "spoof_type": null,
  "face_detected": true,
  "quality_check": {"passed": true, "issues": []},
  "processing_time_ms": 125,
  "timestamp": "2026-04-22T10:30:00Z",
  "face_bbox": {"x": 100, "y": 80, "w": 120, "h": 120},
  "anti_spoof_debug": null
}
```

---

## End Session

**POST** `/v1/liveness/stream/end`

Request body:
```json
{"session_id": "a1b2c3d4"}
```

Response:
```json
{"session_id": "a1b2c3d4", "ended": true}
```

## Latency Targets

| Phase | Target |
|-------|--------|
| MiniFASNet per frame | ≤ 15ms |
| Frame upload round-trip | ≤ 100ms |
| rPPG + signal + fusion (once per window) | ≤ 150ms |
| Total end-to-end | ≤ 200ms |
```

**Step 2: Commit**

```bash
git add docs/api/stream_endpoints.md
git commit -m "docs: add stream API docs"
```

---

## Testing & Validation

### Task 9: Manual testing

**Files:**
- Verify: workshop manual

**Step 1: Jalankan server**

```bash
uvicorn app.main:app --reload
```

**Step 2: Init session**

```bash
curl -X POST http://127.0.0.1:8000/v1/liveness/stream/init \
  -H "X-API-Key: dev-test-key-001"
```

**Step 3: Upload frame (ambil sample JPEG)**

```bash
curl -X POST http://127.0.0.1:8000/v1/liveness/stream/frame \
  -F "session_id=a1b2c3d4" \
  -F "frame=@path/to/sample.jpg" \
  -H "X-API-Key: dev-test-key-001"
```

**Step 4: Repeat upload sampai ≥ 6 frames**

```bash
# lakukan 6 kali upload dengan session_id yang sama
```

**Step 5: Get result**

```bash
curl "http://127.0.0.1:8000/v1/liveness/stream/result?session_id=a1b2c3d4" \
  -H "X-API-Key: dev-test-key-001"
```

**Step 6: Expected output**
- `verdict`: LIVE/SPOOF/UNCERTAIN
- `confidence`: 0.0–1.0
- `processing_time_ms`: < 200

**Step 7: End session**

```bash
curl -X POST http://127.0.0.1:8000/v1/liveness/stream/end \
  -H "X-API-Key: dev-test-key-001" \
  -d '{"session_id":"a1b2c3d4"}'
```

**Step 8: Verify no error logs**

```bash
# check console output untuk exception
```

**Step 9: Commit**

```bash
git add -A
git commit -m "test: manual验证 stream pipeline"
```

---

## Deployment Checklist

### Task 10: Persiapan ONNX model `rppg_physnet.onnx`

**Files:**
- Verify: `models/`

**Step 1: Export/obtain model**

- Jika convert dari PyTorch:
  ```python
  torch.onnx.export(model, dummy_input, "models/rppg_physnet.onnx", opset_version=13)
  ```
- Jika download model:
  - Pastikan model connexion dengan ROI face sequence input
  - Validate output adalah 1D signal array

**Step 2: Validate model load**

```python
import onnxruntime as ort
sess = ort.InferenceSession("models/rppg_physnet.onnx")
print([inp.name for inp in sess.get_inputs()])
print([out.name for out in sess.get_outputs()])
```

**Step 3: Commit**

```bash
git add models/rppg_physnet.onnx
git commit -m "chore: add rPPG ONNX model"
```

---

## Final Verification

### Task 11: End-to-end run

**Files:**
- Verify: full pipeline

**Step 1: Startup server**

```bash
uvicorn app.main:app --reload
```

**Step 2: test session full flow**

1. `init` → dapat `session_id`
2. `upload frame` × 6–10 frames
3. `get result` → verify `verdict` & `confidence`
4. `end session`

**Step 3: Verify no memory leak**

```bash
# Monitor memory usage sebelum dan sesudah 10 sesi
ps aux | grep uvicorn
```

**Step 4: Commit final**

```bash
git add -A
git commit -m "feat: implement full rPPG stream pipeline"
```

---

## Dokumentasi 아카이브

### Task 12: Update README.md (opsional)

**Files:**
- Modify: `README.md`

**Step 1: Tambah section stream API**

```markdown
## Real-Time Stream API

### Init session
```bash
curl -X POST http://127.0.0.1:8000/v1/liveness/stream/init \
  -H "X-API-Key: dev-test-key-001"
```

### Upload frames (min 6)
```bash
curl -X POST http://127.0.0.1:8000/v1/liveness/stream/frame \
  -F "session_id=..." -F "frame=@face.jpg"
```

### Get verdict
```bash
curl "http://127.0.0.1:8000/v1/liveness/stream/result?session_id=..."
```

---
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add stream API section to README"
```

---

## Task Summary

| Task | File | Status |
|------|------|--------|
| 1 | model verification | ✅ |
| 2 | `app/services/rppg_physnet.py` | ➕ |
| 3 | `app/services/signal_analysis.py` | ➕ |
| 4 | `app/services/fusion.py` | ➕ |
| 5 | `app/core/session_store.py` | ➕ |
| 6 | `app/api/routes/stream.py` | ➕ |
| 7 | config `.env` | ✏️ |
| 8 | docs `stream_endpoints.md` | ➕ |
| 9 | manual testing | 🧪 |
| 10 | rPPG ONNX model | ➕ |
| 11 | E2E validation | 🛠️ |
| 12 | README update | 📝 |

**Total estimated:** 12 tasks, ~60–90 menit implementasi + testing.

---

*Plan generated on 2026-04-22.*
