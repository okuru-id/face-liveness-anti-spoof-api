import threading
import time
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

import numpy as np

from app.schemas.liveness import FaceBBox
from app.services.anti_spoof import AntiSpoofResult

SessionState = Literal["collecting", "ready", "decided", "expired"]


@dataclass(slots=True)
class FrameMeta:
    face_crop: np.ndarray
    bbox: FaceBBox
    mini_fas_result: AntiSpoofResult
    fft_score: float = 0.0
    blur_score: float = 0.0
    captured_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class LivenessSession:
    session_id: str
    mode: str
    window_ms: int
    min_frames: int
    created_at: float = field(default_factory=time.time)
    state: SessionState = "collecting"
    frames: list[FrameMeta] = field(default_factory=list)
    cached_result: dict | None = None

    @property
    def age_ms(self) -> int:
        return int((time.time() - self.created_at) * 1000)

    @property
    def has_enough_frames(self) -> bool:
        return len(self.frames) >= self.min_frames


class SessionStore:
    def __init__(self, ttl_ms: int | None = None, max_sessions: int = 128):
        from app.core.config import settings
        self._ttl_ms = ttl_ms if ttl_ms is not None else settings.stream_session_ttl_ms
        self._max_sessions = max_sessions
        self._sessions: dict[str, LivenessSession] = {}
        self._lock = threading.RLock()

    def create(self, mode: str, window_ms: int, min_frames: int) -> LivenessSession:
        with self._lock:
            self.expire_old()
            if len(self._sessions) >= self._max_sessions:
                raise RuntimeError("session store is full")

            session = LivenessSession(
                session_id=uuid4().hex[:12],
                mode=mode,
                window_ms=window_ms,
                min_frames=min_frames,
            )
            self._sessions[session.session_id] = session
            return session

    def get(self, session_id: str) -> LivenessSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.age_ms > self._ttl_ms:
                session.state = "expired"
                del self._sessions[session_id]
                return None
            return session

    def add_frame(self, session_id: str, frame: FrameMeta) -> LivenessSession | None:
        with self._lock:
            session = self.get(session_id)
            if session is None or session.state == "expired":
                return None

            session.frames.append(frame)

            # Keep window bounded for continuous stream processing.
            if len(session.frames) > session.min_frames:
                session.frames = session.frames[-session.min_frames :]

            # New frame invalidates cached result so next /result recomputes.
            session.cached_result = None

            if session.has_enough_frames:
                session.state = "ready"
            return session

    def cache_result(self, session_id: str, result: dict) -> bool:
        with self._lock:
            session = self.get(session_id)
            if session is None:
                return False
            session.cached_result = result
            if session.has_enough_frames:
                session.state = "ready"
            return True

    def remove(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def expire_old(self) -> list[str]:
        expired_ids: list[str] = []
        with self._lock:
            for session_id, session in list(self._sessions.items()):
                if session.age_ms <= self._ttl_ms:
                    continue
                session.state = "expired"
                expired_ids.append(session_id)
                del self._sessions[session_id]
        return expired_ids


session_store = SessionStore()
