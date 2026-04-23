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
        self.model_path = Path(model_path or settings.rppg_model_path)
        self._runtime_model_path = self.model_path
        self._session: ort.InferenceSession | None = None
        self._input_name: str = "input"
        self._output_name: str = "rppg"
        self._expected_frames = settings.stream_min_frames
        self._session_options = ort.SessionOptions()
        self._session_options.intra_op_num_threads = settings.onnx_intra_op_threads
        self._session_options.inter_op_num_threads = settings.onnx_inter_op_threads

    def _resolve_runtime_model_path(self) -> Path:
        if self.model_path.suffix.lower() == ".pth":
            onnx_candidate = self.model_path.with_suffix(".onnx")
            if onnx_candidate.exists():
                return onnx_candidate
        return self.model_path

    def _load(self) -> None:
        if self._session is not None:
            return

        runtime_model_path = self._resolve_runtime_model_path()
        self._runtime_model_path = runtime_model_path
        if not runtime_model_path.exists():
            raise ModelUnavailableError(f"rPPG model not found: {runtime_model_path}")
        if runtime_model_path.suffix.lower() != ".onnx":
            raise ModelUnavailableError(
                f"rPPG runtime model harus .onnx (ditemukan: {runtime_model_path.name}). "
                "Jalankan scripts/export_pth_to_onnx.sh terlebih dahulu."
            )

        try:
            session = ort.InferenceSession(
                str(runtime_model_path),
                sess_options=self._session_options,
                providers=["CPUExecutionProvider"],
            )
            self._session = session
            self._input_name = session.get_inputs()[0].name
            self._output_name = session.get_outputs()[0].name

            input_shape = session.get_inputs()[0].shape
            frame_dim = input_shape[2] if len(input_shape) >= 3 else settings.stream_min_frames
            if isinstance(frame_dim, int):
                self._expected_frames = frame_dim

            if self._expected_frames != settings.stream_min_frames:
                raise ModelUnavailableError(
                    "Mismatch stream_min_frames dengan model ONNX rPPG: "
                    f"stream_min_frames={settings.stream_min_frames}, model_frames={self._expected_frames}."
                )
        except Exception as exc:
            raise ModelUnavailableError(f"Failed to load rPPG model: {exc}") from exc

    def infer(self, frames: np.ndarray) -> RPPGResult:
        self._load()
        assert self._session is not None

        if frames.ndim != 4:
            raise ValueError("frames must have shape [N, H, W, C]")
        if frames.shape[0] == 0:
            raise ValueError("frames must not be empty")

        input_tensor = self._preprocess(frames)

        try:
            output = self._session.run(
                [self._output_name],
                {self._input_name: input_tensor},
            )[0]
        except Exception as exc:
            raise RuntimeError(f"rPPG inference failed: {exc}") from exc

        signal = output.reshape(-1).astype(np.float32)
        signal = signal - float(np.mean(signal))
        signal_std = float(np.std(signal))
        if signal_std > 0:
            signal = signal / signal_std

        return RPPGResult(
            signal=signal,
            confidence=0.0,
            debug={
                "model_path": str(self.model_path),
                "runtime_model_path": str(self._runtime_model_path),
                "input_shape": list(input_tensor.shape),
                "output_shape": list(output.shape),
                "backend": "onnxruntime",
            },
        )

    def _preprocess(self, frames: np.ndarray) -> np.ndarray:
        target_h = 128
        target_w = 128
        target_frames = self._expected_frames
        processed = []
        for frame in frames[:target_frames]:
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            normalized = resized.astype(np.float32) / 255.0
            processed.append(np.transpose(normalized, (2, 0, 1)))

        if len(processed) < target_frames:
            last = processed[-1]
            while len(processed) < target_frames:
                processed.append(last)

        array = np.stack(processed, axis=1).astype(np.float32)
        return np.expand_dims(array, axis=0).astype(np.float32)


rppg_service = PhysNetService()
