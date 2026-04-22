from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch

from app.core.config import settings
from app.core.errors import ModelUnavailableError
from app.vendor.physnet_model import PhysNet_padding_Encoder_Decoder_MAX


class RPPGResult(NamedTuple):
    signal: np.ndarray
    confidence: float
    debug: dict | None = None


class PhysNetService:
    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path or settings.rppg_model_path)
        self._model: PhysNet_padding_Encoder_Decoder_MAX | None = None
        self._device = torch.device("cpu")

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise ModelUnavailableError(f"rPPG model not found: {self.model_path}")

        try:
            model = PhysNet_padding_Encoder_Decoder_MAX(frames=settings.stream_min_frames).to(self._device)
            state_dict = torch.load(str(self.model_path), map_location=self._device)
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
        except Exception as exc:
            raise ModelUnavailableError(f"Failed to load rPPG model: {exc}") from exc

    def infer(self, frames: np.ndarray) -> RPPGResult:
        self._load()
        assert self._model is not None

        if frames.ndim != 4:
            raise ValueError("frames must have shape [N, H, W, C]")
        if frames.shape[0] == 0:
            raise ValueError("frames must not be empty")

        input_tensor = self._preprocess(frames)

        try:
            with torch.no_grad():
                output = self._model(input_tensor)[0]
        except Exception as exc:
            raise RuntimeError(f"rPPG inference failed: {exc}") from exc

        signal = output.detach().cpu().numpy().reshape(-1).astype(np.float32)
        signal = signal - float(np.mean(signal))
        signal_std = float(np.std(signal))
        if signal_std > 0:
            signal = signal / signal_std

        return RPPGResult(
            signal=signal,
            confidence=0.0,
            debug={
                "model_path": str(self.model_path),
                "input_shape": list(input_tensor.shape),
                "output_shape": list(output.shape),
                "backend": "torch",
            },
        )

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        target_h = 128
        target_w = 128
        processed = []
        for frame in frames[: settings.stream_min_frames]:
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            normalized = resized.astype(np.float32) / 255.0
            processed.append(np.transpose(normalized, (2, 0, 1)))

        if len(processed) < settings.stream_min_frames:
            last = processed[-1]
            while len(processed) < settings.stream_min_frames:
                processed.append(last)

        array = np.stack(processed, axis=1).astype(np.float32)
        tensor = torch.from_numpy(array).unsqueeze(0).to(self._device)
        return tensor


rppg_service = PhysNetService()
