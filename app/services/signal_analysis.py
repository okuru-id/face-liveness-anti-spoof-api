from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class SignalMetrics(NamedTuple):
    signal_valid: bool
    signal_confidence: float
    estimated_hr: float | None
    peak_count: int
    snr_db: float
    flags: list[str]


@dataclass(slots=True)
class SignalAnalyzer:
    fps: float = 6.0

    def analyze(self, signal: np.ndarray, min_frames: int = 6) -> SignalMetrics:
        values = np.asarray(signal, dtype=np.float32).reshape(-1)
        if values.size < min_frames:
            return SignalMetrics(False, 0.0, None, 0, 0.0, ["insufficient_frames"])

        centered = values - float(np.mean(values))
        scale = float(np.std(centered))
        if scale > 0:
            centered = centered / scale

        freqs = np.fft.rfftfreq(centered.size, d=1.0 / self.fps)
        power = np.abs(np.fft.rfft(centered)) ** 2
        valid_band = (freqs >= 0.5) & (freqs <= 3.0)
        if not np.any(valid_band):
            return SignalMetrics(False, 0.0, None, 0, 0.0, ["no_valid_frequency_range"])

        band_freqs = freqs[valid_band]
        band_power = power[valid_band]
        peak_index = int(np.argmax(band_power))
        dominant_freq = float(band_freqs[peak_index])
        estimated_hr = dominant_freq * 60.0

        noise_mask = ~valid_band
        signal_power = float(band_power[peak_index])
        noise_power = float(np.mean(power[noise_mask])) if np.any(noise_mask) else 1.0
        if signal_power <= 0 or noise_power <= 0:
            snr_db = -10.0
        else:
            ratio = max(signal_power, 1e-9) / max(noise_power, 1e-9)
            snr_db = float(np.clip(10.0 * np.log10(ratio), -60.0, 120.0))

        peak_count = self._count_peaks(centered)
        confidence = self._score_confidence(snr_db, peak_count, estimated_hr)
        flags = self._build_flags(snr_db, peak_count, estimated_hr)

        return SignalMetrics(
            signal_valid=confidence >= 0.4,
            signal_confidence=confidence,
            estimated_hr=estimated_hr if 40.0 <= estimated_hr <= 180.0 else None,
            peak_count=peak_count,
            snr_db=float(snr_db),
            flags=flags,
        )

    def _count_peaks(self, signal: np.ndarray) -> int:
        if signal.size < 3:
            return 0

        peaks = 0
        last_peak = -10**9
        min_distance = max(int(self.fps * 0.4), 1)
        for idx in range(1, signal.size - 1):
            if signal[idx] <= signal[idx - 1] or signal[idx] <= signal[idx + 1]:
                continue
            if idx - last_peak < min_distance:
                continue
            peaks += 1
            last_peak = idx
        return peaks

    def _score_confidence(self, snr_db: float, peak_count: int, estimated_hr: float) -> float:
        if snr_db < 0:
            confidence = 0.1
        elif snr_db < 5:
            confidence = 0.4
        else:
            confidence = min(0.95, 0.5 + (snr_db / 20.0))

        if peak_count < 2:
            confidence *= 0.8
        elif peak_count > 6:
            confidence *= 0.95

        if not 40.0 <= estimated_hr <= 180.0:
            confidence *= 0.5

        return float(max(0.0, min(confidence, 1.0)))

    def _build_flags(self, snr_db: float, peak_count: int, estimated_hr: float) -> list[str]:
        flags: list[str] = []
        if snr_db < 0:
            flags.append("low_snr")
        elif snr_db < 5:
            flags.append("moderate_snr")

        if peak_count < 2:
            flags.append("few_peaks")
        elif peak_count > 6:
            flags.append("many_peaks")

        if not 40.0 <= estimated_hr <= 180.0:
            flags.append("hr_out_of_range")

        return flags
