from typing import NamedTuple

from app.core.config import settings
from app.schemas.common import Verdict
from app.services.anti_spoof import AntiSpoofResult, SpoofLabel
from app.services.rppg_physnet import RPPGResult
from app.services.signal_analysis import SignalMetrics


class FusionResult(NamedTuple):
    verdict: Verdict
    confidence: float
    signal_confidence: float
    mini_fas_confidence: float
    fusion_debug: dict


def fuse(
    mini_fas: AntiSpoofResult,
    rppg_result: RPPGResult,
    signal: SignalMetrics,
) -> FusionResult:
    mini_fas_confidence = float(mini_fas.confidence)
    rppg_confidence = float(signal.signal_confidence)

    if mini_fas.label == SpoofLabel.LIVE:
        live_score = (0.6 * mini_fas_confidence) + (0.4 * rppg_confidence)
    else:
        live_score = 0.4 * rppg_confidence

    live_score = max(0.0, min(live_score, 1.0))

    if live_score >= settings.fusion_live_threshold:
        verdict = Verdict.LIVE
    elif live_score <= settings.fusion_spoof_threshold:
        verdict = Verdict.SPOOF
    else:
        verdict = Verdict.UNCERTAIN

    return FusionResult(
        verdict=verdict,
        confidence=live_score,
        signal_confidence=rppg_confidence,
        mini_fas_confidence=mini_fas_confidence,
        fusion_debug={
            "mini_fas_label": mini_fas.label.value,
            "mini_fas_confidence": round(mini_fas_confidence, 4),
            "rppg_signal_confidence": round(rppg_confidence, 4),
            "rppg_debug": rppg_result.debug,
            "snr_db": round(float(signal.snr_db), 4),
            "estimated_hr": signal.estimated_hr,
            "flags": signal.flags,
            "live_score": round(live_score, 4),
        },
    )
