from typing import Optional

from app.core.config import settings
from app.schemas.common import Verdict, SpoofType
from app.services.anti_spoof import AntiSpoofResult, SpoofLabel


def _extract_blur_score(quality_issues: list[str]) -> float | None:
    for issue in quality_issues:
        if not issue.startswith("Image too blurry"):
            continue
        marker = "blur_score="
        start = issue.find(marker)
        if start < 0:
            continue
        start += len(marker)
        end = issue.find(",", start)
        raw_value = issue[start:] if end < 0 else issue[start:end]
        try:
            return float(raw_value)
        except ValueError:
            return None
    return None


def _extract_live_score(anti_spoof_result: AntiSpoofResult) -> float:
    if isinstance(anti_spoof_result.debug, dict):
        avg_probs = anti_spoof_result.debug.get("avg_probs")
        if isinstance(avg_probs, list) and len(avg_probs) > 1:
            try:
                return float(avg_probs[1])
            except (TypeError, ValueError):
                pass

    if anti_spoof_result.label == SpoofLabel.LIVE:
        return float(anti_spoof_result.confidence)

    return float(max(0.0, 1.0 - anti_spoof_result.confidence))


def determine_verdict(
    anti_spoof_result: AntiSpoofResult,
    face_detected: bool,
    quality_passed: bool,
    spoof_type: Optional[SpoofType] = None,
    fft_score: float = 0.0,
    quality_issues: Optional[list[str]] = None,
) -> tuple[Verdict, float, Optional[SpoofType]]:
    if not face_detected:
        return Verdict.NO_FACE, 0.0, None

    quality_issues = quality_issues or []
    has_hard_quality_issue = any(issue.startswith("Resolution too small") for issue in quality_issues)
    if has_hard_quality_issue:
        return Verdict.POOR_QUALITY, 0.0, None

    mini_fas_score = _extract_live_score(anti_spoof_result)
    live_thresh = settings.effective_live_threshold
    spoof_thresh = settings.effective_spoof_threshold

    fft_weight = settings.fft_weight
    fas_weight = settings.fas_weight

    # FFT score: high / (low + eps) → high value = more high-freq = spoof (screen replay)
    # For live likelihood: we need high FFT to mean low live score
    fft_live_score = 1.0 - fft_score

    # Fusion: weighted average of MiniFASNet live score and FFT live score
    final_score = (mini_fas_score * fas_weight) + (fft_live_score * fft_weight)

    def resolve_spoof_type() -> SpoofType:
        if spoof_type is not None:
            return spoof_type
        if fft_score >= (settings.effective_fft_spoof_override_threshold + 0.1):
            return SpoofType.SCREEN_REPLAY
        return SpoofType.UNKNOWN

    if fft_score >= settings.effective_fft_spoof_override_threshold and final_score <= live_thresh:
        return Verdict.SPOOF, final_score, resolve_spoof_type()

    is_blurry = any(issue.startswith("Image too blurry") for issue in quality_issues)
    blur_score = _extract_blur_score(quality_issues)
    if is_blurry:
        if (
            mini_fas_score >= 0.995
            and (
                (
                    fft_score >= settings.effective_blurry_high_fft_spoof_threshold
                    and blur_score is not None
                    and blur_score >= settings.effective_blurry_high_blur_min
                )
                or (
                    fft_score <= settings.effective_blurry_low_fft_spoof_threshold
                    and blur_score is not None
                    and blur_score <= settings.effective_blurry_low_blur_max
                )
            )
        ):
            return Verdict.SPOOF, final_score, resolve_spoof_type()

        if fft_score >= settings.effective_blur_fft_spoof_threshold:
            return Verdict.SPOOF, final_score, resolve_spoof_type()
        if mini_fas_score <= spoof_thresh:
            return Verdict.SPOOF, final_score, resolve_spoof_type()
        if mini_fas_score < settings.blurry_live_min_confidence:
            return Verdict.SPOOF, final_score, resolve_spoof_type()

    if final_score >= live_thresh:
        return Verdict.LIVE, final_score, None

    if final_score <= spoof_thresh:
        return Verdict.SPOOF, final_score, resolve_spoof_type()

    return Verdict.UNCERTAIN, final_score, None
