"""Compare Whisper output to expected text — HEURISTIC reading validation."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from src.config import Settings
from src.models import ASRValidationResult, EvaluationStatus


def _norm_for_match(s: str) -> str:
    t = unicodedata.normalize("NFC", s.lower())
    t = re.sub(r"[^\w\sáéíóúüñ]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def validate_asr_against_expected(
    asr_text: str,
    expected_normalized: str,
    settings: Settings,
    *,
    strict: bool,
    allow_partial: bool,
) -> ASRValidationResult:
    """
    strict: treat warn band as reject.
    allow_partial: lower reject threshold slightly (still HEURISTIC).
    """
    a = _norm_for_match(asr_text)
    e = _norm_for_match(expected_normalized)
    if not e:
        return ASRValidationResult(
            similarity=0.0,
            status=EvaluationStatus.NON_EVALUABLE,
            warnings=["empty_expected_text"],
        )

    similarity = float(fuzz.token_sort_ratio(a, e))
    if allow_partial:
        cov_a = fuzz.partial_ratio(a, e)
        similarity = max(similarity, float(cov_a) * 0.97)

    warn_th = settings.asr_warn_threshold
    rej_th = settings.asr_reject_threshold
    if strict:
        warn_th = min(100.0, warn_th + 8.0)
        rej_th = min(100.0, rej_th + 8.0)
    if allow_partial:
        rej_th = max(0.0, rej_th - 7.0)

    warnings: list[str] = []
    if similarity < rej_th:
        return ASRValidationResult(
            similarity=similarity,
            status=EvaluationStatus.NON_EVALUABLE,
            warnings=[f"asr_mismatch_reject:{similarity:.1f}"],
        )
    if similarity < warn_th:
        warnings.append(f"asr_mismatch_warn:{similarity:.1f}")
        status = EvaluationStatus.EVALUABLE_WITH_WARNING
    else:
        status = EvaluationStatus.EVALUABLE

    return ASRValidationResult(similarity=similarity, status=status, warnings=warnings)
