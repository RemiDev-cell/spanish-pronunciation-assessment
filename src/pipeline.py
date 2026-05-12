"""End-to-end orchestration + CLI."""

from __future__ import annotations

import argparse
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Any, List, Literal, Optional, cast

from src.align import run_mfa_align
from src.asr_validation import validate_asr_against_expected
from src.config import Settings, get_settings
from src.features import count_voiced_pitch_frames, extract_features
from src.gop_speaker_adapted import (
    compute_phoneme_gop,
    fine_tune_on_heygen,
    gop_results_to_phonology_issues,
)
from src.models import (
    AlignmentResult,
    ASRValidationResult,
    AudioQualityReport,
    EvaluationReport,
    EvaluationStatus,
    FeatureBundle,
)
from src.preprocess import ensure_mono_wav
from src.reference_pair_phonology import collect_reference_pair_issues
from src.reporting import assemble_report, non_evaluable_report
from src.scoring import compute_domain_scores, compute_global_scores
from src.text_processing import build_word_expectations, normalize_expected_text
from src.transcribe import transcribe_spanish


def _merge_asr_status(a: EvaluationStatus, b: EvaluationStatus) -> EvaluationStatus:
    if a == EvaluationStatus.NON_EVALUABLE or b == EvaluationStatus.NON_EVALUABLE:
        return EvaluationStatus.NON_EVALUABLE
    if a == EvaluationStatus.EVALUABLE_WITH_WARNING or b == EvaluationStatus.EVALUABLE_WITH_WARNING:
        return EvaluationStatus.EVALUABLE_WITH_WARNING
    return EvaluationStatus.EVALUABLE


def _audio_quality_payload(
    *,
    model: Optional[AudioQualityReport] = None,
    learner: Optional[AudioQualityReport] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if model is not None:
        payload["model"] = model.model_dump()
    if learner is not None:
        payload["learner"] = learner.model_dump()
    return payload


def _feature_summary(feat: FeatureBundle) -> dict[str, Any]:
    return {
        "word_durations_sec": feat.word_durations,
        "phone_durations_sec": feat.phone_durations,
        "pause_durations_sec": feat.pause_durations,
        "speech_rate_wpm": feat.speech_rate_wpm,
        "mean_f0_hz": feat.mean_f0_hz,
        "median_f0_hz": feat.median_f0_hz,
        "f0_std_hz": feat.f0_std_hz,
        "f0_std_semitones": feat.f0_std_semitones,
        "mean_intensity_db": feat.mean_intensity_db,
        "intensity_std_db": feat.intensity_std_db,
        "intensity_range_db": feat.intensity_range_db,
        "global_phone_duration_mean_sec": feat.global_phone_duration_mean,
        "global_phone_duration_std_sec": feat.global_phone_duration_std,
        "word_prominence_z": feat.word_prominence_z,
        "vowel_formants": feat.vowel_formants,
        "raw_debug": feat.raw_debug,
    }


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if abs(denominator) < 1e-9:
        return None
    return numerator / denominator


def _raw_metrics_payload(
    model: FeatureBundle,
    learner: FeatureBundle,
    settings: Settings,
) -> dict[str, Any]:
    model_pauses = model.pause_durations
    learner_pauses = learner.pause_durations
    n_pauses = min(len(model_pauses), len(learner_pauses))
    return {
        "assumptions": {
            "same_speaker_mode": settings.same_speaker_mode,
            "same_equipment_and_environment": settings.same_speaker_mode,
            "inter_speaker_normalization_enabled": False,
        },
        "model": _feature_summary(model),
        "learner": _feature_summary(learner),
        "deltas": {
            "speech_rate_wpm_delta": learner.speech_rate_wpm - model.speech_rate_wpm,
            "speech_rate_wpm_ratio_learner_over_model": _safe_ratio(
                learner.speech_rate_wpm, model.speech_rate_wpm
            ),
            "f0_std_hz_delta": None
            if model.f0_std_hz is None or learner.f0_std_hz is None
            else learner.f0_std_hz - model.f0_std_hz,
            "f0_std_semitones_delta": None
            if model.f0_std_semitones is None or learner.f0_std_semitones is None
            else learner.f0_std_semitones - model.f0_std_semitones,
            "mean_f0_hz_delta": None
            if model.mean_f0_hz is None or learner.mean_f0_hz is None
            else learner.mean_f0_hz - model.mean_f0_hz,
            "mean_intensity_db_delta": None
            if model.mean_intensity_db is None or learner.mean_intensity_db is None
            else learner.mean_intensity_db - model.mean_intensity_db,
            "intensity_range_db_delta": None
            if model.intensity_range_db is None or learner.intensity_range_db is None
            else learner.intensity_range_db - model.intensity_range_db,
            "pause_duration_deltas_sec": [
                learner_pauses[i] - model_pauses[i] for i in range(n_pauses)
            ],
            "vowel_formant_deltas": _vowel_formant_deltas(model, learner),
        },
    }


def _vowel_formant_deltas(
    model: FeatureBundle,
    learner: FeatureBundle,
) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    n = min(len(model.vowel_formants), len(learner.vowel_formants))
    for i in range(n):
        m = model.vowel_formants[i]
        learner_item = learner.vowel_formants[i]
        f1_m, f1_l = m.get("f1_hz"), learner_item.get("f1_hz")
        f2_m, f2_l = m.get("f2_hz"), learner_item.get("f2_hz")
        deltas.append(
            {
                "index": i,
                "word_index": m.get("word_index"),
                "phone": m.get("phone"),
                "learner_phone": learner_item.get("phone"),
                "f1_delta_hz": None if f1_m is None or f1_l is None else f1_l - f1_m,
                "f2_delta_hz": None if f2_m is None or f2_l is None else f2_l - f2_m,
            }
        )
    return deltas


def _phoneme_alignments_from_mfa(alignment: AlignmentResult) -> list[dict[str, Any]]:
    alignments: list[dict[str, Any]] = []
    for word_index, word in enumerate(alignment.words):
        for phone_index, phone in enumerate(word.phones):
            label = phone.label.strip()
            if not label:
                continue
            alignments.append(
                {
                    "phoneme": label,
                    "t_start": phone.start,
                    "t_end": phone.end,
                    "word": word.label,
                    "word_index": word_index,
                    "phone_index": phone_index,
                }
            )
    return alignments


def _maybe_compute_speaker_adapted_gop(
    *,
    learner_wav: Path,
    learner_alignment: AlignmentResult,
    settings: Settings,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if not settings.same_speaker_mode or not settings.gop_enabled:
        return []

    phoneme_alignments = _phoneme_alignments_from_mfa(learner_alignment)
    if not phoneme_alignments:
        warnings.append("gop_speaker_adapted:no_phoneme_alignments")
        return []

    try:
        adapter_path = fine_tune_on_heygen(
            str(settings.gop_heygen_reference_dir),
            str(settings.gop_speaker_model_dir),
            base_model_id=settings.gop_base_model_id,
            force_retrain=False,
        )
        return compute_phoneme_gop(
            str(learner_wav),
            phoneme_alignments,
            adapter_path,
            settings.gop_base_model_id,
        )
    except RuntimeError as e:
        warnings.append(f"gop_speaker_adapted_skipped:{e}")
        return []
    except Exception as e:
        warnings.append(f"gop_speaker_adapted_failed:{e}")
        return []


def _vowel_formant_evidence(
    model: FeatureBundle,
    learner: FeatureBundle,
    settings: Settings,
) -> dict[str, Any]:
    paired_count = min(len(model.vowel_formants), len(learner.vowel_formants))
    complete_pairs = 0
    for i in range(paired_count):
        m = model.vowel_formants[i]
        learner_item = learner.vowel_formants[i]
        if (
            m.get("f1_hz") is not None
            and m.get("f2_hz") is not None
            and learner_item.get("f1_hz") is not None
            and learner_item.get("f2_hz") is not None
        ):
            complete_pairs += 1

    coverage = complete_pairs / paired_count if paired_count else 0.0
    if complete_pairs >= settings.min_vowel_formant_pairs_for_high_confidence and coverage >= 0.7:
        level = "high"
    elif complete_pairs >= settings.min_vowel_formant_pairs_for_medium_confidence:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "model_vowel_count": len(model.vowel_formants),
        "learner_vowel_count": len(learner.vowel_formants),
        "paired_vowel_count": paired_count,
        "complete_f1_f2_pair_count": complete_pairs,
        "complete_pair_coverage": round(coverage, 3),
    }


def _confidence_by_domain_payload(
    *,
    model: FeatureBundle,
    learner: FeatureBundle,
    model_asr: ASRValidationResult,
    learner_asr: ASRValidationResult,
    settings: Settings,
) -> dict[str, Any]:
    formants = _vowel_formant_evidence(model, learner, settings)
    prosody_level = "high" if model.f0_std_hz is not None and learner.f0_std_hz is not None else "low"
    alignment_level = (
        "high"
        if model.raw_debug.get("n_words_aligned") == learner.raw_debug.get("n_words_aligned")
        else "medium"
    )
    asr_similarity = min(model_asr.similarity, learner_asr.similarity)
    asr_level = "high" if asr_similarity >= settings.asr_warn_threshold else "medium"
    return {
        "asr_script_match": {
            "level": asr_level,
            "model_similarity": round(model_asr.similarity, 2),
            "learner_similarity": round(learner_asr.similarity, 2),
        },
        "alignment": {
            "level": alignment_level,
            "model_words_aligned": model.raw_debug.get("n_words_aligned"),
            "learner_words_aligned": learner.raw_debug.get("n_words_aligned"),
        },
        "vowel_quality": formants,
        "prosody": {
            "level": prosody_level,
            "model_f0_std_hz_available": model.f0_std_hz is not None,
            "learner_f0_std_hz_available": learner.f0_std_hz is not None,
        },
    }


def _audio_quality_evidence(
    *,
    model: Optional[AudioQualityReport] = None,
    learner: Optional[AudioQualityReport] = None,
) -> dict[str, Any]:
    reports = [r for r in (model, learner) if r is not None]
    if not reports:
        return {"level": "unknown", "available": False}
    ok = all(r.is_evaluable for r in reports)
    return {
        "level": "high" if ok else "blocking",
        "available": True,
        "model_is_evaluable": None if model is None else model.is_evaluable,
        "learner_is_evaluable": None if learner is None else learner.is_evaluable,
        "model_reason": None if model is None else model.reason,
        "learner_reason": None if learner is None else learner.reason,
    }


def _non_evaluable_confidence_payload(
    *,
    reason: str,
    model_quality: Optional[AudioQualityReport] = None,
    learner_quality: Optional[AudioQualityReport] = None,
    model_asr: Optional[ASRValidationResult] = None,
    learner_asr: Optional[ASRValidationResult] = None,
    alignment_attempted: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "overall": {"level": "non_evaluable", "blocking_reason": reason},
        "audio_quality": _audio_quality_evidence(model=model_quality, learner=learner_quality),
        "alignment": {
            "level": "blocking" if alignment_attempted else "unavailable",
            "attempted": alignment_attempted,
        },
        "vowel_quality": {
            "level": "unavailable",
            "reason": "feature_extraction_not_completed",
        },
        "prosody": {
            "level": "unavailable",
            "reason": "feature_extraction_not_completed",
        },
    }
    if model_asr is not None and learner_asr is not None:
        asr_similarity = min(model_asr.similarity, learner_asr.similarity)
        payload["asr_script_match"] = {
            "level": "blocking"
            if model_asr.status == EvaluationStatus.NON_EVALUABLE
            or learner_asr.status == EvaluationStatus.NON_EVALUABLE
            else "medium",
            "model_similarity": round(model_asr.similarity, 2),
            "learner_similarity": round(learner_asr.similarity, 2),
            "minimum_similarity": round(asr_similarity, 2),
        }
    else:
        payload["asr_script_match"] = {"level": "unavailable"}
    return payload


def _alignment_artifacts_payload(
    model_alignment: AlignmentResult,
    learner_alignment: AlignmentResult,
    artifact_dir: Optional[Path],
    job_id: str,
) -> dict[str, str]:
    model_tg = model_alignment.textgrid_path or ""
    learner_tg = learner_alignment.textgrid_path or ""
    payload = {
        "model_textgrid_path": model_tg,
        "learner_textgrid_path": learner_tg,
    }
    if artifact_dir is None or not model_tg or not learner_tg:
        return payload

    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_dst = artifact_dir / f"{job_id}_model.TextGrid"
        learner_dst = artifact_dir / f"{job_id}_learner.TextGrid"
        shutil.copy2(model_tg, model_dst)
        shutil.copy2(learner_tg, learner_dst)
    except OSError as e:
        payload["artifact_copy_error"] = str(e)
        return payload

    return {
        "model_textgrid_path": str(model_dst),
        "learner_textgrid_path": str(learner_dst),
    }


def evaluate_reference_pair(
    model_audio_path: Path,
    learner_audio_path: Path,
    expected_text: str,
    settings: Settings,
    *,
    strict_text_match: bool,
    allow_partial_match: bool,
    debug: bool,
    artifact_dir: Optional[Path] = None,
) -> EvaluationReport:
    """
    Same script read by the same speaker: personal reference and later evaluated take.
    HEURISTIC: deltas are acoustic/temporal vs reference MFA alignment, not native G2P truth.
    """
    expected_norm = normalize_expected_text(expected_text)
    model_s = str(model_audio_path.resolve())
    learner_s = str(learner_audio_path.resolve())
    ctx = dict(
        model_audio_path=model_s,
        learner_audio_path=learner_s,
    )

    if not expected_norm.strip():
        reason = "empty_expected_text"
        return non_evaluable_report(
            reason=reason,
            expected_text=expected_text,
            confidence_by_domain=_non_evaluable_confidence_payload(reason=reason),
            **ctx,
        )

    job = uuid.uuid4().hex[:10]
    work = settings.work_dir / f"job_refpair_{job}"
    work.mkdir(parents=True, exist_ok=True)

    try:
        wav_model, q_model = ensure_mono_wav(model_audio_path, work / "model", settings)
        wav_learner, q_learner = ensure_mono_wav(learner_audio_path, work / "learner", settings)
    except RuntimeError as e:
        reason = str(e)
        return non_evaluable_report(
            reason=reason,
            expected_text=expected_norm,
            warnings=[str(e)],
            confidence_by_domain=_non_evaluable_confidence_payload(reason=reason),
            **ctx,
        )

    if not q_model.is_evaluable:
        reason = f"model_audio_quality:{q_model.reason}"
        return non_evaluable_report(
            reason=reason,
            expected_text=expected_norm,
            warnings=[f"model:{q_model.reason}"],
            audio_quality=_audio_quality_payload(model=q_model),
            confidence_by_domain=_non_evaluable_confidence_payload(
                reason=reason,
                model_quality=q_model,
            ),
            **ctx,
        )
    if not q_learner.is_evaluable:
        reason = f"learner_audio_quality:{q_learner.reason}"
        return non_evaluable_report(
            reason=reason,
            expected_text=expected_norm,
            warnings=[f"learner:{q_learner.reason}"],
            audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
            confidence_by_domain=_non_evaluable_confidence_payload(
                reason=reason,
                model_quality=q_model,
                learner_quality=q_learner,
            ),
            **ctx,
        )

    try:
        asr_model = transcribe_spanish(wav_model, model_name=settings.whisper_model)
        asr_learner = transcribe_spanish(wav_learner, model_name=settings.whisper_model)
    except Exception as e:
        if debug:
            traceback.print_exc()
        reason = f"whisper_failed:{e}"
        return non_evaluable_report(
            reason=reason,
            expected_text=expected_norm,
            warnings=["whisper_failed"],
            audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
            confidence_by_domain=_non_evaluable_confidence_payload(
                reason=reason,
                model_quality=q_model,
                learner_quality=q_learner,
            ),
            **ctx,
        )

    v_model = validate_asr_against_expected(
        asr_model.text, expected_norm, settings, strict=strict_text_match, allow_partial=allow_partial_match
    )
    v_learner = validate_asr_against_expected(
        asr_learner.text, expected_norm, settings, strict=strict_text_match, allow_partial=allow_partial_match
    )
    merged = _merge_asr_status(v_model.status, v_learner.status)
    if merged == EvaluationStatus.NON_EVALUABLE:
        reason = "asr_text_mismatch_model_or_learner"
        return non_evaluable_report(
            reason=reason,
            expected_text=expected_norm,
            asr_text=asr_learner.text,
            asr_model_text=asr_model.text,
            warnings=v_model.warnings
            + v_learner.warnings
            + [
                f"asr_similarity_model={v_model.similarity:.1f}",
                f"asr_similarity_learner={v_learner.similarity:.1f}",
            ],
            audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
            confidence_by_domain=_non_evaluable_confidence_payload(
                reason=reason,
                model_quality=q_model,
                learner_quality=q_learner,
                model_asr=v_model,
                learner_asr=v_learner,
            ),
            **ctx,
        )

    warnings: list[str] = []
    if v_model.warnings:
        warnings.extend([f"model:{w}" for w in v_model.warnings])
    if v_learner.warnings:
        warnings.extend([f"learner:{w}" for w in v_learner.warnings])

    EvalStatus = Literal["evaluable", "evaluable_with_warning", "non_evaluable"]
    eval_status: EvalStatus = cast(
        EvalStatus,
        "evaluable_with_warning" if merged == EvaluationStatus.EVALUABLE_WITH_WARNING else "evaluable",
    )

    expectations = build_word_expectations(expected_norm)
    surfaces = [e.surface for e in expectations]

    try:
        align_model = run_mfa_align(wav_model, surfaces, work / "mfa_model", settings)
        align_learner = run_mfa_align(wav_learner, surfaces, work / "mfa_learner", settings)
    except RuntimeError as e:
        if debug:
            traceback.print_exc()
        reason = str(e)
        return non_evaluable_report(
            reason=reason,
            expected_text=expected_norm,
            asr_text=asr_learner.text,
            asr_model_text=asr_model.text,
            warnings=[str(e)],
            audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
            confidence_by_domain=_non_evaluable_confidence_payload(
                reason=reason,
                model_quality=q_model,
                learner_quality=q_learner,
                model_asr=v_model,
                learner_asr=v_learner,
                alignment_attempted=True,
            ),
            **ctx,
        )
    alignment_artifacts = _alignment_artifacts_payload(
        align_model,
        align_learner,
        artifact_dir,
        job,
    )

    try:
        if settings.require_parselmouth:
            if count_voiced_pitch_frames(str(wav_model)) < 5:
                reason = "no_voiced_segments_detected_model"
                return non_evaluable_report(
                    reason=reason,
                    expected_text=expected_norm,
                    asr_text=asr_learner.text,
                    asr_model_text=asr_model.text,
                    warnings=["model:no_voiced_segments"],
                    audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
                    alignment_artifacts=alignment_artifacts,
                    confidence_by_domain=_non_evaluable_confidence_payload(
                        reason=reason,
                        model_quality=q_model,
                        learner_quality=q_learner,
                        model_asr=v_model,
                        learner_asr=v_learner,
                        alignment_attempted=True,
                    ),
                    **ctx,
                )
            if count_voiced_pitch_frames(str(wav_learner)) < 5:
                reason = "no_voiced_segments_detected_learner"
                return non_evaluable_report(
                    reason=reason,
                    expected_text=expected_norm,
                    asr_text=asr_learner.text,
                    asr_model_text=asr_model.text,
                    warnings=["learner:no_voiced_segments"],
                    audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
                    alignment_artifacts=alignment_artifacts,
                    confidence_by_domain=_non_evaluable_confidence_payload(
                        reason=reason,
                        model_quality=q_model,
                        learner_quality=q_learner,
                        model_asr=v_model,
                        learner_asr=v_learner,
                        alignment_attempted=True,
                    ),
                    **ctx,
                )
    except Exception as e:
        if settings.require_parselmouth:
            reason = f"parselmouth_pitch_failed:{e}"
            return non_evaluable_report(
                reason=reason,
                expected_text=expected_norm,
                asr_text=asr_learner.text,
                asr_model_text=asr_model.text,
                warnings=["parselmouth_pitch_failed"],
                audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
                alignment_artifacts=alignment_artifacts,
                confidence_by_domain=_non_evaluable_confidence_payload(
                    reason=reason,
                    model_quality=q_model,
                    learner_quality=q_learner,
                    model_asr=v_model,
                    learner_asr=v_learner,
                    alignment_attempted=True,
                ),
                **ctx,
            )

    try:
        feat_model = extract_features(str(wav_model), align_model, expectations)
        feat_learner = extract_features(str(wav_learner), align_learner, expectations)
    except Exception as e:
        if debug:
            traceback.print_exc()
        reason = f"feature_extraction_failed:{e}"
        return non_evaluable_report(
            reason=reason,
            expected_text=expected_norm,
            asr_text=asr_learner.text,
            asr_model_text=asr_model.text,
            warnings=["feature_extraction_failed"],
            audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
            alignment_artifacts=alignment_artifacts,
            confidence_by_domain=_non_evaluable_confidence_payload(
                reason=reason,
                model_quality=q_model,
                learner_quality=q_learner,
                model_asr=v_model,
                learner_asr=v_learner,
                alignment_attempted=True,
            ),
            **ctx,
        )

    gop_results = _maybe_compute_speaker_adapted_gop(
        learner_wav=wav_learner,
        learner_alignment=align_learner,
        settings=settings,
        warnings=warnings,
    )

    issues = collect_reference_pair_issues(
        align_model, align_learner, feat_model, feat_learner, expectations, settings
    )
    if gop_results:
        issues = sorted(
            issues + gop_results_to_phonology_issues(gop_results),
            key=lambda x: x.score_penalty_hint * x.confidence,
            reverse=True,
        )[:25]
    domains = compute_domain_scores(issues, feat_learner)
    global_scores = compute_global_scores(domains)
    raw_metrics = _raw_metrics_payload(feat_model, feat_learner, settings)
    if gop_results:
        raw_metrics["gop_speaker_adapted"] = {
            "base_model_id": settings.gop_base_model_id,
            "adapter_dir": str(settings.gop_speaker_model_dir / "lora_adapter"),
            "phoneme_scores": gop_results,
        }

    return assemble_report(
        evaluation_status=eval_status,
        expected_text=expected_norm,
        asr_text=asr_learner.text,
        warnings=warnings,
        domains=domains,
        global_scores=global_scores,
        issues=issues,
        settings=settings,
        model_audio_path=model_s,
        learner_audio_path=learner_s,
        asr_model_text=asr_model.text,
        alignment_artifacts=alignment_artifacts,
        audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
        raw_metrics=raw_metrics,
        confidence_by_domain=_confidence_by_domain_payload(
            model=feat_model,
            learner=feat_learner,
            model_asr=v_model,
            learner_asr=v_learner,
            settings=settings,
        ),
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Spanish same-speaker pronunciation assessment: evaluated audio vs personal "
            "reference audio with the same script."
        )
    )
    p.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Learner audio (wav/mp3/m4a/mp4).",
    )
    p.add_argument(
        "--reference-audio",
        type=Path,
        required=True,
        help="Same speaker's personal reference reading of the same script.",
    )
    p.add_argument("--text", required=True, help="Shared Spanish script (same words for both recordings).")
    p.add_argument("--output", type=Path, default=None, help="Write JSON report to this path.")
    p.add_argument(
        "--strict-text-match",
        action="store_true",
        help="Stricter ASR vs text gate (more non_evaluable).",
    )
    p.add_argument(
        "--allow-partial-match",
        action="store_true",
        help="Looser ASR gate (HEURISTIC partial_ratio boost).",
    )
    p.add_argument("--debug", action="store_true", help="Print stack traces on failures.")
    args = p.parse_args(argv)

    settings = get_settings()
    artifact_dir = (
        args.output.expanduser().resolve().parent
        if args.output
        else settings.output_dir
    ) / "aligned_textgrids"

    report = evaluate_reference_pair(
        args.reference_audio.expanduser().resolve(),
        args.audio.expanduser().resolve(),
        args.text,
        settings,
        strict_text_match=bool(args.strict_text_match),
        allow_partial_match=bool(args.allow_partial_match),
        debug=bool(args.debug),
        artifact_dir=artifact_dir,
    )

    text = report.model_dump_json(indent=2)
    if args.output:
        args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
