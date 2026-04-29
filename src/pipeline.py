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
from src.models import (
    AlignmentResult,
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
        "raw_debug": feat.raw_debug,
    }


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if abs(denominator) < 1e-9:
        return None
    return numerator / denominator


def _raw_metrics_payload(
    model: FeatureBundle,
    learner: FeatureBundle,
) -> dict[str, Any]:
    model_pauses = model.pause_durations
    learner_pauses = learner.pause_durations
    n_pauses = min(len(model_pauses), len(learner_pauses))
    return {
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
        },
    }


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
    Same script read by reference (model) and learner.
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
        return non_evaluable_report(
            reason="empty_expected_text",
            expected_text=expected_text,
            **ctx,
        )

    job = uuid.uuid4().hex[:10]
    work = settings.work_dir / f"job_refpair_{job}"
    work.mkdir(parents=True, exist_ok=True)

    try:
        wav_model, q_model = ensure_mono_wav(model_audio_path, work / "model", settings)
        wav_learner, q_learner = ensure_mono_wav(learner_audio_path, work / "learner", settings)
    except RuntimeError as e:
        return non_evaluable_report(reason=str(e), expected_text=expected_norm, warnings=[str(e)], **ctx)

    if not q_model.is_evaluable:
        return non_evaluable_report(
            reason=f"model_audio_quality:{q_model.reason}",
            expected_text=expected_norm,
            warnings=[f"model:{q_model.reason}"],
            audio_quality=_audio_quality_payload(model=q_model),
            **ctx,
        )
    if not q_learner.is_evaluable:
        return non_evaluable_report(
            reason=f"learner_audio_quality:{q_learner.reason}",
            expected_text=expected_norm,
            warnings=[f"learner:{q_learner.reason}"],
            audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
            **ctx,
        )

    try:
        asr_model = transcribe_spanish(wav_model, model_name=settings.whisper_model)
        asr_learner = transcribe_spanish(wav_learner, model_name=settings.whisper_model)
    except Exception as e:
        if debug:
            traceback.print_exc()
        return non_evaluable_report(
            reason=f"whisper_failed:{e}",
            expected_text=expected_norm,
            warnings=["whisper_failed"],
            audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
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
        return non_evaluable_report(
            reason="asr_text_mismatch_model_or_learner",
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
        return non_evaluable_report(
            reason=str(e),
            expected_text=expected_norm,
            asr_text=asr_learner.text,
            asr_model_text=asr_model.text,
            warnings=[str(e)],
            audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
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
                return non_evaluable_report(
                    reason="no_voiced_segments_detected_model",
                    expected_text=expected_norm,
                    asr_text=asr_learner.text,
                    asr_model_text=asr_model.text,
                    warnings=["model:no_voiced_segments"],
                    audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
                    alignment_artifacts=alignment_artifacts,
                    **ctx,
                )
            if count_voiced_pitch_frames(str(wav_learner)) < 5:
                return non_evaluable_report(
                    reason="no_voiced_segments_detected_learner",
                    expected_text=expected_norm,
                    asr_text=asr_learner.text,
                    asr_model_text=asr_model.text,
                    warnings=["learner:no_voiced_segments"],
                    audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
                    alignment_artifacts=alignment_artifacts,
                    **ctx,
                )
    except Exception as e:
        if settings.require_parselmouth:
            return non_evaluable_report(
                reason=f"parselmouth_pitch_failed:{e}",
                expected_text=expected_norm,
                asr_text=asr_learner.text,
                asr_model_text=asr_model.text,
                warnings=["parselmouth_pitch_failed"],
                audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
                alignment_artifacts=alignment_artifacts,
                **ctx,
            )

    try:
        feat_model = extract_features(str(wav_model), align_model, expectations)
        feat_learner = extract_features(str(wav_learner), align_learner, expectations)
    except Exception as e:
        if debug:
            traceback.print_exc()
        return non_evaluable_report(
            reason=f"feature_extraction_failed:{e}",
            expected_text=expected_norm,
            asr_text=asr_learner.text,
            asr_model_text=asr_model.text,
            warnings=["feature_extraction_failed"],
            audio_quality=_audio_quality_payload(model=q_model, learner=q_learner),
            alignment_artifacts=alignment_artifacts,
            **ctx,
        )

    issues = collect_reference_pair_issues(
        align_model, align_learner, feat_model, feat_learner, expectations, settings
    )
    domains = compute_domain_scores(issues, feat_learner)
    global_scores = compute_global_scores(domains)

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
        raw_metrics=_raw_metrics_payload(feat_model, feat_learner),
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Spanish comparative pronunciation assessment: learner audio vs reference "
            "model audio with the same script."
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
        help="Reference model reading of the same script.",
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
