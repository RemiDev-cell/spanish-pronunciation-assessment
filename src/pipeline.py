"""End-to-end orchestration + CLI."""

from __future__ import annotations

import argparse
import traceback
import uuid
from pathlib import Path
from typing import List, Literal, Optional, cast

from src.align import run_mfa_align
from src.asr_validation import validate_asr_against_expected
from src.config import Settings, get_settings
from src.features import count_voiced_pitch_frames, extract_features
from src.models import EvaluationReport, EvaluationStatus
from src.preprocess import ensure_mono_wav
from src.reference_pair_phonology import collect_reference_pair_issues
from src.reporting import assemble_report, non_evaluable_report
from src.scoring import compute_domain_scores, compute_global_scores
from src.spanish_phonology import collect_phonology_issues
from src.text_processing import build_word_expectations, normalize_expected_text
from src.transcribe import transcribe_spanish


def _merge_asr_status(a: EvaluationStatus, b: EvaluationStatus) -> EvaluationStatus:
    if a == EvaluationStatus.NON_EVALUABLE or b == EvaluationStatus.NON_EVALUABLE:
        return EvaluationStatus.NON_EVALUABLE
    if a == EvaluationStatus.EVALUABLE_WITH_WARNING or b == EvaluationStatus.EVALUABLE_WITH_WARNING:
        return EvaluationStatus.EVALUABLE_WITH_WARNING
    return EvaluationStatus.EVALUABLE


def run_pipeline(
    audio_path: Path,
    expected_text: str,
    settings: Settings,
    *,
    strict_text_match: bool,
    allow_partial_match: bool,
    debug: bool,
) -> EvaluationReport:
    expected_norm = normalize_expected_text(expected_text)
    if not expected_norm.strip():
        return non_evaluable_report(
            reason="empty_expected_text",
            expected_text=expected_text,
            test_audio_path=str(audio_path.resolve()),
        )

    job = uuid.uuid4().hex[:10]
    work = settings.work_dir / f"job_{job}"
    work.mkdir(parents=True, exist_ok=True)

    try:
        wav_path, quality = ensure_mono_wav(audio_path, work, settings)
    except RuntimeError as e:
        return non_evaluable_report(
            reason=str(e),
            expected_text=expected_norm,
            warnings=[str(e)],
            test_audio_path=str(audio_path.resolve()),
        )

    if not quality.is_evaluable:
        return non_evaluable_report(
            reason=f"audio_quality:{quality.reason}",
            expected_text=expected_norm,
            warnings=[quality.reason or "audio_quality"],
            test_audio_path=str(audio_path.resolve()),
        )

    try:
        asr = transcribe_spanish(wav_path, model_name=settings.whisper_model)
    except Exception as e:
        if debug:
            traceback.print_exc()
        return non_evaluable_report(
            reason=f"whisper_failed:{e}",
            expected_text=expected_norm,
            warnings=["whisper_failed"],
            test_audio_path=str(audio_path.resolve()),
        )

    asr_val = validate_asr_against_expected(
        asr.text,
        expected_norm,
        settings,
        strict=strict_text_match,
        allow_partial=allow_partial_match,
    )
    if asr_val.status == EvaluationStatus.NON_EVALUABLE:
        return non_evaluable_report(
            reason="asr_text_mismatch",
            expected_text=expected_norm,
            asr_text=asr.text,
            warnings=asr_val.warnings + [f"asr_similarity={asr_val.similarity:.1f}"],
            test_audio_path=str(audio_path.resolve()),
        )

    warnings: list[str] = list(asr_val.warnings)
    EvalStatus = Literal["evaluable", "evaluable_with_warning", "non_evaluable"]
    eval_status: EvalStatus = cast(
        EvalStatus,
        "evaluable_with_warning"
        if asr_val.status == EvaluationStatus.EVALUABLE_WITH_WARNING
        else "evaluable",
    )

    expectations = build_word_expectations(expected_norm)
    surfaces = [e.surface for e in expectations]

    try:
        alignment = run_mfa_align(wav_path, surfaces, work, settings)
    except RuntimeError as e:
        if debug:
            traceback.print_exc()
        return non_evaluable_report(
            reason=str(e),
            expected_text=expected_norm,
            asr_text=asr.text,
            warnings=[str(e)],
            test_audio_path=str(audio_path.resolve()),
        )

    try:
        if count_voiced_pitch_frames(str(wav_path)) < 5 and settings.require_parselmouth:
            return non_evaluable_report(
                reason="no_voiced_segments_detected",
                expected_text=expected_norm,
                asr_text=asr.text,
                warnings=["no_voiced_segments_detected"],
                test_audio_path=str(audio_path.resolve()),
            )
    except Exception as e:
        if settings.require_parselmouth:
            return non_evaluable_report(
                reason=f"parselmouth_pitch_failed:{e}",
                expected_text=expected_norm,
                asr_text=asr.text,
                warnings=["parselmouth_pitch_failed"],
                test_audio_path=str(audio_path.resolve()),
            )

    try:
        feat = extract_features(str(wav_path), alignment, expectations)
    except Exception as e:
        if debug:
            traceback.print_exc()
        return non_evaluable_report(
            reason=f"feature_extraction_failed:{e}",
            expected_text=expected_norm,
            asr_text=asr.text,
            warnings=["feature_extraction_failed"],
            test_audio_path=str(audio_path.resolve()),
        )

    issues = collect_phonology_issues(alignment, expectations, feat, settings)
    domains = compute_domain_scores(issues, feat)
    global_scores = compute_global_scores(domains)

    return assemble_report(
        evaluation_status=eval_status,
        expected_text=expected_norm,
        asr_text=asr.text,
        warnings=warnings,
        domains=domains,
        global_scores=global_scores,
        issues=issues,
        settings=settings,
        comparison_type="audio_vs_expected_text",
        test_audio_path=str(audio_path.resolve()),
    )


def run_pipeline_reference_pair(
    reference_audio_path: Path,
    test_audio_path: Path,
    expected_text: str,
    settings: Settings,
    *,
    strict_text_match: bool,
    allow_partial_match: bool,
    debug: bool,
) -> EvaluationReport:
    """
    Same script read by reference (model) and learner (test).
    HEURISTIC: deltas are acoustic/temporal vs reference MFA alignment, not native G2P truth.
    """
    expected_norm = normalize_expected_text(expected_text)
    ref_s = str(reference_audio_path.resolve())
    te_s = str(test_audio_path.resolve())
    ctx = dict(
        comparison_type=cast(Literal["audio_vs_reference_audio"], "audio_vs_reference_audio"),
        reference_audio_path=ref_s,
        test_audio_path=te_s,
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
        wav_ref, q_ref = ensure_mono_wav(reference_audio_path, work / "ref", settings)
        wav_te, q_te = ensure_mono_wav(test_audio_path, work / "test", settings)
    except RuntimeError as e:
        return non_evaluable_report(reason=str(e), expected_text=expected_norm, warnings=[str(e)], **ctx)

    if not q_ref.is_evaluable:
        return non_evaluable_report(
            reason=f"reference_audio_quality:{q_ref.reason}",
            expected_text=expected_norm,
            warnings=[f"reference:{q_ref.reason}"],
            **ctx,
        )
    if not q_te.is_evaluable:
        return non_evaluable_report(
            reason=f"test_audio_quality:{q_te.reason}",
            expected_text=expected_norm,
            warnings=[f"test:{q_te.reason}"],
            **ctx,
        )

    try:
        asr_ref = transcribe_spanish(wav_ref, model_name=settings.whisper_model)
        asr_te = transcribe_spanish(wav_te, model_name=settings.whisper_model)
    except Exception as e:
        if debug:
            traceback.print_exc()
        return non_evaluable_report(
            reason=f"whisper_failed:{e}",
            expected_text=expected_norm,
            warnings=["whisper_failed"],
            **ctx,
        )

    v_ref = validate_asr_against_expected(
        asr_ref.text, expected_norm, settings, strict=strict_text_match, allow_partial=allow_partial_match
    )
    v_te = validate_asr_against_expected(
        asr_te.text, expected_norm, settings, strict=strict_text_match, allow_partial=allow_partial_match
    )
    merged = _merge_asr_status(v_ref.status, v_te.status)
    if merged == EvaluationStatus.NON_EVALUABLE:
        return non_evaluable_report(
            reason="asr_text_mismatch_ref_or_test",
            expected_text=expected_norm,
            asr_text=asr_te.text,
            asr_reference_text=asr_ref.text,
            warnings=v_ref.warnings
            + v_te.warnings
            + [f"asr_similarity_ref={v_ref.similarity:.1f}", f"asr_similarity_test={v_te.similarity:.1f}"],
            **ctx,
        )

    warnings: list[str] = []
    if v_ref.warnings:
        warnings.extend([f"ref:{w}" for w in v_ref.warnings])
    if v_te.warnings:
        warnings.extend([f"test:{w}" for w in v_te.warnings])

    EvalStatus = Literal["evaluable", "evaluable_with_warning", "non_evaluable"]
    eval_status: EvalStatus = cast(
        EvalStatus,
        "evaluable_with_warning" if merged == EvaluationStatus.EVALUABLE_WITH_WARNING else "evaluable",
    )

    expectations = build_word_expectations(expected_norm)
    surfaces = [e.surface for e in expectations]

    try:
        align_ref = run_mfa_align(wav_ref, surfaces, work / "mfa_ref", settings)
        align_te = run_mfa_align(wav_te, surfaces, work / "mfa_test", settings)
    except RuntimeError as e:
        if debug:
            traceback.print_exc()
        return non_evaluable_report(
            reason=str(e),
            expected_text=expected_norm,
            asr_text=asr_te.text,
            asr_reference_text=asr_ref.text,
            warnings=[str(e)],
            **ctx,
        )

    try:
        if settings.require_parselmouth:
            if count_voiced_pitch_frames(str(wav_ref)) < 5:
                return non_evaluable_report(
                    reason="no_voiced_segments_detected_reference",
                    expected_text=expected_norm,
                    asr_text=asr_te.text,
                    asr_reference_text=asr_ref.text,
                    warnings=["reference:no_voiced_segments"],
                    **ctx,
                )
            if count_voiced_pitch_frames(str(wav_te)) < 5:
                return non_evaluable_report(
                    reason="no_voiced_segments_detected_test",
                    expected_text=expected_norm,
                    asr_text=asr_te.text,
                    asr_reference_text=asr_ref.text,
                    warnings=["test:no_voiced_segments"],
                    **ctx,
                )
    except Exception as e:
        if settings.require_parselmouth:
            return non_evaluable_report(
                reason=f"parselmouth_pitch_failed:{e}",
                expected_text=expected_norm,
                asr_text=asr_te.text,
                asr_reference_text=asr_ref.text,
                warnings=["parselmouth_pitch_failed"],
                **ctx,
            )

    try:
        feat_ref = extract_features(str(wav_ref), align_ref, expectations)
        feat_te = extract_features(str(wav_te), align_te, expectations)
    except Exception as e:
        if debug:
            traceback.print_exc()
        return non_evaluable_report(
            reason=f"feature_extraction_failed:{e}",
            expected_text=expected_norm,
            asr_text=asr_te.text,
            asr_reference_text=asr_ref.text,
            warnings=["feature_extraction_failed"],
            **ctx,
        )

    issues = collect_reference_pair_issues(
        align_ref, align_te, feat_ref, feat_te, expectations, settings
    )
    domains = compute_domain_scores(issues, feat_te)
    global_scores = compute_global_scores(domains)

    return assemble_report(
        evaluation_status=eval_status,
        expected_text=expected_norm,
        asr_text=asr_te.text,
        warnings=warnings,
        domains=domains,
        global_scores=global_scores,
        issues=issues,
        settings=settings,
        comparison_type="audio_vs_reference_audio",
        reference_audio_path=ref_s,
        test_audio_path=te_s,
        asr_reference_text=asr_ref.text,
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Spanish phonetic assessment: (1) learner audio vs expected text, or "
            "(2) learner vs reference audio with the same script."
        )
    )
    p.add_argument(
        "--audio",
        type=Path,
        help="Learner / test audio (wav/mp3/m4a/mp4). Required unless using legacy single mode with only --audio (see below).",
    )
    p.add_argument("--reference-audio", type=Path, default=None, help="Reference (model) reading of the same script.")
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

    if args.reference_audio is not None:
        if args.audio is None:
            p.error("--audio (learner/test recording) is required when using --reference-audio.")
        report = run_pipeline_reference_pair(
            args.reference_audio.expanduser().resolve(),
            args.audio.expanduser().resolve(),
            args.text,
            settings,
            strict_text_match=bool(args.strict_text_match),
            allow_partial_match=bool(args.allow_partial_match),
            debug=bool(args.debug),
        )
    else:
        if args.audio is None:
            p.error("--audio is required.")
        report = run_pipeline(
            args.audio.expanduser().resolve(),
            args.text,
            settings,
            strict_text_match=bool(args.strict_text_match),
            allow_partial_match=bool(args.allow_partial_match),
            debug=bool(args.debug),
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
