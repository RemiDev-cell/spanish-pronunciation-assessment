from src.config import get_settings
from src.models import ASRValidationResult, AudioQualityReport, EvaluationStatus, FeatureBundle
from src.pipeline import (
    _audio_quality_evidence,
    _non_evaluable_confidence_payload,
    _vowel_formant_evidence,
)


def _vf(f1, f2):
    return {"phone": "a", "f1_hz": f1, "f2_hz": f2}


def test_vowel_formant_evidence_reports_low_when_no_complete_pairs():
    evidence = _vowel_formant_evidence(
        FeatureBundle(vowel_formants=[_vf(None, 1200.0)]),
        FeatureBundle(vowel_formants=[_vf(700.0, 1300.0)]),
        get_settings(),
    )

    assert evidence["level"] == "low"
    assert evidence["complete_f1_f2_pair_count"] == 0


def test_vowel_formant_evidence_reports_high_with_enough_complete_pairs():
    vowels = [_vf(700.0, 1200.0), _vf(500.0, 2100.0), _vf(350.0, 900.0)]
    evidence = _vowel_formant_evidence(
        FeatureBundle(vowel_formants=vowels),
        FeatureBundle(vowel_formants=vowels),
        get_settings(),
    )

    assert evidence["level"] == "high"
    assert evidence["complete_pair_coverage"] == 1.0


def test_audio_quality_evidence_marks_blocking_failure():
    evidence = _audio_quality_evidence(
        model=AudioQualityReport(
            duration_sec=1.0,
            rms_db=-20.0,
            hf_energy_ratio=0.1,
            is_evaluable=True,
        ),
        learner=AudioQualityReport(
            duration_sec=1.0,
            rms_db=-80.0,
            hf_energy_ratio=0.1,
            is_evaluable=False,
            reason="low_rms:-80.0dB",
        ),
    )

    assert evidence["level"] == "blocking"
    assert evidence["learner_reason"] == "low_rms:-80.0dB"


def test_non_evaluable_confidence_reports_asr_blocker():
    payload = _non_evaluable_confidence_payload(
        reason="asr_text_mismatch_model_or_learner",
        model_asr=ASRValidationResult(
            similarity=90.0,
            status=EvaluationStatus.EVALUABLE,
        ),
        learner_asr=ASRValidationResult(
            similarity=40.0,
            status=EvaluationStatus.NON_EVALUABLE,
        ),
    )

    assert payload["overall"]["level"] == "non_evaluable"
    assert payload["asr_script_match"]["level"] == "blocking"
    assert payload["asr_script_match"]["minimum_similarity"] == 40.0
