from src.config import get_settings
from src.models import FeatureBundle
from src.pipeline import _vowel_formant_evidence


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
