"""Unit tests for reference vs test audio comparison heuristics."""

from src.config import get_settings
from src.models import AlignmentResult, FeatureBundle, PhoneInterval, WordExpectation, WordInterval
from src.reference_pair_phonology import collect_reference_pair_issues


def _make_word(label: str, dur: float, start: float = 0.0) -> WordInterval:
    end = start + dur
    return WordInterval(
        label=label,
        start=start,
        end=end,
        phones=[PhoneInterval(label="a", start=start, end=start + dur * 0.5), PhoneInterval(label="b", start=start + dur * 0.5, end=end)],
    )


def _prom_bundle(zs: list[float], exp_stress: int, surface: str) -> dict:
    syls = []
    for i, z in enumerate(zs):
        syls.append(
            {
                "duration": 0.1,
                "f0_mean": 180.0,
                "intensity_mean": 70.0,
                "prominence_z": z,
            }
        )
    return {"syllables": syls, "expected_stress_index": exp_stress, "surface": surface}


def test_reference_pair_detects_stress_gap():
    settings = get_settings()
    exp = [WordExpectation(surface="hola", syllables=["ho", "la"], stressed_syllable_index=0)]
    align_ref = AlignmentResult(words=[_make_word("hola", 0.4, 0.0)])
    align_te = AlignmentResult(words=[_make_word("hola", 0.4, 0.0)])

    feat_ref = FeatureBundle(
        word_prominence_z={"0:hola": _prom_bundle([1.3, -0.5], 0, "hola")},
        pause_durations=[],
        speech_rate_wpm=120.0,
        f0_std_hz=25.0,
    )
    feat_te = FeatureBundle(
        word_prominence_z={"0:hola": _prom_bundle([0.35, 0.32], 0, "hola")},
        pause_durations=[],
        speech_rate_wpm=118.0,
        f0_std_hz=24.0,
    )

    issues = collect_reference_pair_issues(
        align_ref, align_te, feat_ref, feat_te, exp, settings
    )
    types = {i.error_type.value for i in issues}
    # argmax still on tonic syllable (0) but prominence z on tonic is much lower than reference
    assert "syllabe_tonique_pas_assez_saillante" in types
