from src.models import FeatureBundle, PhonologyIssue
from src.models import LocalizedErrorType
from src.scoring import _interpret_fr, compute_domain_scores, compute_global_scores


def test_interpretation_bands():
    assert "quasi identique" in _interpret_fr(92)
    assert "très proche" in _interpret_fr(80)
    assert "satisfaisant" in _interpret_fr(65)
    assert "fragile" in _interpret_fr(50)
    assert "très éloigné" in _interpret_fr(30)


def test_domain_scores_bounded():
    issues = [
        PhonologyIssue(
            error_type=LocalizedErrorType.ACCENT_TONIQUE_MAL_PLACE,
            target_unit="palabra",
            precise_location="test",
            confidence=0.9,
            observation="o",
            observed="o",
            expected="e",
            perceptual_effect="p",
            correction="c",
            priority="haute",
            score_penalty_hint=0.5,
        )
    ]
    feat = FeatureBundle()
    ds = compute_domain_scores(issues, feat)
    assert ds.segmental_precision.points_obtained <= 30
    assert ds.lexical_stress.points_obtained <= 20
    gs = compute_global_scores(ds)
    assert 0 <= gs.score_total_points <= 100
