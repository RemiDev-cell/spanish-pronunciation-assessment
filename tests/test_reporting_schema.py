import json
from pathlib import Path

from jsonschema import validators

from src.config import get_settings
from src.reporting import assemble_report
from src.scoring import compute_domain_scores, compute_global_scores


def _validator():
    schema_path = Path(__file__).resolve().parents[1] / "json_schema" / "evaluation_report.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return validators.validator_for(schema)(schema)


def test_sample_report_validates_against_schema():
    from src.models import PhonologyIssue
    from src.models import LocalizedErrorType
    from src.models import FeatureBundle

    issues = [
        PhonologyIssue(
            error_type=LocalizedErrorType.DEBIT_INADAPTE,
            target_unit="phrase",
            precise_location="global",
            confidence=0.5,
            observation="test",
            observed="x",
            expected="y",
            perceptual_effect="z",
            correction="c",
            priority="moyenne",
            score_penalty_hint=0.2,
        )
    ]
    feat = FeatureBundle()
    domains = compute_domain_scores(issues, feat)
    global_scores = compute_global_scores(domains)
    report = assemble_report(
        evaluation_status="evaluable",
        expected_text="hola",
        asr_text="hola",
        warnings=[],
        domains=domains,
        global_scores=global_scores,
        issues=issues,
        settings=get_settings(),
    )
    instance = json.loads(report.model_dump_json())
    assert instance["comparison_type"] == "audio_vs_reference_audio"
    assert "alignment_artifacts" in instance
    assert "audio_quality" in instance
    assert "raw_metrics" in instance
    v = _validator()
    v.validate(instance)
