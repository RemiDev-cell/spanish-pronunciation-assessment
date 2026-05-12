from __future__ import annotations

import csv
import json

from scripts.analyze_gop_calibration import iter_gop_rows, load_manifest_labels, summarize_rows
from scripts.export_expert_validation import write_annotation_csv
from src.config import get_settings
from src.models import AlignmentResult, PhoneInterval, WordInterval
from src.pipeline import _maybe_compute_speaker_adapted_gop


def test_maybe_compute_gop_respects_skip_flag():
    payload = _maybe_compute_speaker_adapted_gop(
        learner_wav=__file__,
        learner_alignment=AlignmentResult(words=[]),
        settings=get_settings(),
        warnings=[],
        skip_gop=True,
    )

    assert payload["enabled"] is False
    assert payload["status"] == "skipped_by_cli"
    assert payload["phoneme_scores"] == []


def test_maybe_compute_gop_forwards_force_retrain(monkeypatch, tmp_path):
    import src.pipeline as pipeline

    seen = {}

    def fake_fine_tune(*args, **kwargs):
        seen["force_retrain"] = kwargs["force_retrain"]
        return "adapter"

    monkeypatch.setattr(pipeline, "fine_tune_on_heygen", fake_fine_tune)
    monkeypatch.setattr(
        pipeline,
        "compute_phoneme_gop",
        lambda *args, **kwargs: [{"phoneme": "a", "gop": 0.1, "status": "correct"}],
    )

    alignment = AlignmentResult(
        words=[
            WordInterval(
                label="hola",
                start=0.0,
                end=1.0,
                phones=[PhoneInterval(label="a", start=0.0, end=1.0)],
            )
        ]
    )
    payload = _maybe_compute_speaker_adapted_gop(
        learner_wav=tmp_path / "learner.wav",
        learner_alignment=alignment,
        settings=get_settings(),
        warnings=[],
        force_retrain=True,
    )

    assert seen["force_retrain"] is True
    assert payload["status"] == "computed_with_lora"
    assert payload["phoneme_scores"][0]["phoneme"] == "a"


def test_analyze_gop_calibration_estimates_thresholds(tmp_path):
    correct = tmp_path / "correct_001.gop.json"
    error = tmp_path / "error_001.gop.json"
    manifest = tmp_path / "manifest.json"
    correct.write_text(
        json.dumps({"phoneme_scores": [{"phoneme": "a", "gop": 0.5, "status": "correct"}]}),
        encoding="utf-8",
    )
    error.write_text(
        json.dumps({"phoneme_scores": [{"phoneme": "a", "gop": -3.0, "status": "mispronunciation"}]}),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {"id": "correct_001", "expected_outcome": "correct"},
                    {"id": "error_001", "expected_outcome": "error"},
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = iter_gop_rows([correct, error], load_manifest_labels(manifest))
    summary = summarize_rows(rows)

    assert summary["count"] == 2
    assert summary["by_expected_outcome"] == {"correct": 1, "error": 1}
    assert summary["suggested_thresholds"]["warning_threshold"] == 0.5
    assert summary["suggested_thresholds"]["error_threshold"] == -3.0


def test_export_expert_validation_csv(tmp_path):
    report = tmp_path / "sample.report.json"
    output = tmp_path / "annotations.csv"
    report.write_text(
        json.dumps(
            {
                "model_audio_path": "ref.wav",
                "learner_audio_path": "learner.wav",
                "expected_text": "hola",
                "global_scores": {"score_total_points": 88.0},
                "domain_scores": {"segmental_precision": {"percent_similarity": 91.0}},
                "localized_errors": [{"type_erreur": "substitution_segmentale"}],
            }
        ),
        encoding="utf-8",
    )

    write_annotation_csv([report], output)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert rows[0]["sample_id"] == "sample"
    assert rows[0]["machine_global_score"] == "88.0"
    assert rows[0]["teacher_notes"] == ""
