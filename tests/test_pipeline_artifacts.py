from src.models import AlignmentResult
from src.pipeline import _alignment_artifacts_payload


def test_alignment_artifacts_are_copied_to_stable_dir(tmp_path):
    model_tg = tmp_path / "tmp_model.TextGrid"
    learner_tg = tmp_path / "tmp_learner.TextGrid"
    model_tg.write_text("model", encoding="utf-8")
    learner_tg.write_text("learner", encoding="utf-8")

    artifact_dir = tmp_path / "aligned_textgrids"
    payload = _alignment_artifacts_payload(
        AlignmentResult(words=[], textgrid_path=str(model_tg)),
        AlignmentResult(words=[], textgrid_path=str(learner_tg)),
        artifact_dir,
        "job123",
    )

    assert payload == {
        "model_textgrid_path": str(artifact_dir / "job123_model.TextGrid"),
        "learner_textgrid_path": str(artifact_dir / "job123_learner.TextGrid"),
    }
    assert (artifact_dir / "job123_model.TextGrid").read_text(encoding="utf-8") == "model"
    assert (artifact_dir / "job123_learner.TextGrid").read_text(encoding="utf-8") == "learner"


def test_alignment_artifacts_keep_temp_paths_without_artifact_dir(tmp_path):
    model_tg = tmp_path / "tmp_model.TextGrid"
    learner_tg = tmp_path / "tmp_learner.TextGrid"

    payload = _alignment_artifacts_payload(
        AlignmentResult(words=[], textgrid_path=str(model_tg)),
        AlignmentResult(words=[], textgrid_path=str(learner_tg)),
        None,
        "job123",
    )

    assert payload == {
        "model_textgrid_path": str(model_tg),
        "learner_textgrid_path": str(learner_tg),
    }
