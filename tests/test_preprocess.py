import numpy as np
import soundfile as sf

from src.config import Settings
from src.preprocess import analyze_audio_quality


def _settings(tmp_path):
    return Settings(
        project_root=tmp_path,
        data_dir=tmp_path,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        min_duration_sec=0.1,
        min_rms_db=-80.0,
        max_hf_energy_ratio=1.0,
    )


def test_audio_quality_flags_clipping(tmp_path):
    wav = tmp_path / "clipped.wav"
    data = np.ones(16000, dtype=np.float32)
    sf.write(wav, data, 16000)

    report = analyze_audio_quality(wav, _settings(tmp_path))

    assert report.clipping_ratio > 0.99
    assert not report.is_evaluable
    assert "clipping_ratio" in (report.reason or "")


def test_audio_quality_flags_mostly_silent_audio(tmp_path):
    wav = tmp_path / "silent.wav"
    data = np.zeros(16000, dtype=np.float32)
    sf.write(wav, data, 16000)

    report = analyze_audio_quality(wav, _settings(tmp_path))

    assert report.silence_ratio == 1.0
    assert not report.is_evaluable
    assert "silence_ratio" in (report.reason or "")
