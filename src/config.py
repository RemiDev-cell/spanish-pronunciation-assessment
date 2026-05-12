"""Application settings (env + defaults)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPANISH_PHON_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=_project_root)
    data_dir: Path = Field(default_factory=lambda: _project_root() / "data")
    work_dir: Path = Field(default_factory=lambda: _project_root() / "data" / "work")
    output_dir: Path = Field(default_factory=lambda: _project_root() / "data" / "output")

    whisper_model: str = "base"
    asr_warn_threshold: float = 72.0
    asr_reject_threshold: float = 55.0

    min_duration_sec: float = 0.8
    min_rms_db: float = -45.0
    max_hf_energy_ratio: float = 0.42  # HEURISTIC: very noisy broadband hiss
    max_clipping_ratio: float = 0.01
    max_silence_ratio: float = 0.92
    silence_amplitude_threshold: float = 0.003

    mfa_binary: str = "mfa"
    mfa_dictionary: str = "spanish_mfa"
    mfa_acoustic_model: str = "spanish_mfa"
    mfa_beam: int = 10
    mfa_retry_beam: int = 120

    max_localized_errors: int = 10

    # Default product assumption: both recordings are from the same speaker and setup.
    same_speaker_mode: bool = True

    # Speaker-adapted GOP (Goodness of Pronunciation) configuration.
    gop_enabled: bool = True
    gop_base_model_id: str = "carlosdanielhernandezmena/wav2vec2-large-xlsr-53-spanish-ep5-944h"
    gop_error_threshold: float = Field(
        default_factory=lambda: float(os.getenv("GOP_ERROR_THRESHOLD", "-2.0"))
    )
    gop_warning_threshold: float = Field(
        default_factory=lambda: float(os.getenv("GOP_WARNING_THRESHOLD", "-1.0"))
    )
    gop_heygen_reference_dir: Path = Field(
        default_factory=lambda: _project_root() / "data" / "heygen_reference"
    )
    gop_speaker_model_dir: Path = Field(
        default_factory=lambda: _project_root() / "models" / "speaker_adapted"
    )
    gop_threshold_table_path: Optional[Path] = None

    # Same-speaker comparison thresholds — HEURISTIC, calibration-ready via env.
    word_duration_ratio_low: float = 0.58
    word_duration_ratio_high: float = 1.42
    word_duration_penalty_scale: float = 0.12
    word_duration_penalty_cap: float = 0.35
    stress_prominence_delta_threshold: float = 0.85
    pause_delta_threshold_sec: float = 0.38
    pause_min_context_sec: float = 0.12
    speech_rate_ratio_low: float = 0.72
    speech_rate_ratio_high: float = 1.35
    f0_std_min_hz_for_intonation: float = 10.0
    f0_std_ratio_low: float = 0.55
    low_f0_std_hz_threshold: float = 10.0
    vowel_formant_f1_delta_threshold_hz: float = 120.0
    vowel_formant_f2_delta_threshold_hz: float = 180.0
    vowel_formant_distance_threshold_hz: float = 220.0
    min_vowel_formant_pairs_for_medium_confidence: int = 1
    min_vowel_formant_pairs_for_high_confidence: int = 3

    # Legacy single-audio heuristics retained for unused compatibility helpers.
    pause_outlier_multiplier: float = 2.8
    min_words_for_pause_stats: int = 3

    # Prominence: stressed syllable should exceed others by this z-score (HEURISTIC)
    stress_prominence_z_expected: float = 0.35

    # If Parselmouth fails, entire evaluation becomes non_evaluable (MVP default)
    require_parselmouth: bool = True


def get_settings() -> Settings:
    return Settings()
