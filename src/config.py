"""Application settings (env + defaults)."""

from __future__ import annotations

from pathlib import Path

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

    # Long pause vs median pause — HEURISTIC disfluence / boundary confusion
    pause_outlier_multiplier: float = 2.8
    min_words_for_pause_stats: int = 3

    # Prominence: stressed syllable should exceed others by this z-score (HEURISTIC)
    stress_prominence_z_expected: float = 0.35

    # If Parselmouth fails, entire evaluation becomes non_evaluable (MVP default)
    require_parselmouth: bool = True


def get_settings() -> Settings:
    return Settings()
