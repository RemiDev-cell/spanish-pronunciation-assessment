"""Pydantic models mirroring the public evaluation JSON schema."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class EvaluationStatus(str, Enum):
    EVALUABLE = "evaluable"
    EVALUABLE_WITH_WARNING = "evaluable_with_warning"
    NON_EVALUABLE = "non_evaluable"


class LocalizedErrorType(str, Enum):
    ACCENT_TONIQUE_MAL_PLACE = "accent_tonique_mal_place"
    SYLLABE_TONIQUE_PAS_ASSEZ_SAILLANTE = "syllabe_tonique_pas_assez_saillante"
    VOYELLE_MAL_REALISEE = "voyelle_mal_realisee"
    CONSONNE_MAL_REALISEE = "consonne_mal_realisee"
    SUBSTITUTION_SEGMENTALE = "substitution_segmentale"
    OMISSION_SEGMENTALE = "omission_segmentale"
    AJOUT_SEGMENTAL = "ajout_segmental"
    INTONATION_NON_CONFORME = "intonation_non_conforme"
    PAUSE_MAL_PLACEE = "pause_mal_placee"
    DEBIT_INADAPTE = "debit_inadapte"
    DISFLUENCE = "disfluence"
    ARTICULATION_FLOUE = "articulation_floue"


class GlobalScores(BaseModel):
    score_total_points: float = 0
    percent_similarity: float = 0
    percent_delta: float = 0
    interpretation: str = ""


class DomainScoreBlock(BaseModel):
    points_obtained: float = 0
    points_max: float
    percent_similarity: float = 0
    percent_delta: float = 0
    interpretation: str = ""


class DomainScores(BaseModel):
    segmental_precision: DomainScoreBlock = Field(
        default_factory=lambda: DomainScoreBlock(points_max=30)
    )
    syllabic_structure_and_articulation: DomainScoreBlock = Field(
        default_factory=lambda: DomainScoreBlock(points_max=15)
    )
    lexical_stress: DomainScoreBlock = Field(
        default_factory=lambda: DomainScoreBlock(points_max=20)
    )
    sentence_prosody: DomainScoreBlock = Field(
        default_factory=lambda: DomainScoreBlock(points_max=20)
    )
    fluency_and_voice_control: DomainScoreBlock = Field(
        default_factory=lambda: DomainScoreBlock(points_max=10)
    )
    phonological_intelligibility: DomainScoreBlock = Field(
        default_factory=lambda: DomainScoreBlock(points_max=5)
    )


class LocalizedError(BaseModel):
    type_erreur: str
    unite_cible: str
    localisation_precise: str
    niveau_confiance: float = Field(ge=0, le=1)
    observation: str
    realisation_observee: str
    realisation_attendue: str
    effet_perceptif: str
    correction_concrete: str
    priorite: str


class FeedbackItem(BaseModel):
    observation: str
    effect: str
    recommendation: str
    priority: str


ComparisonType = Literal["audio_vs_reference_audio"]


class EvaluationReport(BaseModel):
    evaluation_status: Literal["evaluable", "evaluable_with_warning", "non_evaluable"]
    language: Literal["es"] = "es"
    comparison_type: ComparisonType = "audio_vs_reference_audio"
    model_audio_path: str = ""
    learner_audio_path: str = ""
    asr_model_text: str = ""
    global_scores: GlobalScores = Field(default_factory=GlobalScores)
    domain_scores: DomainScores = Field(default_factory=DomainScores)
    localized_errors: list[LocalizedError] = Field(default_factory=list)
    priority_issues: list[str] = Field(default_factory=list)
    feedbacks: list[FeedbackItem] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    final_summary: str = ""
    expected_text: str = ""
    asr_text: str = ""
    warnings: list[str] = Field(default_factory=list)
    alignment_artifacts: dict[str, str] = Field(default_factory=dict)
    audio_quality: dict[str, Any] = Field(default_factory=dict)
    raw_metrics: dict[str, Any] = Field(default_factory=dict)

    def model_dump_json_pretty(self) -> str:
        return self.model_dump_json(indent=2, exclude_none=True)


# --- Internal pipeline structures (not part of public schema) ---


class WordExpectation(BaseModel):
    """Expected phonological info for one word from text + silabeador."""

    surface: str
    syllables: list[str]
    stressed_syllable_index: int  # 0-based from word onset
    char_start_in_normalized: int = 0
    char_end_in_normalized: int = 0


class PhoneInterval(BaseModel):
    label: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class WordInterval(BaseModel):
    label: str
    start: float
    end: float
    phones: list[PhoneInterval] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class AlignmentResult(BaseModel):
    words: list[WordInterval]
    textgrid_path: Optional[str] = None


class AudioQualityReport(BaseModel):
    duration_sec: float
    rms_db: float
    hf_energy_ratio: float  # HEURISTIC noise proxy
    is_evaluable: bool
    reason: Optional[str] = None


class ASRValidationResult(BaseModel):
    similarity: float
    status: EvaluationStatus
    warnings: list[str] = Field(default_factory=list)


class FeatureBundle(BaseModel):
    """Aggregated acoustic/prosodic features for scoring."""

    word_durations: list[float] = Field(default_factory=list)
    phone_durations: list[float] = Field(default_factory=list)
    pause_durations: list[float] = Field(default_factory=list)
    speech_rate_wpm: float = 0.0
    mean_f0_hz: Optional[float] = None
    median_f0_hz: Optional[float] = None
    f0_std_hz: Optional[float] = None
    f0_std_semitones: Optional[float] = None
    mean_intensity_db: Optional[float] = None
    intensity_std_db: Optional[float] = None
    intensity_range_db: Optional[float] = None
    word_prominence_z: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # per-word keys -> syllable index -> {"duration","f0_mean","intensity_mean","prominence_z"}
    global_phone_duration_mean: float = 0.0
    global_phone_duration_std: float = 1.0
    raw_debug: dict[str, Any] = Field(default_factory=dict)


class PhonologyIssue(BaseModel):
    """Intermediate issue before mapping to LocalizedError + scoring."""

    error_type: LocalizedErrorType
    target_unit: str
    precise_location: str
    confidence: float
    observation: str
    observed: str
    expected: str
    perceptual_effect: str
    correction: str
    priority: str
    score_penalty_hint: float = 0.0  # internal weight 0–1 per domain
