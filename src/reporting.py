"""Assemble localized errors, feedbacks, and final JSON report."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from src.config import Settings
from src.models import (
    DomainScoreBlock,
    DomainScores,
    EvaluationReport,
    FeedbackItem,
    GlobalScores,
    LocalizedError,
    PhonologyIssue,
)


def _priority_fr(p: str) -> str:
    m = {"haute": "haute", "moyenne": "moyenne", "basse": "basse"}
    return m.get(p.lower(), "moyenne")


def issues_to_localized_errors(
    issues: list[PhonologyIssue],
    settings: Settings,
) -> list[LocalizedError]:
    out: list[LocalizedError] = []
    cap = min(10, max(1, settings.max_localized_errors))
    for iss in issues[:cap]:
        out.append(
            LocalizedError(
                type_erreur=iss.error_type.value,
                unite_cible=iss.target_unit,
                localisation_precise=iss.precise_location,
                niveau_confiance=round(float(iss.confidence), 3),
                observation=iss.observation,
                realisation_observee=iss.observed,
                realisation_attendue=iss.expected,
                effet_perceptif=iss.perceptual_effect,
                correction_concrete=iss.correction,
                priorite=_priority_fr(iss.priority),
            )
        )
    return out


def build_feedbacks(errors: list[LocalizedError]) -> list[FeedbackItem]:
    fbs: list[FeedbackItem] = []
    for e in errors[:8]:
        fbs.append(
            FeedbackItem(
                observation=e.observation,
                effect=e.effet_perceptif,
                recommendation=e.correction_concrete,
                priority=e.priorite,
            )
        )
    return fbs


def build_priority_issues(errors: list[LocalizedError]) -> list[str]:
    return [
        f"{e.type_erreur} — {e.unite_cible} ({e.priorite})"
        for e in errors
        if e.priorite == "haute"
    ][:5]


def build_strengths(domains: DomainScores) -> list[str]:
    strengths: list[str] = []
    if domains.lexical_stress.percent_similarity >= 78:
        strengths.append("Stress / saillance proches du locuteur de référence sur plusieurs mots.")
    if domains.sentence_prosody.percent_similarity >= 76:
        strengths.append("Pauses et mouvement prosodique assez alignés avec la référence.")
    if domains.fluency_and_voice_control.percent_similarity >= 76:
        strengths.append("Débit global raisonnablement proche de l'enregistrement de référence.")
    if not strengths:
        strengths.append(
            "Production analysable: poursuivre en ciblant les erreurs localisées ci-dessous."
        )
    return strengths[:4]


def build_final_summary(global_scores: GlobalScores, status: str) -> str:
    if status == "non_evaluable":
        return "Évaluation non disponible: signal audio ou correspondance texte insuffisante."
    return (
        f"Score global {global_scores.score_total_points:.1f}/100 "
        f"({global_scores.percent_similarity:.1f}% de similarité). "
        "Évaluation par rapport à la référence audio et au script partagé: "
        f"{global_scores.interpretation}."
    )


def assemble_report(
    *,
    evaluation_status: Literal["evaluable", "evaluable_with_warning", "non_evaluable"],
    expected_text: str,
    asr_text: str,
    warnings: list[str],
    domains: DomainScores,
    global_scores: GlobalScores,
    issues: list[PhonologyIssue],
    settings: Settings,
    model_audio_path: str = "",
    learner_audio_path: str = "",
    asr_model_text: str = "",
    alignment_artifacts: Optional[dict[str, str]] = None,
    audio_quality: Optional[dict[str, Any]] = None,
    raw_metrics: Optional[dict[str, Any]] = None,
    confidence_by_domain: Optional[dict[str, Any]] = None,
) -> EvaluationReport:
    loc = issues_to_localized_errors(issues, settings)
    return EvaluationReport(
        evaluation_status=evaluation_status,
        comparison_type="audio_vs_reference_audio",
        model_audio_path=model_audio_path,
        learner_audio_path=learner_audio_path,
        asr_model_text=asr_model_text,
        global_scores=global_scores,
        domain_scores=domains,
        localized_errors=loc,
        priority_issues=build_priority_issues(loc),
        feedbacks=build_feedbacks(loc),
        strengths=build_strengths(domains),
        final_summary=build_final_summary(global_scores, evaluation_status),
        expected_text=expected_text,
        asr_text=asr_text,
        warnings=warnings,
        alignment_artifacts=dict(alignment_artifacts or {}),
        audio_quality=dict(audio_quality or {}),
        raw_metrics=dict(raw_metrics or {}),
        confidence_by_domain=dict(confidence_by_domain or {}),
    )


def non_evaluable_report(
    *,
    reason: str,
    expected_text: str,
    asr_text: str = "",
    warnings: Optional[List[str]] = None,
    comparison_type: Literal["audio_vs_reference_audio"] = "audio_vs_reference_audio",
    model_audio_path: str = "",
    learner_audio_path: str = "",
    asr_model_text: str = "",
    alignment_artifacts: Optional[dict[str, str]] = None,
    audio_quality: Optional[dict[str, Any]] = None,
    raw_metrics: Optional[dict[str, Any]] = None,
    confidence_by_domain: Optional[dict[str, Any]] = None,
) -> EvaluationReport:
    ws = list(warnings or [])
    ws.append(reason)

    def _zero_block(mx: float) -> DomainScoreBlock:
        return DomainScoreBlock(
            points_obtained=0.0,
            points_max=mx,
            percent_similarity=0.0,
            percent_delta=100.0,
            interpretation="très éloigné",
        )

    domains = DomainScores(
        segmental_precision=_zero_block(30),
        syllabic_structure_and_articulation=_zero_block(15),
        lexical_stress=_zero_block(20),
        sentence_prosody=_zero_block(20),
        fluency_and_voice_control=_zero_block(10),
        phonological_intelligibility=_zero_block(5),
    )
    return EvaluationReport(
        evaluation_status="non_evaluable",
        comparison_type=comparison_type,
        model_audio_path=model_audio_path,
        learner_audio_path=learner_audio_path,
        asr_model_text=asr_model_text,
        global_scores=GlobalScores(
            score_total_points=0.0,
            percent_similarity=0.0,
            percent_delta=100.0,
            interpretation="très éloigné",
        ),
        domain_scores=domains,
        warnings=ws,
        expected_text=expected_text,
        asr_text=asr_text,
        alignment_artifacts=dict(alignment_artifacts or {}),
        audio_quality=dict(audio_quality or {}),
        raw_metrics=dict(raw_metrics or {}),
        confidence_by_domain=dict(confidence_by_domain or {}),
        final_summary=f"Non évaluable: {reason}",
    )
