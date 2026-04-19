"""Map features + phonology issues to weighted domain scores (MVP heuristics)."""

from __future__ import annotations

from src.models import (
    DomainScoreBlock,
    DomainScores,
    FeatureBundle,
    GlobalScores,
    LocalizedErrorType,
    PhonologyIssue,
)


def _interpret_fr(percent: float) -> str:
    if percent >= 90:
        return "quasi identique à la cible textuelle attendue"
    if percent >= 75:
        return "très proche"
    if percent >= 60:
        return "satisfaisant mais écarts perceptibles"
    if percent >= 40:
        return "fragile"
    return "très éloigné"


def _block(points_obtained: float, points_max: float) -> DomainScoreBlock:
    obtained = max(0.0, min(float(points_max), float(points_obtained)))
    pct = 100.0 * obtained / points_max if points_max else 0.0
    return DomainScoreBlock(
        points_obtained=round(obtained, 2),
        points_max=points_max,
        percent_similarity=round(pct, 2),
        percent_delta=round(100.0 - pct, 2),
        interpretation=_interpret_fr(pct),
    )


def _issue_domain_weights(issue: PhonologyIssue) -> dict[str, float]:
    """Return penalty weights per domain for one issue (HEURISTIC)."""
    t = issue.error_type
    w_seg = 0.0
    w_syl = 0.0
    w_stress = 0.0
    w_pros = 0.0
    w_flu = 0.0
    w_intel = 0.0

    base = issue.score_penalty_hint * (0.6 + 0.4 * issue.confidence)

    if t in (LocalizedErrorType.VOYELLE_MAL_REALISEE, LocalizedErrorType.CONSONNE_MAL_REALISEE):
        w_seg = 2.2 * base
    elif t == LocalizedErrorType.SUBSTITUTION_SEGMENTALE:
        w_seg = 3.0 * base
        w_intel = 1.0 * base
    elif t in (LocalizedErrorType.ACCENT_TONIQUE_MAL_PLACE,):
        w_stress = 4.5 * base
        w_seg += 0.8 * base
    elif t in (LocalizedErrorType.SYLLABE_TONIQUE_PAS_ASSEZ_SAILLANTE,):
        w_stress = 3.0 * base
        w_pros += 0.8 * base
    elif t == LocalizedErrorType.PAUSE_MAL_PLACEE:
        w_pros = 2.4 * base
        w_flu = 1.0 * base
    elif t == LocalizedErrorType.INTONATION_NON_CONFORME:
        w_pros = 2.8 * base
    elif t == LocalizedErrorType.DEBIT_INADAPTE:
        w_flu = 3.0 * base
        w_pros += 0.6 * base
    elif t == LocalizedErrorType.DISFLUENCE:
        w_flu = 2.4 * base
    elif t == LocalizedErrorType.ARTICULATION_FLOUE:
        w_syl = 2.6 * base
        w_seg += 1.2 * base
        w_intel += 0.8 * base
    else:
        w_seg += 1.0 * base

    return {
        "segmental": w_seg,
        "syllabic": w_syl,
        "stress": w_stress,
        "prosody": w_pros,
        "fluency": w_flu,
        "intel": w_intel,
    }


def compute_domain_scores(
    issues: list[PhonologyIssue],
    feat: FeatureBundle,
) -> DomainScores:
    """
    Start from maxima and subtract pooled penalties (MVP).
    Extra soft adjustments from global feature stats (HEURISTIC).
    """
    max_seg, max_syl, max_stress, max_pros, max_flu, max_intel = 30, 15, 20, 20, 10, 5
    pen_seg = pen_syl = pen_stress = pen_pros = pen_flu = pen_intel = 0.0

    for iss in issues:
        w = _issue_domain_weights(iss)
        pen_seg += w["segmental"]
        pen_syl += w["syllabic"]
        pen_stress += w["stress"]
        pen_pros += w["prosody"]
        pen_flu += w["fluency"]
        pen_intel += w["intel"]

    # Global prosody nudge from f0 variability
    if feat.f0_std_hz is not None and feat.f0_std_hz < 10:
        pen_pros += 0.8

    # Cap penalties
    pen_seg = min(pen_seg, max_seg)
    pen_syl = min(pen_syl, max_syl)
    pen_stress = min(pen_stress, max_stress)
    pen_pros = min(pen_pros, max_pros)
    pen_flu = min(pen_flu, max_flu)
    pen_intel = min(pen_intel, max_intel)

    ds = DomainScores(
        segmental_precision=_block(max_seg - pen_seg, max_seg),
        syllabic_structure_and_articulation=_block(max_syl - pen_syl, max_syl),
        lexical_stress=_block(max_stress - pen_stress, max_stress),
        sentence_prosody=_block(max_pros - pen_pros, max_pros),
        fluency_and_voice_control=_block(max_flu - pen_flu, max_flu),
        phonological_intelligibility=_block(max_intel - pen_intel, max_intel),
    )
    return ds


def compute_global_scores(domains: DomainScores) -> GlobalScores:
    total = 0.0
    max_total = 0.0
    for b in (
        domains.segmental_precision,
        domains.syllabic_structure_and_articulation,
        domains.lexical_stress,
        domains.sentence_prosody,
        domains.fluency_and_voice_control,
        domains.phonological_intelligibility,
    ):
        total += b.points_obtained
        max_total += b.points_max
    pct = 100.0 * total / max_total if max_total else 0.0
    return GlobalScores(
        score_total_points=round(total, 2),
        percent_similarity=round(pct, 2),
        percent_delta=round(100.0 - pct, 2),
        interpretation=_interpret_fr(pct),
    )
