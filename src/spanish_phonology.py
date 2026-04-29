"""Spanish phonological heuristics from alignment + acoustic proxies."""

from __future__ import annotations

import math
import re
from statistics import median

from src.config import Settings
from src.models import (
    AlignmentResult,
    FeatureBundle,
    LocalizedErrorType,
    PhonologyIssue,
    WordExpectation,
)


def _is_vowel_phone(label: str) -> bool:
    """HEURISTIC MFA Spanish phone → vowel class."""
    lab = re.sub(r"[ˈˌ]", "", label.strip().lower())
    if not lab:
        return False
    ch = lab[0]
    return ch in "aeiouáéíóúɑɛɪɔʊəɨʉ"


def _phone_duration_issues(
    alignment: AlignmentResult, feat: FeatureBundle
) -> list[PhonologyIssue]:
    issues: list[PhonologyIssue] = []
    mu, sd = feat.global_phone_duration_mean, feat.global_phone_duration_std
    for wi, w in enumerate(alignment.words):
        for ph in w.phones:
            d = ph.duration
            z = abs(d - mu) / (sd + 1e-9)
            if z < 2.0:
                continue
            etype = (
                LocalizedErrorType.VOYELLE_MAL_REALISEE
                if _is_vowel_phone(ph.label)
                else LocalizedErrorType.CONSONNE_MAL_REALISEE
            )
            issues.append(
                PhonologyIssue(
                    error_type=etype,
                    target_unit=w.label,
                    precise_location=f"phone[{ph.label}]@{ph.start:.2f}-{ph.end:.2f}s",
                    confidence=min(0.95, 0.45 + 0.12 * z),
                    observation=(
                        f"Durée segmentale atypique (z≈{z:.1f}) pour le phone «{ph.label}» "
                        f"dans «{w.label}»."
                    ),
                    observed=f"{d*1000:.0f} ms",
                    expected=f"autour de {mu*1000:.0f} ms (écart-type global)",
                    perceptual_effect="Segment parfois étiré ou élidé; intelligibilité locale réduite.",
                    correction=(
                        "Cibler l'articulation nette de ce son sans allonger inutilement la voyelle "
                        "ou affaiblir la consonne."
                    ),
                    priority="moyenne",
                    score_penalty_hint=min(0.35, 0.08 + 0.04 * z),
                )
            )
    return issues


def _stress_issues(feat: FeatureBundle, settings: Settings) -> list[PhonologyIssue]:
    issues: list[PhonologyIssue] = []
    for key, payload in feat.word_prominence_z.items():
        exp_idx = int(payload["expected_stress_index"])
        syls: list[dict] = list(payload["syllables"])
        if not syls:
            continue
        z_vals = [float(s.get("prominence_z", 0.0)) for s in syls]
        argmax = max(range(len(z_vals)), key=lambda i: z_vals[i])
        surface = str(payload.get("surface", ""))
        if argmax != exp_idx:
            issues.append(
                PhonologyIssue(
                    error_type=LocalizedErrorType.ACCENT_TONIQUE_MAL_PLACE,
                    target_unit=surface,
                    precise_location=f"mot:{surface}|syllabe_obs:{argmax}|syllabe_attendue:{exp_idx}",
                    confidence=0.55,
                    observation=(
                        f"HEURISTIC: la syllabe la plus saillante acoustiquement semble être "
                        f"l'index {argmax}, alors que l'accent lexical attendu est {exp_idx}."
                    ),
                    observed=f"prominence max sur syllabe {argmax}",
                    expected=f"accent tonique sur syllabe {exp_idx}",
                    perceptual_effect="Risque de mot perçu comme 'mal accentué' ou ambigu.",
                    correction=(
                        "Renforcer durée, intensité et mouvement mélodique sur la syllabe tonique "
                        "attendue, sans déplacer le stress sur une syllabe voisine."
                    ),
                    priority="haute",
                    score_penalty_hint=0.45,
                )
            )
        else:
            z_exp = z_vals[exp_idx]
            z_max = max(z_vals)
            if z_max - z_exp < settings.stress_prominence_z_expected:
                issues.append(
                    PhonologyIssue(
                        error_type=LocalizedErrorType.SYLLABE_TONIQUE_PAS_ASSEZ_SAILLANTE,
                        target_unit=surface,
                        precise_location=f"mot:{surface}|syllabe_tonique:{exp_idx}",
                        confidence=0.5,
                        observation=(
                            "La syllabe tonique attendue n'est pas nettement plus prominente "
                            "que les autres (proxy durée/F0/intensité)."
                        ),
                        observed=f"écart de z-scores internes ≈{z_max - z_exp:.2f}",
                        expected="mise en relief claire de la syllabe tonique",
                        perceptual_effect="Stress lexical peu audible; intelligibilité légèrement réduite.",
                        correction=(
                            "Allonger légèrement la voyelle tonique, augmenter l'énergie (intensité) "
                            "et marquer une courbe de hauteur plus nette sur cette syllabe."
                        ),
                        priority="moyenne",
                        score_penalty_hint=0.25,
                    )
                )
    return issues


def _pause_issues(feat: FeatureBundle, settings: Settings) -> list[PhonologyIssue]:
    issues: list[PhonologyIssue] = []
    if len(feat.pause_durations) < settings.min_words_for_pause_stats:
        return issues
    med = float(median(feat.pause_durations))
    if med <= 1e-3:
        med = 0.05
    for j, p in enumerate(feat.pause_durations):
        if p > max(0.85, settings.pause_outlier_multiplier * med):
            issues.append(
                PhonologyIssue(
                    error_type=LocalizedErrorType.PAUSE_MAL_PLACEE,
                    target_unit=f"jonction_{j}",
                    precise_location=f"pause_apres_mot_index_{j}",
                    confidence=0.45,
                    observation=f"Pause inter-mot longue ({p:.2f}s) vs médiane ({med:.2f}s).",
                    observed=f"{p:.2f}s",
                    expected="pauses plus régulières, sauf ponctuation forte",
                    perceptual_effect="Rythme haché; effet de hésitation ou frontière mal placée.",
                    correction=(
                        "Fluidifier entre les mots en évitant les coupures longues sans raison "
                        "prosodique; marquer plutôt les frontières syntaxiques attendues."
                    ),
                    priority="moyenne",
                    score_penalty_hint=0.2,
                )
            )
    return issues


def _fluency_issues(feat: FeatureBundle) -> list[PhonologyIssue]:
    issues: list[PhonologyIssue] = []
    wpm = feat.speech_rate_wpm
    if wpm <= 0:
        return issues
    if wpm < 85 or wpm > 215:
        issues.append(
            PhonologyIssue(
                error_type=LocalizedErrorType.DEBIT_INADAPTE,
                target_unit="phrase",
                precise_location="global",
                confidence=0.5,
                observation=f"Débit global estimé à ~{wpm:.0f} mots/min (HEURISTIC).",
                observed=f"{wpm:.0f} mpm",
                expected="environ 110–190 mpm pour une lecture calme",
                perceptual_effect="Texte perçu trop lent ou précipité.",
                correction=(
                    "Ajuster le tempo: respirer aux frontières syntaxiques et maintenir un "
                    "rythme régulier sans précipitation."
                ),
                priority="moyenne",
                score_penalty_hint=0.25,
            )
        )
    n_long = sum(1 for p in feat.pause_durations if p > 0.75)
    if n_long >= 3:
        issues.append(
            PhonologyIssue(
                error_type=LocalizedErrorType.DISFLUENCE,
                target_unit="phrase",
                precise_location="pauses_multiples",
                confidence=0.4,
                observation="Plusieurs pauses longues successives (proxy de disfluence).",
                observed=f"{n_long} pauses > 0.75s",
                expected="enchaînement plus continu",
                perceptual_effect="Impression d'hésitation ou de reprise.",
                correction="Repérer les mots difficiles et les stabiliser par répétition segmentée.",
                priority="basse",
                score_penalty_hint=0.2,
            )
        )
    return issues


def _intonation_issue(feat: FeatureBundle) -> list[PhonologyIssue]:
    if feat.f0_std_hz is None or math.isnan(feat.f0_std_hz):
        return []
    if feat.f0_std_hz < 7.5 and (feat.mean_f0_hz or 0) > 50:
        return [
            PhonologyIssue(
                error_type=LocalizedErrorType.INTONATION_NON_CONFORME,
                target_unit="phrase",
                precise_location="contour_global",
                confidence=0.35,
                observation=(
                    f"Variabilité F0 faible (σ≈{feat.f0_std_hz:.1f} Hz) — HEURISTIC monotonie."
                ),
                observed=f"σ_F0={feat.f0_std_hz:.1f} Hz",
                expected="contours plus marqués aux frontières de groupe prosodique",
                perceptual_effect="Lecture perçue comme plate ou peu guidée pour l'auditeur.",
                correction=(
                    "Introduire des variations de hauteur légères mais intentionnelles aux "
                    "frontières de syntagmes, sans caricature."
                ),
                priority="basse",
                score_penalty_hint=0.15,
            )
        ]
    return []


def _articulation_blur(feat: FeatureBundle) -> list[PhonologyIssue]:
    """
    HEURISTIC: many phones with moderately high duration variance + low intensity std.
    """
    if feat.mean_intensity_db is None or feat.intensity_std_db is None:
        return []
    if feat.intensity_std_db < 1.8 and feat.global_phone_duration_std / (
        feat.global_phone_duration_mean + 1e-6
    ) > 0.55:
        return [
            PhonologyIssue(
                error_type=LocalizedErrorType.ARTICULATION_FLOUE,
                target_unit="phrase",
                precise_location="global",
                confidence=0.35,
                observation=(
                    "Faible modulation d'intensité combinée à des durées segmentales dispersées "
                    "(proxy d'articulation peu nette)."
                ),
                observed=f"σ_intensité≈{feat.intensity_std_db:.2f} dB",
                expected="attaques de segments plus nettes",
                perceptual_effect="Segments moins distincts; charge cognitive pour l'auditeur.",
                correction=(
                    "Travailler les attaques consonantiques et la voyelle accentuée avec un "
                    "meilleur contrôle d'énergie."
                ),
                priority="moyenne",
                score_penalty_hint=0.2,
            )
        ]
    return []


def collect_phonology_issues(
    alignment: AlignmentResult,
    expectations: list[WordExpectation],
    feat: FeatureBundle,
    settings: Settings,
) -> list[PhonologyIssue]:
    """
    Merge heuristics. Does not attempt true G2P mismatch detection (MVP limitation).
    """
    issues: list[PhonologyIssue] = []
    issues.extend(_phone_duration_issues(alignment, feat))
    issues.extend(_stress_issues(feat, settings))
    issues.extend(_pause_issues(feat, settings))
    issues.extend(_fluency_issues(feat))
    issues.extend(_intonation_issue(feat))
    issues.extend(_articulation_blur(feat))

    # HEURISTIC: word count mismatch between MFA and expected → segmental warning
    if len(alignment.words) != len(expectations):
        issues.append(
            PhonologyIssue(
                error_type=LocalizedErrorType.SUBSTITUTION_SEGMENTALE,
                target_unit="phrase",
                precise_location="comptage_mots",
                confidence=0.55,
                observation=(
                    f"Nombre de mots alignés ({len(alignment.words)}) ≠ nombre attendu "
                    f"({len(expectations)}); transcription forcée possiblement biaisée."
                ),
                observed=f"{len(alignment.words)} mots",
                expected=f"{len(expectations)} mots",
                perceptual_effect="Les scores segmentaux deviennent moins fiables.",
                correction="Vérifier la prononciation des mots omis/ajoutés et la qualité audio.",
                priority="haute",
                score_penalty_hint=0.35,
            )
        )

    # Rank by penalty * confidence. HEURISTIC: cap list so scoring stays stable on long utterances.
    issues.sort(key=lambda x: x.score_penalty_hint * x.confidence, reverse=True)
    return issues[:25]
