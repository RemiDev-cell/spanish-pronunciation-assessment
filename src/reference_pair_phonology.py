"""HEURISTIC phonological deltas: learner (test) vs reference, same scripted text."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from src.config import Settings
from src.models import (
    AlignmentResult,
    FeatureBundle,
    LocalizedErrorType,
    PhonologyIssue,
    WordExpectation,
)


def _word_key(i: int, exp: WordExpectation) -> str:
    return f"{i}:{exp.surface}"


def _get_prom_payload(feat: FeatureBundle, i: int, exp: WordExpectation) -> Optional[Dict[str, Any]]:
    return feat.word_prominence_z.get(_word_key(i, exp))


def collect_reference_pair_issues(
    alignment_ref: AlignmentResult,
    alignment_test: AlignmentResult,
    feat_ref: FeatureBundle,
    feat_test: FeatureBundle,
    expectations: list[WordExpectation],
    settings: Settings,
) -> list[PhonologyIssue]:
    """
    Compare test (learner) features to reference on aligned word indices.
    HEURISTIC: assumes same script; uses MFA word order + text expectations.
    """
    issues: list[PhonologyIssue] = []
    n_ref = len(alignment_ref.words)
    n_te = len(alignment_test.words)
    n_exp = len(expectations)
    n = min(n_ref, n_te, n_exp)

    if n_ref != n_te or n_ref != n_exp:
        issues.append(
            PhonologyIssue(
                error_type=LocalizedErrorType.SUBSTITUTION_SEGMENTALE,
                target_unit="phrase",
                precise_location="comptage_mots_ref_vs_test",
                confidence=0.6,
                observation=(
                    f"Nombre de mots alignés diffère: référence={n_ref}, test={n_te}, "
                    f"attendu (texte)={n_exp}. Comparaison limitée aux {n} premiers mots."
                ),
                observed=f"ref={n_ref}, test={n_te}",
                expected=f"{n_exp} mots (script partagé)",
                perceptual_effect="Comparaison temporelle moins fiable sur la fin de phrase.",
                correction="Réenregistrer en suivant le même script, articulation claire des mots.",
                priority="haute",
                score_penalty_hint=0.35,
            )
        )

    for i in range(n):
        exp = expectations[i]
        d_ref = alignment_ref.words[i].duration
        d_te = alignment_test.words[i].duration
        if d_ref > 1e-4:
            ratio = d_te / d_ref
        else:
            ratio = 1.0
        if ratio > 1.42 or ratio < 0.58:
            issues.append(
                PhonologyIssue(
                    error_type=LocalizedErrorType.DEBIT_INADAPTE,
                    target_unit=exp.surface,
                    precise_location=f"mot_index_{i}|duree_mot",
                    confidence=0.5,
                    observation=(
                        f"Durée du mot «{exp.surface}» très différente de la référence "
                        f"(ratio test/ref ≈ {ratio:.2f}). HEURISTIC."
                    ),
                    observed=f"{d_te*1000:.0f} ms (test)",
                    expected=f"{d_ref*1000:.0f} ms (référence)",
                    perceptual_effect="Écart de tempo local vs locuteur de référence.",
                    correction="Rapprocher le débit segmentaire du modèle sans caricature.",
                    priority="moyenne",
                    score_penalty_hint=min(0.35, abs(math.log(ratio + 1e-6)) * 0.12),
                )
            )

        pr = _get_prom_payload(feat_ref, i, exp)
        pt = _get_prom_payload(feat_test, i, exp)
        if not pr or not pt:
            continue
        syl_ref: list[dict] = list(pr.get("syllables") or [])
        syl_te: list[dict] = list(pt.get("syllables") or [])
        if not syl_ref or not syl_te or len(syl_ref) != len(syl_te):
            continue
        exp_idx = int(exp.stressed_syllable_index)
        exp_idx = max(0, min(exp_idx, len(syl_te) - 1))

        zr = [float(s.get("prominence_z", 0.0)) for s in syl_ref]
        zt = [float(s.get("prominence_z", 0.0)) for s in syl_te]
        arg_r = max(range(len(zr)), key=lambda j: zr[j])
        arg_t = max(range(len(zt)), key=lambda j: zt[j])

        if arg_r == exp_idx and arg_t != exp_idx:
            issues.append(
                PhonologyIssue(
                    error_type=LocalizedErrorType.ACCENT_TONIQUE_MAL_PLACE,
                    target_unit=exp.surface,
                    precise_location=f"ref_stress_ok|test_argmax_{arg_t}",
                    confidence=0.55,
                    observation=(
                        f"Sur «{exp.surface}», la référence marque l'accent attendu (syllabe {exp_idx}), "
                        f"mais le test met l'acuité acoustique max sur {arg_t}."
                    ),
                    observed=f"argmax_prominence_test={arg_t}",
                    expected=f"comme référence: argmax={arg_r} (syllabe tonique {exp_idx})",
                    perceptual_effect="Écart de stress lexical vs référence pédagogique.",
                    correction=(
                        "Rejouer le mot en imitant le relief de la syllabe tonique du modèle "
                        "(durée + intensité + mélodie)."
                    ),
                    priority="haute",
                    score_penalty_hint=0.42,
                )
            )
        elif zr[exp_idx] - zt[exp_idx] > 0.85:
            issues.append(
                PhonologyIssue(
                    error_type=LocalizedErrorType.SYLLABE_TONIQUE_PAS_ASSEZ_SAILLANTE,
                    target_unit=exp.surface,
                    precise_location=f"syllabe_{exp_idx}|vs_reference",
                    confidence=0.52,
                    observation=(
                        f"La syllabe tonique ({exp_idx}) de «{exp.surface}» est moins saillante "
                        f"que dans la référence (Δz≈{zr[exp_idx]-zt[exp_idx]:.2f}). HEURISTIC."
                    ),
                    observed=f"z_tonique_test≈{zt[exp_idx]:.2f}",
                    expected=f"z_tonique_ref≈{zr[exp_idx]:.2f}",
                    perceptual_effect="Moins de contraste interne au mot vs modèle.",
                    correction="Renforcer localement la syllabe tonique comme dans l'enregistrement de référence.",
                    priority="moyenne",
                    score_penalty_hint=0.28,
                )
            )

    # Pauses: compare same boundary index between words
    pa_ref = feat_ref.pause_durations
    pa_te = feat_test.pause_durations
    for j in range(min(len(pa_ref), len(pa_te))):
        diff = abs(pa_te[j] - pa_ref[j])
        if diff > 0.38 and max(pa_ref[j], pa_te[j]) > 0.12:
            issues.append(
                PhonologyIssue(
                    error_type=LocalizedErrorType.PAUSE_MAL_PLACEE,
                    target_unit=f"apres_mot_{j}",
                    precise_location=f"pause_boundary_{j}|vs_reference",
                    confidence=0.45,
                    observation=(
                        f"Pause après le mot d'index {j} diffère nettement de la référence "
                        f"(Δ≈{diff:.2f}s)."
                    ),
                    observed=f"{pa_te[j]:.2f}s (test)",
                    expected=f"{pa_ref[j]:.2f}s (référence)",
                    perceptual_effect="Rythme différent du modèle au même point du script.",
                    correction="Rejouer en calquant les coupures de respiration du locuteur de référence.",
                    priority="moyenne",
                    score_penalty_hint=0.22,
                )
            )

    # Global tempo
    r_wpm = 1.0
    if feat_ref.speech_rate_wpm > 1e-3:
        r_wpm = feat_test.speech_rate_wpm / feat_ref.speech_rate_wpm
    if r_wpm < 0.72 or r_wpm > 1.35:
        issues.append(
            PhonologyIssue(
                error_type=LocalizedErrorType.DEBIT_INADAPTE,
                target_unit="phrase",
                precise_location="speech_rate_vs_reference",
                confidence=0.48,
                observation=(
                    f"Débit global (test/ref) ≈ {r_wpm:.2f} — HEURISTIC basé sur mots/min alignés."
                ),
                observed=f"{feat_test.speech_rate_wpm:.0f} mpm",
                expected=f"{feat_ref.speech_rate_wpm:.0f} mpm (référence)",
                perceptual_effect="Tempo global éloigné du modèle.",
                correction="Ajuster le tempo d'ensemble sur la même phrase de référence.",
                priority="moyenne",
                score_penalty_hint=0.25,
            )
        )

    # Intonation variability vs reference
    if (
        feat_ref.f0_std_hz is not None
        and feat_test.f0_std_hz is not None
        and feat_ref.f0_std_hz > 10
        and feat_test.f0_std_hz < 0.55 * feat_ref.f0_std_hz
    ):
        issues.append(
            PhonologyIssue(
                error_type=LocalizedErrorType.INTONATION_NON_CONFORME,
                target_unit="phrase",
                precise_location="f0_std_vs_reference",
                confidence=0.42,
                observation=(
                    "Variabilité F0 du test nettement plus faible que la référence "
                    f"(σ_test≈{feat_test.f0_std_hz:.1f} Hz vs σ_ref≈{feat_ref.f0_std_hz:.1f} Hz)."
                ),
                observed=f"σ_F0 test={feat_test.f0_std_hz:.1f} Hz",
                expected=f"σ_F0 ref={feat_ref.f0_std_hz:.1f} Hz",
                perceptual_effect="Moins de relief mélodique que le locuteur de référence.",
                correction="Introduire des variations de hauteur plus proches du modèle (sans copier la voix).",
                priority="basse",
                score_penalty_hint=0.18,
            )
        )

    issues.sort(key=lambda x: x.score_penalty_hint * x.confidence, reverse=True)
    return issues[:25]
