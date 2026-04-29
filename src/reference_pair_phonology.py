"""HEURISTIC phonological deltas: learner vs model reference, same scripted text."""

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


def _collect_vowel_formant_issues(
    feat_model: FeatureBundle,
    feat_learner: FeatureBundle,
    settings: Settings,
) -> list[PhonologyIssue]:
    issues: list[PhonologyIssue] = []
    n = min(len(feat_model.vowel_formants), len(feat_learner.vowel_formants))
    for i in range(n):
        m = feat_model.vowel_formants[i]
        learner_item = feat_learner.vowel_formants[i]
        f1_m, f1_l = m.get("f1_hz"), learner_item.get("f1_hz")
        f2_m, f2_l = m.get("f2_hz"), learner_item.get("f2_hz")
        if f1_m is None or f1_l is None or f2_m is None or f2_l is None:
            continue

        d_f1 = float(f1_l) - float(f1_m)
        d_f2 = float(f2_l) - float(f2_m)
        dist = math.sqrt(d_f1 * d_f1 + d_f2 * d_f2)
        if (
            abs(d_f1) < settings.vowel_formant_f1_delta_threshold_hz
            and abs(d_f2) < settings.vowel_formant_f2_delta_threshold_hz
            and dist < settings.vowel_formant_distance_threshold_hz
        ):
            continue

        word = str(m.get("word") or "")
        phone = str(m.get("phone") or "")
        issues.append(
            PhonologyIssue(
                error_type=LocalizedErrorType.VOYELLE_MAL_REALISEE,
                target_unit=word or phone or "voyelle",
                precise_location=(
                    f"vowel_formant_index_{i}|word_index_{m.get('word_index')}|phone_{phone}"
                ),
                confidence=0.5,
                observation=(
                    "Écart formantique F1/F2 notable par rapport à la référence du même locuteur "
                    f"(ΔF1≈{d_f1:.0f} Hz, ΔF2≈{d_f2:.0f} Hz, distance≈{dist:.0f} Hz)."
                ),
                observed=f"F1={float(f1_l):.0f} Hz, F2={float(f2_l):.0f} Hz",
                expected=f"F1={float(f1_m):.0f} Hz, F2={float(f2_m):.0f} Hz",
                perceptual_effect="La qualité vocalique peut s'éloigner de la réalisation de référence.",
                correction="Reprendre la voyelle ciblée en cherchant la même ouverture et le même arrondissement que la référence.",
                priority="moyenne",
                score_penalty_hint=min(0.35, 0.12 + dist / 1800.0),
            )
        )
    return issues


def collect_reference_pair_issues(
    alignment_model: AlignmentResult,
    alignment_learner: AlignmentResult,
    feat_model: FeatureBundle,
    feat_learner: FeatureBundle,
    expectations: list[WordExpectation],
    settings: Settings,
) -> list[PhonologyIssue]:
    """
    Compare learner features to the model reference on aligned word indices.
    HEURISTIC: assumes same script; uses MFA word order + text expectations.
    """
    issues: list[PhonologyIssue] = []
    n_model = len(alignment_model.words)
    n_learner = len(alignment_learner.words)
    n_exp = len(expectations)
    n = min(n_model, n_learner, n_exp)

    if n_model != n_learner or n_model != n_exp:
        issues.append(
            PhonologyIssue(
                error_type=LocalizedErrorType.SUBSTITUTION_SEGMENTALE,
                target_unit="phrase",
                precise_location="comptage_mots_model_vs_learner",
                confidence=0.6,
                observation=(
                    f"Nombre de mots alignés diffère: modèle={n_model}, apprenant={n_learner}, "
                    f"attendu (texte)={n_exp}. Comparaison limitée aux {n} premiers mots."
                ),
                observed=f"model={n_model}, learner={n_learner}",
                expected=f"{n_exp} mots (script partagé)",
                perceptual_effect="Comparaison temporelle moins fiable sur la fin de phrase.",
                correction="Réenregistrer en suivant le même script, articulation claire des mots.",
                priority="haute",
                score_penalty_hint=0.35,
            )
        )

    issues.extend(_collect_vowel_formant_issues(feat_model, feat_learner, settings))

    for i in range(n):
        exp = expectations[i]
        d_model = alignment_model.words[i].duration
        d_learner = alignment_learner.words[i].duration
        if d_model > 1e-4:
            ratio = d_learner / d_model
        else:
            ratio = 1.0
        if ratio > settings.word_duration_ratio_high or ratio < settings.word_duration_ratio_low:
            issues.append(
                PhonologyIssue(
                    error_type=LocalizedErrorType.DEBIT_INADAPTE,
                    target_unit=exp.surface,
                    precise_location=f"mot_index_{i}|duree_mot",
                    confidence=0.5,
                    observation=(
                        f"Durée du mot «{exp.surface}» très différente de la référence "
                        f"(ratio learner/model ≈ {ratio:.2f}). HEURISTIC."
                    ),
                    observed=f"{d_learner*1000:.0f} ms (apprenant)",
                    expected=f"{d_model*1000:.0f} ms (modèle)",
                    perceptual_effect="Écart de tempo local vs locuteur de référence.",
                    correction="Rapprocher le débit segmentaire du modèle sans caricature.",
                    priority="moyenne",
                    score_penalty_hint=min(
                        settings.word_duration_penalty_cap,
                        abs(math.log(ratio + 1e-6)) * settings.word_duration_penalty_scale,
                    ),
                )
            )

        pr = _get_prom_payload(feat_model, i, exp)
        pt = _get_prom_payload(feat_learner, i, exp)
        if not pr or not pt:
            continue
        syl_model: list[dict] = list(pr.get("syllables") or [])
        syl_learner: list[dict] = list(pt.get("syllables") or [])
        if not syl_model or not syl_learner or len(syl_model) != len(syl_learner):
            continue
        exp_idx = int(exp.stressed_syllable_index)
        exp_idx = max(0, min(exp_idx, len(syl_learner) - 1))

        z_model = [float(s.get("prominence_z", 0.0)) for s in syl_model]
        z_learner = [float(s.get("prominence_z", 0.0)) for s in syl_learner]
        arg_model = max(range(len(z_model)), key=lambda j: z_model[j])
        arg_learner = max(range(len(z_learner)), key=lambda j: z_learner[j])

        if arg_model == exp_idx and arg_learner != exp_idx:
            issues.append(
                PhonologyIssue(
                    error_type=LocalizedErrorType.ACCENT_TONIQUE_MAL_PLACE,
                    target_unit=exp.surface,
                    precise_location=f"model_stress_ok|learner_argmax_{arg_learner}",
                    confidence=0.55,
                    observation=(
                        f"Sur «{exp.surface}», la référence marque l'accent attendu (syllabe {exp_idx}), "
                        f"mais l'apprenant met l'acuité acoustique max sur {arg_learner}."
                    ),
                    observed=f"argmax_prominence_learner={arg_learner}",
                    expected=f"comme référence: argmax={arg_model} (syllabe tonique {exp_idx})",
                    perceptual_effect="Écart de stress lexical vs référence pédagogique.",
                    correction=(
                        "Rejouer le mot en imitant le relief de la syllabe tonique du modèle "
                        "(durée + intensité + mélodie)."
                    ),
                    priority="haute",
                    score_penalty_hint=0.42,
                )
            )
        elif z_model[exp_idx] - z_learner[exp_idx] > settings.stress_prominence_delta_threshold:
            issues.append(
                PhonologyIssue(
                    error_type=LocalizedErrorType.SYLLABE_TONIQUE_PAS_ASSEZ_SAILLANTE,
                    target_unit=exp.surface,
                    precise_location=f"syllabe_{exp_idx}|vs_reference",
                    confidence=0.52,
                    observation=(
                        f"La syllabe tonique ({exp_idx}) de «{exp.surface}» est moins saillante "
                        f"que dans la référence (Δz≈{z_model[exp_idx]-z_learner[exp_idx]:.2f}). HEURISTIC."
                    ),
                    observed=f"z_tonique_learner≈{z_learner[exp_idx]:.2f}",
                    expected=f"z_tonique_model≈{z_model[exp_idx]:.2f}",
                    perceptual_effect="Moins de contraste interne au mot vs modèle.",
                    correction="Renforcer localement la syllabe tonique comme dans l'enregistrement de référence.",
                    priority="moyenne",
                    score_penalty_hint=0.28,
                )
            )

    # Pauses: compare same boundary index between words
    pa_model = feat_model.pause_durations
    pa_learner = feat_learner.pause_durations
    for j in range(min(len(pa_model), len(pa_learner))):
        diff = abs(pa_learner[j] - pa_model[j])
        if (
            diff > settings.pause_delta_threshold_sec
            and max(pa_model[j], pa_learner[j]) > settings.pause_min_context_sec
        ):
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
                    observed=f"{pa_learner[j]:.2f}s (apprenant)",
                    expected=f"{pa_model[j]:.2f}s (référence)",
                    perceptual_effect="Rythme différent du modèle au même point du script.",
                    correction="Rejouer en calquant les coupures de respiration du locuteur de référence.",
                    priority="moyenne",
                    score_penalty_hint=0.22,
                )
            )

    # Global tempo
    r_wpm = 1.0
    if feat_model.speech_rate_wpm > 1e-3:
        r_wpm = feat_learner.speech_rate_wpm / feat_model.speech_rate_wpm
    if r_wpm < settings.speech_rate_ratio_low or r_wpm > settings.speech_rate_ratio_high:
        issues.append(
            PhonologyIssue(
                error_type=LocalizedErrorType.DEBIT_INADAPTE,
                target_unit="phrase",
                precise_location="speech_rate_vs_reference",
                confidence=0.48,
                observation=(
                    f"Débit global (learner/model) ≈ {r_wpm:.2f} — HEURISTIC basé sur mots/min alignés."
                ),
                observed=f"{feat_learner.speech_rate_wpm:.0f} mpm",
                expected=f"{feat_model.speech_rate_wpm:.0f} mpm (référence)",
                perceptual_effect="Tempo global éloigné du modèle.",
                correction="Ajuster le tempo d'ensemble sur la même phrase de référence.",
                priority="moyenne",
                score_penalty_hint=0.25,
            )
        )

    # Same-speaker mode: direct F0-Hz variability deltas are interpretable.
    if (
        feat_model.f0_std_hz is not None
        and feat_learner.f0_std_hz is not None
        and feat_model.f0_std_hz > settings.f0_std_min_hz_for_intonation
        and feat_learner.f0_std_hz < settings.f0_std_ratio_low * feat_model.f0_std_hz
    ):
        issues.append(
            PhonologyIssue(
                error_type=LocalizedErrorType.INTONATION_NON_CONFORME,
                target_unit="phrase",
                precise_location="f0_std_hz_vs_reference",
                confidence=0.42,
                observation=(
                    "Variabilité F0 de l'apprenant nettement plus faible que la référence "
                    f"(σ_learner≈{feat_learner.f0_std_hz:.1f} Hz vs "
                    f"σ_model≈{feat_model.f0_std_hz:.1f} Hz)."
                ),
                observed=f"σ_F0 learner={feat_learner.f0_std_hz:.1f} Hz",
                expected=f"σ_F0 model={feat_model.f0_std_hz:.1f} Hz",
                perceptual_effect="Moins de relief mélodique que le locuteur de référence.",
                correction="Introduire des variations de hauteur plus proches du modèle (sans copier la voix).",
                priority="basse",
                score_penalty_hint=0.18,
            )
        )

    issues.sort(key=lambda x: x.score_penalty_hint * x.confidence, reverse=True)
    return issues[:25]
