"""Acoustic / prosodic features via Parselmouth + alignment-derived timings."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any

import numpy as np
import parselmouth as pm

from src.models import AlignmentResult, FeatureBundle, WordExpectation


def _sample_series(
    pitch: pm.Pitch,
    intensity: pm.Intensity,
    t0: float,
    t1: float,
    step: float = 0.01,
) -> tuple[list[float], list[float]]:
    """Sample voiced F0 (Hz) and intensity (dB) in [t0, t1]."""
    f0s: list[float] = []
    dbs: list[float] = []
    if t1 <= t0:
        return f0s, dbs
    t = t0
    while t <= t1:
        f = pitch.get_value_at_time(t)
        if f is not None and not math.isnan(f) and f > 0:
            f0s.append(float(f))
        db = intensity.get_value(t)
        if db is not None and not math.isnan(db):
            dbs.append(float(db))
        t += step
    return f0s, dbs


def _proportional_syllable_boundaries(
    w_start: float, w_end: float, syllables: list[str]
) -> list[tuple[float, float]]:
    """
    HEURISTIC: split word interval proportionally by grapheme length of each syllable.
    Not physiological syllable boundaries — MVP only.
    """
    if not syllables:
        return [(w_start, w_end)]
    lens = np.array([max(1, len(s)) for s in syllables], dtype=np.float64)
    parts = lens / lens.sum()
    bounds: list[tuple[float, float]] = []
    cur = w_start
    for p in parts:
        span = (w_end - w_start) * float(p)
        bounds.append((cur, cur + span))
        cur += span
    bounds[-1] = (bounds[-1][0], w_end)
    return bounds


def extract_features(
    wav_path: str,
    alignment: AlignmentResult,
    expectations: list[WordExpectation],
) -> FeatureBundle:
    snd = pm.Sound(wav_path)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
    intensity = snd.to_intensity(time_step=0.01, minimum_pitch=75.0)

    words = alignment.words
    word_durs = [w.duration for w in words]
    phone_durs: list[float] = []
    for w in words:
        for p in w.phones:
            phone_durs.append(p.duration)

    pause_durs: list[float] = []
    for i in range(len(words) - 1):
        gap = words[i + 1].start - words[i].end
        if gap > 0.02:
            pause_durs.append(gap)

    if words:
        span = max(1e-6, words[-1].end - words[0].start)
        speech_rate_wpm = (len(words) / span) * 60.0
    else:
        speech_rate_wpm = 0.0

    all_f0: list[float] = []
    all_db: list[float] = []
    t0, t1 = 0.0, snd.duration
    f0s, dbs = _sample_series(pitch, intensity, t0, t1)
    all_f0.extend(f0s)
    all_db.extend(dbs)

    mean_f0 = float(mean(all_f0)) if all_f0 else None
    f0_std = float(pstdev(all_f0)) if len(all_f0) > 1 else (0.0 if all_f0 else None)
    mean_int = float(mean(all_db)) if all_db else None
    int_std = float(pstdev(all_db)) if len(all_db) > 1 else (0.0 if all_db else None)

    if phone_durs:
        g_mean = float(mean(phone_durs))
        g_std = float(pstdev(phone_durs)) if len(phone_durs) > 1 else 1e-6
    else:
        g_mean, g_std = 0.0, 1e-6

    prominence: dict[str, dict[str, Any]] = {}

    n_pairs = min(len(words), len(expectations))

    for i in range(n_pairs):
        w = words[i]
        exp = expectations[i]
        key = f"{i}:{exp.surface}"
        syl_bounds = _proportional_syllable_boundaries(w.start, w.end, exp.syllables)
        syl_feats: list[dict[str, Any]] = []
        raw_scores: list[float] = []
        for si, (a, b) in enumerate(syl_bounds):
            f0_loc, db_loc = _sample_series(pitch, intensity, a, b)
            dur = max(0.0, b - a)
            fm = float(mean(f0_loc)) if f0_loc else None
            im = float(mean(db_loc)) if db_loc else None
            raw = dur + 0.004 * (fm or 0.0) + 0.06 * (im or 0.0)
            raw_scores.append(raw)
            syl_feats.append(
                {
                    "start": a,
                    "end": b,
                    "f0_mean": fm,
                    "intensity_mean": im,
                    "duration": dur,
                }
            )
        mu = float(mean(raw_scores)) if raw_scores else 0.0
        sd = float(pstdev(raw_scores)) if len(raw_scores) > 1 else 1e-6
        for si, sf in enumerate(syl_feats):
            z = (raw_scores[si] - mu) / (sd + 1e-6)
            sf["prominence_z"] = float(z)

        prominence[key] = {
            "syllables": syl_feats,
            "expected_stress_index": exp.stressed_syllable_index,
            "surface": exp.surface,
        }

    return FeatureBundle(
        word_durations=word_durs,
        phone_durations=phone_durs,
        pause_durations=pause_durs,
        speech_rate_wpm=speech_rate_wpm,
        mean_f0_hz=mean_f0,
        f0_std_hz=f0_std,
        mean_intensity_db=mean_int,
        intensity_std_db=int_std,
        word_prominence_z=prominence,
        global_phone_duration_mean=g_mean,
        global_phone_duration_std=max(g_std, 1e-6),
        raw_debug={"n_words_aligned": len(words), "n_words_expected": len(expectations)},
    )


def count_voiced_pitch_frames(wav_path: str) -> int:
    """Quick gate: enough voiced samples for prosody."""
    snd = pm.Sound(wav_path)
    pitch = snd.to_pitch(time_step=0.02, pitch_floor=75.0, pitch_ceiling=500.0)
    n = 0
    t = 0.0
    while t <= snd.duration:
        f = pitch.get_value_at_time(t)
        if f is not None and not math.isnan(f) and f > 0:
            n += 1
        t += 0.02
    return n
