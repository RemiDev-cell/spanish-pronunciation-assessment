"""Normalize Spanish expected text and derive syllables + lexical stress (silabeador)."""

from __future__ import annotations

import re
import unicodedata

import silabeador

from src.models import WordExpectation


def normalize_expected_text(raw: str) -> str:
    """
    Normalize for alignment + ASR comparison.
    HEURISTIC: collapse whitespace, strip outer punctuation, Unicode NFC.
    Keeps Spanish accents and internal apostrophes.
    """
    t = unicodedata.normalize("NFC", raw)
    t = t.replace("\u00a0", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def tokenize_words(normalized: str) -> list[str]:
    """Split on whitespace; strip leading/trailing punctuation per token."""
    words: list[str] = []
    for tok in normalized.split():
        w = re.sub(r"^[^\wáéíóúüñÁÉÍÓÚÜÑ]+", "", tok, flags=re.UNICODE)
        w = re.sub(r"[^\wáéíóúüñÁÉÍÓÚÜÑ]+$", "", w, flags=re.UNICODE)
        if w:
            words.append(w)
    return words


def _tonica_to_index(tonica: int, n_syllables: int) -> int:
    """
    silabeador.tonica returns 0-based from start OR negative from end (-1 last).
    """
    if n_syllables <= 0:
        return 0
    if tonica >= 0:
        return max(0, min(tonica, n_syllables - 1))
    return max(0, min(n_syllables + tonica, n_syllables - 1))


def build_word_expectations(normalized: str) -> list[WordExpectation]:
    """For each token, syllabify and recover stressed syllable index from orthography."""
    expectations: list[WordExpectation] = []
    cursor = 0
    for w in tokenize_words(normalized):
        idx = normalized.find(w, cursor)
        if idx < 0:
            idx = cursor
        cursor = idx + len(w)

        syllables = silabeador.syllabify(w)
        if not syllables:
            syllables = [w]
        try:
            stress_raw = silabeador.tonica(w)
        except Exception:
            stress_raw = silabeador.stressed_s(syllables)
        stress_idx = _tonica_to_index(int(stress_raw), len(syllables))

        expectations.append(
            WordExpectation(
                surface=w,
                syllables=syllables,
                stressed_syllable_index=stress_idx,
                char_start_in_normalized=idx,
                char_end_in_normalized=idx + len(w),
            )
        )
    return expectations


def words_to_mfa_transcript_line(words: list[str]) -> str:
    """Single line transcript for MFA .txt (space-separated surface tokens)."""
    return " ".join(words)


def expected_words_surfaces(expectations: list[WordExpectation]) -> list[str]:
    return [e.surface for e in expectations]
