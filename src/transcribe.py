"""Whisper ASR wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TranscriptionResult:
    text: str
    segments: list[dict[str, Any]]


def transcribe_spanish(wav_path: Path, *, model_name: str = "base") -> TranscriptionResult:
    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(str(wav_path), language="es", task="transcribe", fp16=False)
    text = (result.get("text") or "").strip()
    segments = list(result.get("segments") or [])
    return TranscriptionResult(text=text, segments=segments)
