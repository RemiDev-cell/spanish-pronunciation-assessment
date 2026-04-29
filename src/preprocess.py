"""Audio ingest: decode to mono 16 kHz WAV + basic quality gates (HEURISTIC)."""

from __future__ import annotations

import shutil
from typing import Optional
import subprocess
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from src.config import Settings
from src.models import AudioQualityReport


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def convert_to_mono_wav_16k(
    input_path: Path,
    output_wav: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    """Use ffmpeg to produce mono 16 kHz PCM WAV."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(output_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64))) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def _hf_energy_ratio(x: np.ndarray, sr: int) -> float:
    """
    HEURISTIC noise proxy: share of energy above ~4 kHz.
    Not a true SNR — only flags obviously harsh broadband noise.
    """
    if x.size == 0 or sr <= 0:
        return 0.0
    X = np.abs(np.fft.rfft(x.astype(np.float64)))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
    total = float(np.sum(X**2) + 1e-12)
    hf = float(np.sum(X[freqs > 4000.0] ** 2))
    return hf / total


def _clipping_ratio(x: np.ndarray, *, threshold: float = 0.98) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x.astype(np.float64)) >= threshold))


def _silence_ratio(x: np.ndarray, *, threshold: float) -> float:
    if x.size == 0:
        return 1.0
    return float(np.mean(np.abs(x.astype(np.float64)) <= threshold))


def analyze_audio_quality(wav_path: Path, settings: Settings) -> AudioQualityReport:
    data, sr = sf.read(str(wav_path), always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    dur = float(len(data) / sr) if sr else 0.0
    rms = _rms_db(data)
    hf = _hf_energy_ratio(data, int(sr))
    clipping = _clipping_ratio(data)
    silence = _silence_ratio(data, threshold=settings.silence_amplitude_threshold)

    reasons: list[str] = []
    if dur < settings.min_duration_sec:
        reasons.append(f"audio_too_short:{dur:.2f}s")
    if rms < settings.min_rms_db:
        reasons.append(f"low_rms:{rms:.1f}dB")
    if hf > settings.max_hf_energy_ratio:
        reasons.append(f"high_frequency_noise_proxy:{hf:.2f}")
    if clipping > settings.max_clipping_ratio:
        reasons.append(f"clipping_ratio:{clipping:.3f}")
    if silence > settings.max_silence_ratio:
        reasons.append(f"silence_ratio:{silence:.3f}")

    ok = len(reasons) == 0
    return AudioQualityReport(
        duration_sec=dur,
        rms_db=rms,
        hf_energy_ratio=hf,
        clipping_ratio=clipping,
        silence_ratio=silence,
        is_evaluable=ok,
        reason=None if ok else ";".join(reasons),
    )


def ensure_mono_wav(
    audio_path: Path,
    work_dir: Path,
    settings: Settings,
) -> tuple[Path, AudioQualityReport]:
    """
    Returns path to normalized wav in work_dir and quality report.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    out_wav = work_dir / f"{stem}_mono16k.wav"

    ffmpeg_bin = _which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("missing_dependency:ffmpeg")

    suffix = audio_path.suffix.lower()
    if suffix == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wf:
                ch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            if ch == 1 and sr == 16000 and sw == 2:
                import shutil as sh

                sh.copy2(audio_path, out_wav)
            else:
                convert_to_mono_wav_16k(audio_path, out_wav, ffmpeg_bin=ffmpeg_bin)
        except Exception:
            convert_to_mono_wav_16k(audio_path, out_wav, ffmpeg_bin=ffmpeg_bin)
    else:
        convert_to_mono_wav_16k(audio_path, out_wav, ffmpeg_bin=ffmpeg_bin)

    report = analyze_audio_quality(out_wav, settings)
    return out_wav, report
