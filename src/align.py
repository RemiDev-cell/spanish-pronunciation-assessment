"""Montreal Forced Aligner subprocess integration."""

from __future__ import annotations

import shutil
from typing import Optional
import subprocess
import uuid
from pathlib import Path

from src.config import Settings
from src.models import AlignmentResult, PhoneInterval, WordInterval
from src.text_processing import words_to_mfa_transcript_line
from src.textgrid_io import parse_textgrid, pick_phone_tier, pick_word_tier


def _which_mfa(settings: Settings) -> Optional[str]:
    return shutil.which(settings.mfa_binary)


def run_mfa_align(
    wav_path: Path,
    word_surfaces: list[str],
    work_dir: Path,
    settings: Settings,
) -> AlignmentResult:
    """
    Run MFA forced alignment for a single file.

    Uses `mfa align_one` (recommended by MFA docs for single-file use) instead of
    `mfa align` on a one-file corpus, which can be fragile.
    """
    mfa = _which_mfa(settings)
    if mfa is None:
        raise RuntimeError("missing_dependency:mfa")

    job = uuid.uuid4().hex[:12]
    out_dir = work_dir / f"mfa_out_{job}"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_dst = work_dir / f"utt_{job}.txt"
    line = words_to_mfa_transcript_line(word_surfaces)
    txt_dst.write_text(line + "\n", encoding="utf-8")

    temp_dir = work_dir / f"mfa_tmp_{job}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    def _run(beam: int) -> subprocess.CompletedProcess[str]:
        cmd = [
            mfa,
            "align_one",
            str(wav_path),
            str(txt_dst),
            settings.mfa_dictionary,
            settings.mfa_acoustic_model,
            str(out_dir),
            "--clean",
            "--num_jobs",
            "1",
            "--temporary_directory",
            str(temp_dir),
            "--beam",
            str(int(beam)),
        ]
        return subprocess.run(cmd, capture_output=True, text=True)

    proc = _run(settings.mfa_beam)
    if proc.returncode != 0 and settings.mfa_retry_beam > settings.mfa_beam:
        proc = _run(settings.mfa_retry_beam)

    if proc.returncode != 0:
        msg = (proc.stderr.strip() or proc.stdout.strip()).strip()
        raise RuntimeError(
            "mfa_align_failed:"
            f"{proc.returncode}:"
            f"{msg[:2000]}\n"
            "hint: check MFA logs under ~/Documents/MFA (or MFA_ROOT_DIR). "
            f"also check temporary_directory={temp_dir}."
        )

    tg_candidates = sorted(out_dir.glob("*.TextGrid"))
    tg_path = tg_candidates[0] if tg_candidates else None
    if tg_path is None or not tg_path.exists():
        raise RuntimeError("mfa_missing_textgrid")

    tiers = parse_textgrid(tg_path)
    wtier = pick_word_tier(tiers)
    ptier = pick_phone_tier(tiers)
    if wtier is None:
        raise RuntimeError("mfa_missing_words_tier")
    if ptier is None:
        raise RuntimeError("mfa_missing_phones_tier")

    words: list[WordInterval] = []
    for wi in wtier.intervals:
        phones = []
        for pi in ptier.intervals:
            if pi.end <= wi.start + 1e-4:
                continue
            if pi.start >= wi.end - 1e-4:
                continue
            phones.append(PhoneInterval(label=pi.text.strip(), start=pi.start, end=pi.end))
        words.append(
            WordInterval(
                label=wi.text.strip(),
                start=wi.start,
                end=wi.end,
                phones=phones,
            )
        )

    return AlignmentResult(words=words, textgrid_path=str(tg_path))
