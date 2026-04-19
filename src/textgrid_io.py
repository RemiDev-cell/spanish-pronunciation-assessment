"""Minimal Praat TextGrid (long format) parser for MFA outputs."""

from __future__ import annotations

import re
from typing import Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Interval:
    start: float
    end: float
    text: str


@dataclass
class IntervalTier:
    name: str
    intervals: list[Interval]


def parse_textgrid(path: Path) -> dict[str, IntervalTier]:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_textgrid_from_string(raw)


def parse_textgrid_from_string(raw: str) -> dict[str, IntervalTier]:
    """
    Parse long TextGrid text into tiers keyed by tier name.
    HEURISTIC: assumes standard MFA export structure.
    """
    tiers: dict[str, IntervalTier] = {}

    # Split into item blocks
    for m in re.finditer(
        r'item\s*\[\s*\d+\s*\]:\s*'
        r'class\s*=\s*"IntervalTier"\s*'
        r'name\s*=\s*"(?P<name>[^"]*)"\s*'
        r"xmin\s*=\s*(?P<xmin>[0-9eE.+-]+)\s*"
        r"xmax\s*=\s*(?P<xmax>[0-9eE.+-]+)\s*"
        r"intervals:\s*size\s*=\s*(?P<size>\d+)\s*"
        r"(?P<body>.*?)(?=(?:item\s*\[\s*\d+\s*\]:)|\Z)",
        raw,
        flags=re.DOTALL,
    ):
        name = m.group("name")
        body = m.group("body")
        intervals: list[Interval] = []
        for im in re.finditer(
            r"intervals\s*\[\s*\d+\s*\]:\s*"
            r"xmin\s*=\s*([0-9eE.+-]+)\s*"
            r"xmax\s*=\s*([0-9eE.+-]+)\s*"
            r'text\s*=\s*"(.*?)"\s*',
            body,
            flags=re.DOTALL,
        ):
            xmin, xmax, text = float(im.group(1)), float(im.group(2)), im.group(3)
            text = text.replace('""', '"')
            if text.strip() == "":
                continue
            intervals.append(Interval(start=xmin, end=xmax, text=text))
        tiers[name] = IntervalTier(name=name, intervals=intervals)

    return tiers


def pick_phone_tier(tiers: dict[str, IntervalTier]) -> Optional[IntervalTier]:
    for key in ("phones", "phone", "phones_mfa"):
        if key in tiers:
            return tiers[key]
    return None


def pick_word_tier(tiers: dict[str, IntervalTier]) -> Optional[IntervalTier]:
    for key in ("words", "word", "words_mfa"):
        if key in tiers:
            return tiers[key]
    return None
