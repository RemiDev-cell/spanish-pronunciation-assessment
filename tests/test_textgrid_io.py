from pathlib import Path

from src.textgrid_io import parse_textgrid, pick_phone_tier, pick_word_tier


def test_parse_minimal_textgrid():
    p = Path(__file__).parent / "fixtures" / "minimal_word.TextGrid"
    tiers = parse_textgrid(p)
    w = pick_word_tier(tiers)
    ph = pick_phone_tier(tiers)
    assert w is not None and len(w.intervals) == 1
    assert ph is not None and len(ph.intervals) == 4
