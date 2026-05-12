"""Aggregate GOP debug/report JSON files and estimate initial calibration thresholds."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Optional


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gop_payload(data: dict[str, Any]) -> dict[str, Any]:
    return dict(data.get("raw_metrics", {}).get("gop_speaker_adapted") or data)


def _sample_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".gop.json", ".report.json", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def load_manifest_labels(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    manifest = _load_json(path)
    labels: dict[str, dict[str, Any]] = {}
    for sample in manifest.get("samples", []):
        if "id" in sample:
            labels[str(sample["id"])] = dict(sample)
        for key in ("report_path", "gop_debug_path"):
            if sample.get(key):
                labels[_sample_id_from_path(Path(str(sample[key])))] = dict(sample)
    return labels


def iter_gop_rows(paths: Iterable[Path], manifest_labels: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        sample_id = _sample_id_from_path(path)
        label = manifest_labels.get(sample_id, {})
        payload = _gop_payload(_load_json(path))
        for idx, item in enumerate(payload.get("phoneme_scores") or []):
            gop = item.get("gop")
            if gop is None:
                continue
            phone = str(item.get("phoneme") or "")
            target_phonemes = {str(p) for p in label.get("target_phonemes", [])}
            if target_phonemes and phone not in target_phonemes:
                expected = "untargeted"
            else:
                expected = str(label.get("expected_outcome") or "unknown")
            rows.append(
                {
                    "sample_id": sample_id,
                    "path": str(path),
                    "phoneme_index": idx,
                    "phoneme": phone,
                    "gop": float(gop),
                    "gop_normalized": item.get("gop_normalized"),
                    "status": item.get("status"),
                    "best_competing": item.get("best_competing"),
                    "duration_sec": float(item.get("t_end", 0.0) or 0.0)
                    - float(item.get("t_start", 0.0) or 0.0),
                    "expected_outcome": expected,
                    "expected_error_type": label.get("expected_error_type"),
                }
            )
    return rows


def _quantile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = Counter(str(r.get("status")) for r in rows)
    by_expected = Counter(str(r.get("expected_outcome")) for r in rows)
    by_phone: dict[str, dict[str, Any]] = {}
    phone_groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        phone_groups[str(row["phoneme"])].append(float(row["gop"]))
    for phone, values in sorted(phone_groups.items()):
        by_phone[phone] = {
            "count": len(values),
            "median_gop": median(values),
            "p10_gop": _quantile(values, 0.10),
            "p90_gop": _quantile(values, 0.90),
        }

    correct = [float(r["gop"]) for r in rows if r.get("expected_outcome") == "correct"]
    errors = [float(r["gop"]) for r in rows if r.get("expected_outcome") == "error"]
    suggested: dict[str, Any] = {
        "warning_threshold": None,
        "error_threshold": None,
        "method": "requires expected_outcome=correct and expected_outcome=error labels",
    }
    if correct and errors:
        correct_p10 = _quantile(correct, 0.10)
        error_p90 = _quantile(errors, 0.90)
        suggested = {
            "warning_threshold": correct_p10,
            "error_threshold": error_p90,
            "method": "warning=p10(correct GOP), error=p90(error GOP)",
            "correct_count": len(correct),
            "error_count": len(errors),
        }

    return {
        "count": len(rows),
        "by_status": dict(by_status),
        "by_expected_outcome": dict(by_expected),
        "by_phoneme": by_phone,
        "suggested_thresholds": suggested,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="GOP debug JSON or full report JSON files.")
    parser.add_argument("--manifest", type=Path, default=None, help="Controlled validation manifest JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Write summary JSON to this path.")
    args = parser.parse_args(argv)

    rows = iter_gop_rows(args.inputs, load_manifest_labels(args.manifest))
    summary = summarize_rows(rows)
    summary["rows"] = rows
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
