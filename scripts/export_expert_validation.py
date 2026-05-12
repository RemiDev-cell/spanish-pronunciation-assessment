"""Create a CSV template for ELE expert validation from machine report JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional


FIELDNAMES = [
    "sample_id",
    "model_audio_path",
    "learner_audio_path",
    "expected_text",
    "machine_global_score",
    "machine_segmental_percent",
    "machine_errors_json",
    "teacher_global_score_0_100",
    "teacher_segmental_errors",
    "teacher_intelligibility_0_5",
    "teacher_priority",
    "teacher_notes",
]


def _sample_id(path: Path) -> str:
    return path.name.removesuffix(".report.json").removesuffix(".json")


def report_to_annotation_row(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    segmental = report.get("domain_scores", {}).get("segmental_precision", {})
    errors = report.get("localized_errors", [])
    return {
        "sample_id": _sample_id(path),
        "model_audio_path": report.get("model_audio_path", ""),
        "learner_audio_path": report.get("learner_audio_path", ""),
        "expected_text": report.get("expected_text", ""),
        "machine_global_score": report.get("global_scores", {}).get("score_total_points", ""),
        "machine_segmental_percent": segmental.get("percent_similarity", ""),
        "machine_errors_json": json.dumps(errors, ensure_ascii=False),
        "teacher_global_score_0_100": "",
        "teacher_segmental_errors": "",
        "teacher_intelligibility_0_5": "",
        "teacher_priority": "",
        "teacher_notes": "",
    }


def write_annotation_csv(report_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for path in report_paths:
            writer.writerow(report_to_annotation_row(path))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="Machine report JSON files.")
    parser.add_argument("--output", required=True, type=Path, help="CSV file to create.")
    args = parser.parse_args(argv)
    write_annotation_csv(args.reports, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
