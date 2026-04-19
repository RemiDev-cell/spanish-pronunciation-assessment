"""Generate json_schema/evaluation_report.schema.json from Pydantic models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import EvaluationReport  # noqa: E402


def main() -> None:
    schema = EvaluationReport.model_json_schema(ref_template="#/$defs/{model}")
    out = ROOT / "json_schema" / "evaluation_report.schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
