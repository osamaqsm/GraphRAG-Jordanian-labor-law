from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.legal_query_planner import LegalQueryPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the current query-planner prompt, schema, and thresholds without calling OpenAI."
    )
    parser.add_argument(
        "--output",
        default="/tmp/query_planner_config_snapshot.json",
        help="JSON output path inside the container.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    planner = LegalQueryPlanner(get_settings())

    document = {
        "snapshot_name": "Stage 7.7-A Query Planner Configuration Snapshot",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "planner_enabled": planner.enabled,
        "planner_model": planner.model,
        "reasoning_effort": planner.reasoning_effort,
        "route_confidence": planner.route_confidence,
        "retrieve_override_confidence": planner.retrieve_override_confidence,
        "instructions": planner._instructions(),
        "strict_response_schema": planner._strict_response_schema(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(document, ensure_ascii=False, indent=2))
    print(f"\nSaved JSON snapshot to: {output_path}")


if __name__ == "__main__":
    main()
