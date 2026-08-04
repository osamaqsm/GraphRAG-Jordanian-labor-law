from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.grounded_answer_generator import GroundedAnswerGenerator
from app.retrieval_contract import RetrievalResultV1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a grounded answer from a saved retrieval.v1 JSON file. "
            "This command never runs retrieval."
        )
    )
    parser.add_argument(
        "retrieval_json",
        type=Path,
        help="Path to a saved retrieval.v1 JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for generation.v1 JSON.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include generation diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(
        args.retrieval_json.read_text(encoding="utf-8-sig")
    )
    retrieval = RetrievalResultV1.model_validate(payload)

    generator = GroundedAnswerGenerator()
    result = generator.generate(
        retrieval,
        include_debug=args.debug,
    )
    rendered = json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
