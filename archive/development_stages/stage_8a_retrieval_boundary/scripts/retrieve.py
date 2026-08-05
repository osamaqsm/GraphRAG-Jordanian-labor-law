from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.retrieval_pipeline import RetrievalOnlyPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Jordanian Labor Law retrieval process without answer generation."
    )
    parser.add_argument("question")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to save for later generation-only testing.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with RetrievalOnlyPipeline() as pipeline:
        result = pipeline.retrieve(
            args.question,
            include_debug=args.debug,
        )

    payload = result.model_dump(mode="json")
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=None if args.compact else 2,
    )
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Saved retrieval result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
