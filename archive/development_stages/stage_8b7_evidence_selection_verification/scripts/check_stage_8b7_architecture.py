from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the Stage 8-B7 frozen generation architecture."
    )
    parser.add_argument(
        "--generator",
        type=Path,
        default=Path("app/grounded_answer_generator.py"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.generator.read_text(encoding="utf-8").lower()

    forbidden = [
        "retrievalonlypipeline",
        "retrievalservice",
        "import weaviate",
        "graphtraversalservice",
        "embeddings.create",
        "ugh01",
        "g01",
        "المادة 47",
    ]
    for value in forbidden:
        assert value not in source, value

    required = [
        "evidenceselectionplan",
        "_validate_selection_plan",
        "_validate_answer_coverage",
        "selected_article_numbers",
        "excluded_articles",
        "issue_support",
        "_canonicalize_citations",
    ]
    for value in required:
        assert value.lower() in source, value

    print("Stage 8-B7 architecture checks passed.")
    print("Retrieval dependencies: 0")
    print("Benchmark-specific question IDs: 0")
    print("Evidence selection and issue verification: enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
