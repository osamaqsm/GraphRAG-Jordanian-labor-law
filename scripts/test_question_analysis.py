from __future__ import annotations

import json
from pathlib import Path

from app.legal_question_analysis import analyze_legal_question


BENCHMARK_PATH = Path(
    "/app/data/benchmarks/"
    "retrieval_benchmark_20.json"
)


def main() -> int:
    with BENCHMARK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        benchmark = json.load(file)

    questions = benchmark.get(
        "questions",
        [],
    )

    for item in questions:
        analysis = analyze_legal_question(
            str(item["question"])
        )

        print(
            f"{item['id']} | "
            f"issues={list(analysis.issue_ids)} | "
            f"concepts={list(analysis.preferred_concepts)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
