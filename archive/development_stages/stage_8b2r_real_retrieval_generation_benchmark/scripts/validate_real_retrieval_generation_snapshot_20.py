from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.retrieval_contract import RetrievalResultV1


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a frozen generation benchmark created from actual "
            "retrieval.v1 outputs."
        )
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = json.loads(
        args.benchmark.read_text(encoding="utf-8-sig")
    )

    assert benchmark["frozen"] is True
    assert benchmark["question_count"] == 20
    assert benchmark["input_mode"] == (
        "frozen_actual_retrieval_outputs"
    )
    assert len(benchmark["cases"]) == 20

    seen: set[str] = set()
    context_counts: dict[str, int] = {}

    for case in benchmark["cases"]:
        case_id = str(case["id"])
        assert case_id not in seen
        seen.add(case_id)

        path = args.retrieval_dir / case["retrieval_file"]
        assert path.exists(), path
        assert sha256_file(path) == case["retrieval_sha256"]

        retrieval = RetrievalResultV1.model_validate_json(
            path.read_text(encoding="utf-8-sig")
        )
        assert retrieval.question == case["question"]
        assert retrieval.decision.behavior == case[
            "actual_retrieval"
        ]["behavior"]
        assert retrieval.diagnostics.article_numbers == case[
            "actual_retrieval"
        ]["article_numbers"]

        state = case["actual_retrieval"]["context_state"]
        assert state in {
            "complete",
            "partial",
            "missing",
            "route_mismatch",
        }
        context_counts[state] = context_counts.get(state, 0) + 1

    runner_path = Path(
        "scripts/run_generation_benchmark_real_retrieval_20.py"
    )
    if runner_path.exists():
        source = runner_path.read_text(
            encoding="utf-8"
        ).lower()
        forbidden = [
            "retrievalonlypipeline",
            "retrievalservice",
            "import weaviate",
            "graphtraversalservice",
            "embeddings.create",
        ]
        for value in forbidden:
            assert value not in source, value

    print("Real-retrieval generation snapshot validation passed.")
    print("Cases: 20")
    print("Every retrieval SHA-256: verified")
    print(f"Context states: {context_counts}")
    print("Retrieval calls in generation runner: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
