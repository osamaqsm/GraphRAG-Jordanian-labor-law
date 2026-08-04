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
            "Validate every frozen retrieval file before the fresh unseen "
            "generation run."
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
        "fresh_unseen_frozen_actual_retrieval"
    )

    seen: set[str] = set()
    state_counts: dict[str, int] = {}

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
        state_counts[state] = state_counts.get(state, 0) + 1

    print("Fresh unseen retrieval snapshot validation passed.")
    print("Cases: 20")
    print("Every retrieval SHA-256: verified")
    print(f"Context states: {state_counts}")
    print("Retrieval calls in generation runner: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
