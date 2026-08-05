from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.retrieval_contract import RetrievalResultV1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the frozen generation benchmark package."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = json.loads(
        args.benchmark.read_text(encoding="utf-8-sig")
    )

    assert benchmark["question_count"] == 20
    assert benchmark["input_mode"] == "oracle_frozen_retrieval"
    assert len(benchmark["cases"]) == 20
    assert len(benchmark["retrieval_inputs"]) == 20

    ids = [case["id"] for case in benchmark["cases"]]
    assert len(ids) == len(set(ids))

    manifest = {
        item["id"]: item
        for item in benchmark["retrieval_inputs"]
    }

    for case in benchmark["cases"]:
        path = args.retrieval_dir / case["retrieval_file"]
        assert path.exists(), path

        item = manifest[case["id"]]
        assert sha256(path) == item["sha256"]

        retrieval = RetrievalResultV1.model_validate_json(
            path.read_text(encoding="utf-8-sig")
        )
        assert retrieval.schema_version == "retrieval.v1"
        assert retrieval.decision.behavior == "retrieve"
        assert retrieval.question == case["question"]

        numbers = [
            article.article_number
            for article in retrieval.articles
        ]
        assert numbers == case["articles"]
        assert all(article.text.strip() for article in retrieval.articles)

    runner = Path("scripts/run_generation_benchmark_20.py")
    if runner.exists():
        source = runner.read_text(encoding="utf-8").lower()
        forbidden = [
            "retrievalonlypipeline",
            "retrievalservice",
            "import weaviate",
            "graphtraversalservice",
            "embeddings.create",
        ]
        for value in forbidden:
            assert value not in source, value

    print("Generation benchmark validation passed.")
    print("Cases: 20")
    print("Input contract: retrieval.v1")
    print("Input mode: oracle-frozen retrieval")
    print("Retrieval calls during scoring: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
