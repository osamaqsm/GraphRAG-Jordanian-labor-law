from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.retrieval_pipeline import RetrievalOnlyPipeline


SNAPSHOT_NAME = "generation_real_retrieval_20"
BENCHMARK_FILENAME = "generation_benchmark_real_retrieval_20.json"
MANIFEST_FILENAME = "retrieval_snapshot_manifest.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def context_state(
    *,
    expected_behavior: str,
    actual_behavior: str,
    required_articles: list[int],
    actual_articles: list[int],
) -> str:
    if actual_behavior != expected_behavior:
        return "route_mismatch"

    if expected_behavior != "retrieve":
        return "complete"

    required = set(int(value) for value in required_articles)
    actual = set(int(value) for value in actual_articles)

    if not required:
        return "complete"
    if required.issubset(actual):
        return "complete"
    if required & actual:
        return "partial"
    return "missing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real retrieval pipeline for 20 questions and freeze the "
            "exact retrieval.v1 outputs for later generation-only testing."
        )
    )
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include the retrieval debug field in the frozen outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing snapshot directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rubric = json.loads(
        args.rubric.read_text(encoding="utf-8-sig")
    )
    cases = list(rubric.get("cases", []))

    if len(cases) != 20:
        raise RuntimeError(
            f"Expected exactly 20 rubric cases, found {len(cases)}."
        )

    if args.output_dir.exists():
        existing = any(args.output_dir.iterdir())
        if existing and not args.overwrite:
            raise RuntimeError(
                f"Snapshot directory is not empty: {args.output_dir}. "
                "Use --overwrite only when intentionally creating a new "
                "retrieval snapshot."
            )
        if args.overwrite:
            shutil.rmtree(args.output_dir)

    retrieval_dir = args.output_dir / "retrieval_inputs"
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    frozen_cases: list[dict[str, Any]] = []
    manifest_items: list[dict[str, Any]] = []

    with RetrievalOnlyPipeline() as pipeline:
        for index, original_case in enumerate(cases, start=1):
            case = dict(original_case)
            case_id = str(case["id"])
            question = str(case["question"])
            expected_behavior = str(
                case.get("expected_behavior", "retrieve")
            )
            required_articles = [
                int(value)
                for value in case.get("articles", [])
            ]

            result = pipeline.retrieve(
                question,
                include_debug=args.debug,
            )
            payload = result.model_dump(mode="json")
            serialized = (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8")

            filename = f"{case_id.lower()}_retrieval.json"
            output_path = retrieval_dir / filename
            output_path.write_bytes(serialized)

            actual_behavior = result.decision.behavior
            actual_articles = [
                int(value)
                for value in result.diagnostics.article_numbers
            ]
            missing_articles = sorted(
                set(required_articles) - set(actual_articles)
            )
            extra_articles = sorted(
                set(actual_articles) - set(required_articles)
            )
            state = context_state(
                expected_behavior=expected_behavior,
                actual_behavior=actual_behavior,
                required_articles=required_articles,
                actual_articles=actual_articles,
            )
            digest = sha256_bytes(serialized)

            case["retrieval_file"] = filename
            case["retrieval_sha256"] = digest
            case["actual_retrieval"] = {
                "behavior": actual_behavior,
                "article_numbers": actual_articles,
                "missing_required_articles": missing_articles,
                "extra_articles": extra_articles,
                "context_state": state,
                "elapsed_ms": result.elapsed_ms,
            }
            frozen_cases.append(case)

            manifest_items.append(
                {
                    "id": case_id,
                    "question": question,
                    "file": filename,
                    "sha256": digest,
                    "expected_behavior": expected_behavior,
                    "actual_behavior": actual_behavior,
                    "required_articles": required_articles,
                    "actual_articles": actual_articles,
                    "missing_required_articles": missing_articles,
                    "extra_articles": extra_articles,
                    "context_state": state,
                }
            )

            print(
                f"[{index:02d}/20] {case_id} "
                f"behavior={actual_behavior} "
                f"articles={actual_articles} "
                f"context={state}"
            )

    frozen_benchmark = {
        "benchmark_name": (
            "Jordanian Labor Law Generation Benchmark — "
            "Frozen Real Retrieval 20"
        ),
        "benchmark_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": True,
        "input_contract": "retrieval.v1",
        "output_contract": "generation.v1",
        "input_mode": "frozen_actual_retrieval_outputs",
        "question_count": 20,
        "scope": (
            "Generation is evaluated from the exact retrieval.v1 JSON outputs "
            "created by the deployed retrieval pipeline. The generation runner "
            "does not rerun retrieval."
        ),
        "interpretation": {
            "retrieval_context_complete": (
                "All required gold articles were present in the actual "
                "retrieval output."
            ),
            "retrieval_limited": (
                "The actual retrieval route or articles were incomplete; "
                "generation fact completeness is not attributed to the prompt."
            ),
            "generation_evaluable": (
                "Only complete retrieval contexts are included in strict "
                "generation-quality accuracy."
            ),
        },
        "cases": frozen_cases,
    }

    benchmark_path = args.output_dir / BENCHMARK_FILENAME
    benchmark_path.write_text(
        json.dumps(
            frozen_benchmark,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    counts: dict[str, int] = {}
    for item in manifest_items:
        state = item["context_state"]
        counts[state] = counts.get(state, 0) + 1

    manifest = {
        "snapshot_name": SNAPSHOT_NAME,
        "created_at_utc": frozen_benchmark["created_at_utc"],
        "rubric_file": str(args.rubric),
        "rubric_sha256": sha256_file(args.rubric),
        "benchmark_file": BENCHMARK_FILENAME,
        "benchmark_sha256": sha256_file(benchmark_path),
        "retrieval_input_count": 20,
        "include_debug": args.debug,
        "context_state_counts": counts,
        "retrieval_inputs": manifest_items,
        "freeze_rule": (
            "Do not modify the retrieval JSON files. The generation runner "
            "verifies every SHA-256 hash before testing."
        ),
    }
    manifest_path = args.output_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFrozen real-retrieval snapshot created.")
    print(f"Directory: {args.output_dir}")
    print(f"Benchmark: {benchmark_path}")
    print(f"Manifest:  {manifest_path}")
    print(f"Context states: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
