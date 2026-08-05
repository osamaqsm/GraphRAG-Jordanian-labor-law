from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.retrieval_pipeline import RetrievalOnlyPipeline


BENCHMARK_FILENAME = "generation_holdout_fresh_unseen_20_snapshot.json"
MANIFEST_FILENAME = "retrieval_snapshot_manifest.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context_state(
    expected_behavior: str,
    actual_behavior: str,
    required_articles: list[int],
    actual_articles: list[int],
) -> str:
    if actual_behavior != expected_behavior:
        return "route_mismatch"

    if expected_behavior != "retrieve":
        return "complete"

    required = set(required_articles)
    actual = set(actual_articles)

    if required.issubset(actual):
        return "complete"
    if required & actual:
        return "partial"
    return "missing"


def optional_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return sha256_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real retrieval pipeline once for the fresh unseen "
            "generation holdout and freeze the exact retrieval.v1 outputs."
        )
    )
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing retrieval snapshot. Do not use this after "
            "the first generation run."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    holdout = json.loads(
        args.holdout.read_text(encoding="utf-8-sig")
    )
    cases = list(holdout.get("cases", []))

    if len(cases) != 20:
        raise RuntimeError("Expected exactly 20 holdout cases.")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise RuntimeError(
                f"Output directory is not empty: {args.output_dir}"
            )
        shutil.rmtree(args.output_dir)

    retrieval_dir = args.output_dir / "retrieval_inputs"
    retrieval_dir.mkdir(parents=True, exist_ok=True)

    frozen_cases: list[dict[str, Any]] = []
    manifest_items: list[dict[str, Any]] = []

    with RetrievalOnlyPipeline() as pipeline:
        for index, source_case in enumerate(cases, start=1):
            case = dict(source_case)
            case_id = str(case["id"])
            expected_behavior = str(case["expected_behavior"])
            required_articles = [
                int(value)
                for value in case.get("articles", [])
            ]

            result = pipeline.retrieve(
                str(case["question"]),
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
            missing = sorted(
                set(required_articles) - set(actual_articles)
            )
            extra = sorted(
                set(actual_articles) - set(required_articles)
            )
            state = context_state(
                expected_behavior,
                actual_behavior,
                required_articles,
                actual_articles,
            )
            digest = sha256_bytes(serialized)

            case["retrieval_file"] = filename
            case["retrieval_sha256"] = digest
            case["actual_retrieval"] = {
                "behavior": actual_behavior,
                "article_numbers": actual_articles,
                "missing_required_articles": missing,
                "extra_articles": extra,
                "context_state": state,
                "elapsed_ms": result.elapsed_ms,
            }
            frozen_cases.append(case)

            manifest_items.append(
                {
                    "id": case_id,
                    "file": filename,
                    "sha256": digest,
                    "expected_behavior": expected_behavior,
                    "actual_behavior": actual_behavior,
                    "required_articles": required_articles,
                    "actual_articles": actual_articles,
                    "missing_required_articles": missing,
                    "extra_articles": extra,
                    "context_state": state,
                }
            )

            print(
                f"[{index:02d}/20] {case_id} "
                f"expected={expected_behavior} "
                f"actual={actual_behavior} "
                f"articles={actual_articles} "
                f"context={state}"
            )

    created_at = datetime.now(timezone.utc).isoformat()
    frozen = {
        "benchmark_name": (
            "Jordanian Labor Law Fresh Unseen Generation Holdout — "
            "Frozen Real Retrieval"
        ),
        "benchmark_version": "1.0.0",
        "created_at_utc": created_at,
        "frozen": True,
        "input_contract": "retrieval.v1",
        "output_contract": "generation.v1",
        "input_mode": "fresh_unseen_frozen_actual_retrieval",
        "question_count": 20,
        "source_holdout_sha256": sha256_file(args.holdout),
        "warning": (
            "Do not modify this snapshot, its retrieval files, or the "
            "generation implementation before the first sealed run."
        ),
        "cases": frozen_cases,
    }

    benchmark_path = args.output_dir / BENCHMARK_FILENAME
    benchmark_path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    states: dict[str, int] = {}
    for item in manifest_items:
        state = item["context_state"]
        states[state] = states.get(state, 0) + 1

    project_root = Path("/app")
    code_hashes = {
        "retrieval_pipeline.py": optional_hash(
            project_root / "app" / "retrieval_pipeline.py"
        ),
        "retrieval_service.py": optional_hash(
            project_root / "app" / "retrieval_service.py"
        ),
        "legal_query_planner.py": optional_hash(
            project_root / "app" / "legal_query_planner.py"
        ),
        "legal_article_reranker.py": optional_hash(
            project_root / "app" / "legal_article_reranker.py"
        ),
    }

    manifest = {
        "created_at_utc": created_at,
        "holdout_file": str(args.holdout),
        "holdout_sha256": sha256_file(args.holdout),
        "benchmark_file": BENCHMARK_FILENAME,
        "benchmark_sha256": sha256_file(benchmark_path),
        "retrieval_input_count": 20,
        "include_debug": args.debug,
        "context_state_counts": states,
        "retrieval_code_hashes": code_hashes,
        "retrieval_inputs": manifest_items,
        "freeze_rule": (
            "Every retrieval SHA-256 must remain unchanged for the first "
            "sealed generation run."
        ),
    }
    manifest_path = args.output_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFresh unseen retrieval snapshot created.")
    print(f"Benchmark: {benchmark_path}")
    print(f"Manifest:  {manifest_path}")
    print(f"Context states: {states}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
