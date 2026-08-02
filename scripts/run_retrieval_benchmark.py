from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_BENCHMARK_PATH = Path(
    "/app/data/benchmarks/retrieval_benchmark_20.json"
)
# /app/data is mounted read-only by this project. Write inside the
# container's writable /tmp directory, then copy the result to Windows.
DEFAULT_OUTPUT_PATH = Path(
    "/tmp/retrieval_benchmark_results.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 20-question Jordanian Labor Law retrieval "
            "benchmark against scripts.test_retrieval."
        )
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help="Benchmark JSON path inside the container.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Result JSON path inside the container.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="First 1-based question number to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of questions to run.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a question fails to execute.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return value


def extract_result_json(text: str) -> dict[str, Any]:
    """Extract the retrieval JSON even if logs surround it."""

    decoder = json.JSONDecoder()

    for index, character in enumerate(text):
        if character != "{":
            continue

        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue

        if (
            isinstance(value, dict)
            and "retrieval" in value
            and "status" in value
        ):
            return value

    raise ValueError(
        "Could not find the retrieval JSON object in command output."
    )


def run_question(question: str) -> tuple[dict[str, Any], float]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    command = [
        sys.executable,
        "-m",
        "scripts.test_retrieval",
        question,
    ]

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd="/app",
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        raise RuntimeError(
            "scripts.test_retrieval failed with exit code "
            f"{completed.returncode}.\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    return extract_result_json(completed.stdout), elapsed


def ordered_unique(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result


def actual_articles(result: dict[str, Any]) -> list[int]:
    diagnostics = result.get("diagnostics", {})
    numbers = diagnostics.get("top_article_numbers")

    if isinstance(numbers, list):
        return ordered_unique(
            [int(number) for number in numbers if number is not None]
        )

    retrieval = result.get("retrieval", {})
    hits = retrieval.get("article_hits", [])

    values: list[int] = []
    for hit in hits:
        number = hit.get("article_number")
        if number is not None:
            values.append(int(number))

    return ordered_unique(values)


def graph_articles(result: dict[str, Any]) -> list[int]:
    diagnostics = result.get("diagnostics", {})
    values = diagnostics.get("graph_supported_articles", [])

    if not isinstance(values, list):
        return []

    return ordered_unique(
        [int(value) for value in values if value is not None]
    )


def actual_concepts(result: dict[str, Any]) -> list[str]:
    retrieval = result.get("retrieval", {})
    values: list[str] = []

    for key in ("concept_hits", "expanded_concept_hits"):
        for hit in retrieval.get(key, []):
            local_name = str(hit.get("local_name", "")).strip()
            if local_name:
                values.append(local_name)

    return list(dict.fromkeys(values))


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_case(
    case: dict[str, Any],
    result: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    actual = actual_articles(result)
    graph_supported = graph_articles(result)
    concepts = actual_concepts(result)

    primary = int(case["primary_article"])
    required = [int(value) for value in case["required_articles"]]
    acceptable = [
        int(value)
        for value in case.get("acceptable_articles", required)
    ]
    expected_concepts = [
        str(value)
        for value in case.get("expected_concepts_any", [])
    ]

    actual_set = set(actual)
    required_set = set(required)
    acceptable_set = set(acceptable)

    strict_true_positive = len(actual_set & required_set)
    lenient_true_positive = len(actual_set & acceptable_set)

    strict_precision = safe_ratio(
        strict_true_positive,
        len(actual_set),
    )
    required_recall = safe_ratio(
        strict_true_positive,
        len(required_set),
    )
    strict_f1 = f1_score(strict_precision, required_recall)

    lenient_precision = safe_ratio(
        lenient_true_positive,
        len(actual_set),
    )
    lenient_recall = safe_ratio(
        lenient_true_positive,
        len(acceptable_set),
    )
    lenient_f1 = f1_score(lenient_precision, lenient_recall)

    primary_rank = (
        actual.index(primary) + 1
        if primary in actual
        else None
    )

    concept_matches = sorted(
        set(concepts) & set(expected_concepts)
    )

    all_required_found = required_set.issubset(actual_set)
    exact_set_match = actual_set == required_set
    ordered_exact_match = actual == required
    hit_at_1 = bool(actual and actual[0] == primary)
    hit_at_3 = primary in actual[:3]
    reciprocal_rank = (
        1.0 / primary_rank
        if primary_rank is not None
        else 0.0
    )
    concept_hit = bool(concept_matches)

    passed = all_required_found and hit_at_1

    return {
        "id": case["id"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "question": case["question"],
        "expected": {
            "primary_article": primary,
            "required_articles": required,
            "acceptable_articles": acceptable,
            "concepts_any": expected_concepts,
            "description": case.get("expected_result", ""),
        },
        "actual": {
            "articles": actual,
            "graph_supported_articles": graph_supported,
            "concepts": concepts,
        },
        "metrics": {
            "pass": passed,
            "all_required_found": all_required_found,
            "exact_set_match": exact_set_match,
            "ordered_exact_match": ordered_exact_match,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "primary_rank": primary_rank,
            "reciprocal_rank": round(reciprocal_rank, 6),
            "required_recall": round(required_recall, 6),
            "strict_precision": round(strict_precision, 6),
            "strict_f1": round(strict_f1, 6),
            "lenient_precision": round(lenient_precision, 6),
            "lenient_recall": round(lenient_recall, 6),
            "lenient_f1": round(lenient_f1, 6),
            "concept_hit": concept_hit,
            "matched_expected_concepts": concept_matches,
            "unexpected_articles": sorted(
                actual_set - acceptable_set
            ),
            "missing_required_articles": sorted(
                required_set - actual_set
            ),
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def average_metric(
    rows: list[dict[str, Any]],
    key: str,
) -> float:
    if not rows:
        return 0.0

    return round(
        statistics.fmean(
            float(row["metrics"][key])
            for row in rows
        ),
        6,
    )


def make_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if "metrics" in row]
    errors = [row for row in rows if "error" in row]

    total = len(rows)
    passed = sum(
        1
        for row in completed
        if row["metrics"]["pass"]
    )

    return {
        "questions_requested": total,
        "questions_completed": len(completed),
        "questions_failed_to_run": len(errors),
        "passed": passed,
        "pass_rate": round(
            safe_ratio(passed, total),
            6,
        ),
        "hit_at_1_accuracy": average_metric(
            completed,
            "hit_at_1",
        ),
        "hit_at_3_accuracy": average_metric(
            completed,
            "hit_at_3",
        ),
        "mean_reciprocal_rank": average_metric(
            completed,
            "reciprocal_rank",
        ),
        "exact_set_accuracy": average_metric(
            completed,
            "exact_set_match",
        ),
        "required_article_recall": average_metric(
            completed,
            "required_recall",
        ),
        "mean_strict_precision": average_metric(
            completed,
            "strict_precision",
        ),
        "mean_strict_f1": average_metric(
            completed,
            "strict_f1",
        ),
        "mean_lenient_precision": average_metric(
            completed,
            "lenient_precision",
        ),
        "mean_lenient_f1": average_metric(
            completed,
            "lenient_f1",
        ),
        "concept_hit_accuracy": average_metric(
            completed,
            "concept_hit",
        ),
        "mean_elapsed_seconds": round(
            statistics.fmean(
                row["elapsed_seconds"]
                for row in completed
            ),
            3,
        )
        if completed
        else 0.0,
    }


def print_case(row: dict[str, Any], index: int, total: int) -> None:
    if "error" in row:
        print(
            f"[{index:02d}/{total:02d}] {row['id']} ERROR: "
            f"{row['error']}"
        )
        return

    metrics = row["metrics"]
    marker = "PASS" if metrics["pass"] else "FAIL"

    print(
        f"[{index:02d}/{total:02d}] {row['id']} {marker} | "
        f"expected={row['expected']['required_articles']} | "
        f"actual={row['actual']['articles']} | "
        f"Hit@1={int(metrics['hit_at_1'])} | "
        f"Recall={metrics['required_recall']:.2f} | "
        f"F1={metrics['strict_f1']:.2f} | "
        f"{row['elapsed_seconds']:.1f}s"
    )

    # Print compact diagnostics only for failed cases.
    if not metrics["pass"]:
        print(
            "    graph="
            f"{row['actual']['graph_supported_articles']} | "
            "missing="
            f"{metrics['missing_required_articles']} | "
            "unexpected="
            f"{metrics['unexpected_articles']}"
        )
        print(
            "    concepts="
            f"{row['actual']['concepts'][:12]}"
        )


def main() -> int:
    args = parse_args()
    benchmark = load_json(args.benchmark)

    questions = benchmark.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(
            "The benchmark must contain a non-empty questions array."
        )

    start_index = max(args.start - 1, 0)
    selected = questions[start_index:]

    if args.limit is not None:
        selected = selected[: max(args.limit, 0)]

    if not selected:
        raise ValueError("No benchmark questions were selected.")

    rows: list[dict[str, Any]] = []
    total = len(selected)

    for index, case in enumerate(selected, start=1):
        try:
            result, elapsed = run_question(case["question"])
            row = evaluate_case(case, result, elapsed)
        except Exception as exc:  # noqa: BLE001 - benchmark must record failures.
            row = {
                "id": case.get("id", f"question_{index}"),
                "category": case.get("category", "unknown"),
                "difficulty": case.get("difficulty", "unknown"),
                "question": case.get("question", ""),
                "error": str(exc),
            }

            if args.stop_on_error:
                rows.append(row)
                print_case(row, index, total)
                break

        rows.append(row)
        print_case(row, index, total)

    output = {
        "benchmark_name": benchmark.get("benchmark_name"),
        "benchmark_version": benchmark.get("benchmark_version"),
        "runner": "scripts.run_retrieval_benchmark",
        "summary": make_summary(rows),
        "results": rows,
    }

    output_path = args.output

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                output,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")
    except OSError as exc:
        fallback_path = Path(
            "/tmp/retrieval_benchmark_results.json"
        )

        if output_path == fallback_path:
            raise

        print(
            "\nWarning: could not write to "
            f"{output_path}: {exc}"
        )
        print(f"Falling back to: {fallback_path}")

        fallback_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with fallback_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                output,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

        output_path = fallback_path

    print("\nSummary")
    print(json.dumps(output["summary"], indent=2))
    print(f"\nSaved: {output_path}")

    return 0 if output["summary"]["questions_failed_to_run"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())