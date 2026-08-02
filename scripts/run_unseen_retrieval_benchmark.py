from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_BENCHMARK_PATH = Path(
    "/app/data/benchmarks/retrieval_benchmark_unseen_50.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "/tmp/retrieval_benchmark_unseen_50_results.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 50-question unseen Jordanian Labor Law "
            "retrieval benchmark."
        )
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8-sig") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_result_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "retrieval" in value and "status" in value:
            return value
    raise ValueError("Could not find retrieval JSON in command output.")


def run_question(question: str) -> tuple[dict[str, Any], float]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    command = [sys.executable, "-m", "scripts.test_retrieval", question]
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
            f"scripts.test_retrieval exited with {completed.returncode}.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return extract_result_json(completed.stdout), elapsed


def ordered_unique(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def actual_articles(result: dict[str, Any]) -> list[int]:
    diagnostics = result.get("diagnostics", {})
    numbers = diagnostics.get("top_article_numbers")
    if isinstance(numbers, list):
        return ordered_unique([int(v) for v in numbers if v is not None])
    hits = result.get("retrieval", {}).get("article_hits", [])
    return ordered_unique(
        [int(hit["article_number"]) for hit in hits if hit.get("article_number") is not None]
    )


def graph_articles(result: dict[str, Any]) -> list[int]:
    values = result.get("diagnostics", {}).get("graph_supported_articles", [])
    if not isinstance(values, list):
        return []
    return ordered_unique([int(v) for v in values if v is not None])


def actual_concepts(result: dict[str, Any]) -> list[str]:
    retrieval = result.get("retrieval", {})
    values: list[str] = []
    for key in ("concept_hits", "expanded_concept_hits"):
        for hit in retrieval.get(key, []):
            name = str(hit.get("local_name", "")).strip()
            if name:
                values.append(name)
    return list(dict.fromkeys(values))


def flatten_status_text(value: Any) -> list[str]:
    output: list[str] = []
    if isinstance(value, str):
        output.append(value.lower())
    elif isinstance(value, dict):
        for key, item in value.items():
            if any(token in str(key).lower() for token in (
                "status", "decision", "reason", "confidence", "scope", "clarif", "abstain"
            )):
                output.extend(flatten_status_text(item))
    elif isinstance(value, list):
        for item in value:
            output.extend(flatten_status_text(item))
    return output


def infer_behavior_signal(result: dict[str, Any], articles: list[int]) -> str:
    text = " ".join(flatten_status_text(result))
    if any(token in text for token in ("out_of_scope", "out of scope", "outside scope")):
        return "abstain"
    if any(token in text for token in ("clarification", "clarify", "ambiguous", "insufficient detail")):
        return "clarify"
    if any(token in text for token in ("abstain", "low_confidence", "low confidence", "no_answer")):
        return "safe_no_answer"
    return "retrieved" if articles else "no_retrieval"


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_retrieve(case: dict[str, Any], result: dict[str, Any], elapsed: float) -> dict[str, Any]:
    actual = actual_articles(result)
    concepts = actual_concepts(result)
    graph = graph_articles(result)
    primary = int(case["primary_article"])
    required = [int(v) for v in case["required_articles"]]
    acceptable = [int(v) for v in case.get("acceptable_articles", required)]
    expected_concepts = [str(v) for v in case.get("expected_concepts_any", [])]

    actual_set, required_set, acceptable_set = set(actual), set(required), set(acceptable)
    tp = len(actual_set & required_set)
    precision = safe_ratio(tp, len(actual_set)) if actual_set else 0.0
    recall = safe_ratio(tp, len(required_set))
    primary_rank = actual.index(primary) + 1 if primary in actual else None
    concept_matches = sorted(set(concepts) & set(expected_concepts))
    concept_scorable = bool(expected_concepts)

    metrics = {
        "pass": required_set.issubset(actual_set) and bool(actual) and actual[0] == primary,
        "all_required_found": required_set.issubset(actual_set),
        "exact_set_match": actual_set == required_set,
        "ordered_exact_match": actual == required,
        "hit_at_1": bool(actual) and actual[0] == primary,
        "hit_at_3": primary in actual[:3],
        "primary_rank": primary_rank,
        "reciprocal_rank": round(1.0 / primary_rank, 6) if primary_rank else 0.0,
        "required_recall": round(recall, 6),
        "strict_precision": round(precision, 6),
        "strict_f1": round(f1(precision, recall), 6),
        "lenient_precision": round(safe_ratio(len(actual_set & acceptable_set), len(actual_set)), 6) if actual_set else 0.0,
        "concept_scorable": concept_scorable,
        "concept_hit": bool(concept_matches) if concept_scorable else None,
        "matched_expected_concepts": concept_matches,
        "unexpected_articles": sorted(actual_set - acceptable_set),
        "missing_required_articles": sorted(required_set - actual_set),
    }
    return {
        "id": case["id"], "test_type": case["test_type"], "category": case["category"],
        "difficulty": case["difficulty"], "question": case["question"],
        "expected_behavior": "retrieve",
        "expected": {
            "primary_article": primary, "required_articles": required,
            "acceptable_articles": acceptable, "concepts_any": expected_concepts,
            "description": case.get("expected_result", ""),
        },
        "actual": {
            "behavior_signal": infer_behavior_signal(result, actual),
            "articles": actual, "graph_supported_articles": graph, "concepts": concepts,
        },
        "metrics": metrics, "elapsed_seconds": round(elapsed, 3),
    }


def evaluate_safety(case: dict[str, Any], result: dict[str, Any], elapsed: float) -> dict[str, Any]:
    actual = actual_articles(result)
    concepts = actual_concepts(result)
    signal = infer_behavior_signal(result, actual)
    safe_no_retrieval = len(actual) == 0
    expected_behavior = str(case["expected_behavior"])
    explicit_match = signal == expected_behavior
    metrics = {
        "pass": safe_no_retrieval,
        "safe_no_retrieval": safe_no_retrieval,
        "explicit_behavior_match": explicit_match,
        "returned_article_count": len(actual),
    }
    return {
        "id": case["id"], "test_type": case["test_type"], "category": case["category"],
        "difficulty": case["difficulty"], "question": case["question"],
        "expected_behavior": expected_behavior,
        "expected": {"description": case.get("expected_result", "")},
        "actual": {
            "behavior_signal": signal, "articles": actual,
            "graph_supported_articles": graph_articles(result), "concepts": concepts,
        },
        "metrics": metrics, "elapsed_seconds": round(elapsed, 3),
    }


def evaluate_case(case: dict[str, Any], result: dict[str, Any], elapsed: float) -> dict[str, Any]:
    if case.get("expected_behavior") == "retrieve":
        return evaluate_retrieve(case, result, elapsed)
    return evaluate_safety(case, result, elapsed)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row["metrics"][key]) for row in rows if row["metrics"].get(key) is not None]
    return round(statistics.fmean(values), 6) if values else 0.0


def category_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "metrics" in row:
            groups[row["test_type"]].append(row)
    output: dict[str, Any] = {}
    for name, items in groups.items():
        passed = sum(bool(item["metrics"]["pass"]) for item in items)
        output[name] = {
            "count": len(items), "passed": passed,
            "accuracy": round(safe_ratio(passed, len(items)), 6),
        }
    return output


def make_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if "metrics" in row]
    errors = [row for row in rows if "error" in row]
    retrieve = [row for row in completed if row["expected_behavior"] == "retrieve"]
    safety = [row for row in completed if row["expected_behavior"] != "retrieve"]
    clarify = [row for row in safety if row["expected_behavior"] == "clarify"]
    abstain = [row for row in safety if row["expected_behavior"] == "abstain"]
    concept_rows = [row for row in retrieve if row["metrics"].get("concept_scorable")]
    passed = sum(bool(row["metrics"]["pass"]) for row in completed)
    return {
        "questions_requested": len(rows),
        "questions_completed": len(completed),
        "questions_failed_to_run": len(errors),
        "passed": passed,
        "overall_accuracy": round(safe_ratio(passed, len(rows)), 6),
        "retrieve_cases": len(retrieve),
        "retrieve_pass_rate": mean(retrieve, "pass"),
        "hit_at_1_accuracy": mean(retrieve, "hit_at_1"),
        "hit_at_3_accuracy": mean(retrieve, "hit_at_3"),
        "mean_reciprocal_rank": mean(retrieve, "reciprocal_rank"),
        "exact_set_accuracy": mean(retrieve, "exact_set_match"),
        "required_article_recall": mean(retrieve, "required_recall"),
        "mean_strict_precision": mean(retrieve, "strict_precision"),
        "mean_strict_f1": mean(retrieve, "strict_f1"),
        "concept_hit_accuracy": mean(concept_rows, "concept_hit"),
        "safety_cases": len(safety),
        "safe_non_answer_accuracy": mean(safety, "safe_no_retrieval"),
        "clarification_accuracy": mean(clarify, "safe_no_retrieval"),
        "out_of_scope_abstention_accuracy": mean(abstain, "safe_no_retrieval"),
        "explicit_safety_signal_accuracy": mean(safety, "explicit_behavior_match"),
        "mean_elapsed_seconds": round(statistics.fmean(row["elapsed_seconds"] for row in completed), 3) if completed else 0.0,
        "by_test_type": category_breakdown(completed),
    }


def print_case(row: dict[str, Any], index: int, total: int) -> None:
    if "error" in row:
        print(f"[{index:02d}/{total:02d}] {row['id']} ERROR | {row['error']}")
        return
    marker = "PASS" if row["metrics"]["pass"] else "FAIL"
    if row["expected_behavior"] == "retrieve":
        print(
            f"[{index:02d}/{total:02d}] {row['id']} {marker} "
            f"({row['test_type']}) | expected={row['expected']['required_articles']} "
            f"| actual={row['actual']['articles']} | Hit@1={int(row['metrics']['hit_at_1'])} "
            f"| Recall={row['metrics']['required_recall']:.2f} | {row['elapsed_seconds']:.1f}s"
        )
        if not row["metrics"]["pass"]:
            print(
                f"    missing={row['metrics']['missing_required_articles']} "
                f"unexpected={row['metrics']['unexpected_articles']} "
                f"graph={row['actual']['graph_supported_articles']}"
            )
    else:
        print(
            f"[{index:02d}/{total:02d}] {row['id']} {marker} "
            f"({row['expected_behavior']}) | actual_articles={row['actual']['articles']} "
            f"| signal={row['actual']['behavior_signal']} | {row['elapsed_seconds']:.1f}s"
        )


def main() -> int:
    args = parse_args()
    benchmark = load_json(args.benchmark)
    if not benchmark.get("frozen"):
        raise ValueError("Unseen benchmark must be marked frozen=true.")
    questions = benchmark.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Benchmark has no questions.")
    selected = questions[max(args.start - 1, 0):]
    if args.limit is not None:
        selected = selected[:max(args.limit, 0)]
    if not selected:
        raise ValueError("No benchmark questions selected.")

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(selected, start=1):
        try:
            result, elapsed = run_question(str(case["question"]))
            row = evaluate_case(case, result, elapsed)
        except Exception as exc:  # benchmark records execution failures
            row = {
                "id": case.get("id", f"question_{index}"),
                "test_type": case.get("test_type", "unknown"),
                "category": case.get("category", "unknown"),
                "difficulty": case.get("difficulty", "unknown"),
                "question": case.get("question", ""),
                "error": str(exc),
            }
            rows.append(row)
            print_case(row, index, len(selected))
            if args.stop_on_error:
                break
            continue
        rows.append(row)
        print_case(row, index, len(selected))

    output = {
        "benchmark_name": benchmark.get("benchmark_name"),
        "benchmark_version": benchmark.get("benchmark_version"),
        "benchmark_sha256": file_sha256(args.benchmark),
        "runner": "scripts.run_unseen_retrieval_benchmark",
        "summary": make_summary(rows),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print("\nSummary")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"\nBenchmark SHA-256: {output['benchmark_sha256']}")
    print(f"Saved: {args.output}")
    return 0 if output["summary"]["questions_failed_to_run"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
