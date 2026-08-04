#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def post_json(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    response = session.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        timeout=timeout,
    )

    try:
        body = response.json()
    except ValueError:
        body = {
            "non_json_response": response.text,
            "status_code": response.status_code,
        }

    if not response.ok:
        raise RuntimeError(
            f"POST {url} failed with HTTP {response.status_code}: "
            f"{json.dumps(body, ensure_ascii=False)}"
        )

    if not isinstance(body, dict):
        raise RuntimeError(f"POST {url} returned a non-object JSON response.")

    return body


def extract_article_numbers(retrieval: dict[str, Any]) -> list[int]:
    result: list[int] = []
    for article in retrieval.get("articles", []) or []:
        number = article.get("article_number")
        if isinstance(number, int) and number not in result:
            result.append(number)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 20-question Step 7 -> Step 8 flow. "
            "The exact JSON returned by /retrieve is sent directly to /generate."
        )
    )
    parser.add_argument(
        "--questions",
        default="data/retrieval_benchmark_unseen_20_stage_7_7d.json",
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--output-dir",
        default="data/manual_review/stage7_step8_20",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retrieve-debug", action="store_true")
    parser.add_argument("--generate-debug", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    questions_path = Path(args.questions)
    output_dir = Path(args.output_dir)
    retrieval_dir = output_dir / "step7"
    generation_dir = output_dir / "step8"

    source = read_json(questions_path)
    questions = source.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Input JSON must contain a non-empty 'questions' array.")

    base_url = args.base_url.rstrip("/")
    retrieve_url = f"{base_url}/retrieve"
    generate_url = (
        f"{base_url}/generate"
        f"?include_debug={'true' if args.generate_debug else 'false'}"
    )

    retrieval_dir.mkdir(parents=True, exist_ok=True)
    generation_dir.mkdir(parents=True, exist_ok=True)

    combined: dict[str, Any] = {
        "source_benchmark_name": source.get("benchmark_name", ""),
        "source_benchmark_version": source.get("benchmark_version", ""),
        "question_count": len(questions),
        "base_url": base_url,
        "results": [],
    }

    failures = 0

    with requests.Session() as session:
        for index, item in enumerate(questions, start=1):
            question_id = str(item.get("id") or f"Q{index:02d}")
            question = str(item.get("question") or "").strip()
            started = time.perf_counter()

            record: dict[str, Any] = {
                "id": question_id,
                "test_type": item.get("test_type"),
                "category": item.get("category"),
                "difficulty": item.get("difficulty"),
                "question": question,
                "expected_behavior": item.get("expected_behavior"),
                "expected_result": item.get("expected_result"),
                "required_articles": item.get("required_articles", []),
            }

            try:
                if not question:
                    raise ValueError("Question is empty.")

                print(
                    f"[{index:02d}/{len(questions):02d}] "
                    f"{question_id}: calling Step 7...",
                    flush=True,
                )

                retrieval_request = {
                    "question": question,
                    "include_debug": bool(args.retrieve_debug),
                }

                retrieval = post_json(
                    session,
                    retrieve_url,
                    retrieval_request,
                    args.timeout,
                )

                retrieval_path = (
                    retrieval_dir / f"{question_id.lower()}_retrieval.json"
                )
                write_json(retrieval_path, retrieval)

                print(
                    f"[{index:02d}/{len(questions):02d}] "
                    f"{question_id}: Step 7 done; calling Step 8...",
                    flush=True,
                )

                # The exact Step 7 response is the Step 8 request body.
                generation = post_json(
                    session,
                    generate_url,
                    retrieval,
                    args.timeout,
                )

                generation_path = (
                    generation_dir / f"{question_id.lower()}_generation.json"
                )
                write_json(generation_path, generation)

                record.update(
                    {
                        "run_status": "success",
                        "step7_file": str(retrieval_path),
                        "step8_file": str(generation_path),
                        "step7_behavior": retrieval.get(
                            "decision", {}
                        ).get("behavior"),
                        "retrieved_article_numbers": extract_article_numbers(
                            retrieval
                        ),
                        "step8_status": generation.get("status"),
                        "answer_ar": generation.get("answer_ar", ""),
                        "cited_article_numbers": generation.get(
                            "cited_article_numbers", []
                        ),
                        "grounded": generation.get("grounded", False),
                        "warnings": generation.get("warnings", []),
                        "elapsed_ms": round(
                            (time.perf_counter() - started) * 1000
                        ),
                    }
                )

                print(
                    f"[{index:02d}/{len(questions):02d}] {question_id}: "
                    f"behavior={record['step7_behavior']} "
                    f"articles={record['retrieved_article_numbers']} "
                    f"generation={record['step8_status']} "
                    f"citations={record['cited_article_numbers']}",
                    flush=True,
                )

            except Exception as exc:
                failures += 1
                record.update(
                    {
                        "run_status": "failed",
                        "error": str(exc),
                        "elapsed_ms": round(
                            (time.perf_counter() - started) * 1000
                        ),
                    }
                )
                print(
                    f"[{index:02d}/{len(questions):02d}] "
                    f"{question_id}: FAILED: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

            combined["results"].append(record)
            write_json(output_dir / "combined_results.json", combined)

            if args.delay > 0 and index < len(questions):
                time.sleep(args.delay)

    combined["questions_succeeded"] = sum(
        1 for item in combined["results"]
        if item.get("run_status") == "success"
    )
    combined["questions_failed"] = failures
    write_json(output_dir / "combined_results.json", combined)

    lines = [
        "# Step 7 → Step 8 Manual Review",
        "",
        f"- Questions: {len(questions)}",
        f"- Succeeded: {combined['questions_succeeded']}",
        f"- Failed: {combined['questions_failed']}",
        "",
    ]

    for record in combined["results"]:
        lines.extend(
            [
                f"## {record['id']}",
                "",
                f"**Question:** {record['question']}",
                "",
                f"**Expected behavior:** "
                f"{record.get('expected_behavior', '')}",
                "",
                f"**Step 7 behavior:** "
                f"{record.get('step7_behavior', '')}",
                "",
                "**Retrieved articles:** "
                + ", ".join(
                    str(x)
                    for x in record.get(
                        "retrieved_article_numbers", []
                    )
                ),
                "",
                f"**Step 8 status:** "
                f"{record.get('step8_status', '')}",
                "",
                f"**Answer:** {record.get('answer_ar', '')}",
                "",
                "**Cited articles:** "
                + ", ".join(
                    str(x)
                    for x in record.get(
                        "cited_article_numbers", []
                    )
                ),
                "",
                f"**Expected result:** "
                f"{record.get('expected_result', '')}",
                "",
                "### Manual checks",
                "",
                "- [ ] The answer addresses the exact question.",
                "- [ ] The answer is supported by the retrieved article text.",
                "- [ ] Actors, conditions, numbers, and deadlines are preserved.",
                "- [ ] Citations refer only to retrieved articles.",
                "- [ ] Arabic wording is clear and concise.",
                "",
                "---",
                "",
            ]
        )

    (output_dir / "manual_review.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )

    print()
    print(f"Combined JSON: {output_dir / 'combined_results.json'}")
    print(f"Manual review: {output_dir / 'manual_review.md'}")
    print(f"Step 7 files: {retrieval_dir}")
    print(f"Step 8 files: {generation_dir}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
