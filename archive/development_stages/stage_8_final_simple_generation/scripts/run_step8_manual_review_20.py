from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.grounded_answer_generator import GroundedAnswerGenerator
from app.retrieval_contract import RetrievalResultV1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Step 8 on 20 saved retrieval.v1 files and create a manual "
            "review report. This script performs no automatic grading."
        )
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Optional case ID, for example --case G01.",
    )
    return parser.parse_args()


def article_numbers(retrieval: RetrievalResultV1) -> list[int]:
    return [
        int(article.article_number)
        for article in retrieval.articles
        if article.article_number is not None
    ]


def markdown_report(items: list[dict[str, Any]]) -> str:
    lines = [
        "# Step 8 — Manual Answer Review (20 Questions)",
        "",
        "This report does not calculate pass/fail metrics. Review every answer manually.",
        "",
    ]

    for index, item in enumerate(items, start=1):
        lines.extend(
            [
                f"## {index}. {item['id']}",
                "",
                f"**Question:** {item['question']}",
                "",
                f"**Retrieval behavior:** `{item['retrieval_behavior']}`",
                "",
                "**Retrieved articles:** "
                + (", ".join(map(str, item["retrieved_article_numbers"])) or "None"),
                "",
                f"**Generation status:** `{item['generation_status']}`",
                "",
                "**Answer:**",
                "",
                item["answer_ar"] or "_(empty)_",
                "",
                "**Cited articles:** "
                + (", ".join(map(str, item["cited_article_numbers"])) or "None"),
                "",
                "**Limitations/warnings:** "
                + ("; ".join(item["warnings"]) or "None"),
                "",
                "### Manual checklist",
                "",
                "- [ ] The answer directly addresses the question.",
                "- [ ] Every requested part is answered.",
                "- [ ] Legal actors are correct.",
                "- [ ] Conditions and exceptions are preserved.",
                "- [ ] Numbers, percentages, periods, and calculations are correct.",
                "- [ ] Only necessary retrieved articles are used.",
                "- [ ] Every legal sentence has an inline citation.",
                "- [ ] The Arabic wording is clear and concise.",
                "",
                "**Reviewer notes:**",
                "",
                "---",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    manifest = json.loads(
        args.questions.read_text(encoding="utf-8-sig")
    )
    cases = list(manifest.get("questions", []))
    selected = {
        str(value).strip().upper()
        for value in args.case
        if str(value).strip()
    }
    if selected:
        cases = [case for case in cases if case["id"].upper() in selected]

    generator = GroundedAnswerGenerator()
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        path = args.retrieval_dir / case["retrieval_file"]
        retrieval = RetrievalResultV1.model_validate_json(
            path.read_text(encoding="utf-8-sig")
        )
        if retrieval.question != case["question"]:
            raise RuntimeError(
                f"Question mismatch for {case['id']}: "
                f"manifest={case['question']!r}, retrieval={retrieval.question!r}"
            )

        generated = generator.generate(
            retrieval,
            include_debug=args.debug,
        )
        output = generated.model_dump(mode="json")
        item = {
            "id": case["id"],
            "question": case["question"],
            "retrieval_file": case["retrieval_file"],
            "retrieval_behavior": retrieval.decision.behavior,
            "retrieved_article_numbers": article_numbers(retrieval),
            "generation_status": generated.status,
            "answer_ar": generated.answer_ar,
            "cited_article_numbers": generated.cited_article_numbers,
            "warnings": generated.warnings,
            "model": generated.model,
            "usage": generated.usage.model_dump(mode="json"),
            "generation_elapsed_ms": generated.elapsed_ms,
            "debug": output.get("debug") if args.debug else None,
        }
        results.append(item)

        print(
            f"[{index:02d}/{len(cases):02d}] {case['id']} "
            f"route={retrieval.decision.behavior} "
            f"status={generated.status} "
            f"retrieved={item['retrieved_article_numbers']} "
            f"cited={item['cited_article_numbers']}"
        )

    rendered = {
        "review_type": "manual_only",
        "automatic_grading": False,
        "question_count": len(results),
        "generator_model": generator.model,
        "items": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(rendered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(
        markdown_report(results),
        encoding="utf-8",
    )

    print(f"\nJSON: {args.output_json}")
    print(f"Markdown review: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
