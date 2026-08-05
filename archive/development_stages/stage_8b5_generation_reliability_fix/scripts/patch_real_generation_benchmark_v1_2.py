from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create real-retrieval generation benchmark rubric v1.2. "
            "Frozen retrieval JSON files and SHA-256 hashes are preserved."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def add_alternative(
    cases: dict[str, dict[str, Any]],
    case_id: str,
    fact_name: str,
    tokens: list[str],
) -> None:
    case = cases[case_id]
    for fact in case.get("required_facts", []):
        if fact.get("name") == fact_name:
            alternatives = fact.setdefault("any_of", [])
            if tokens not in alternatives:
                alternatives.append(tokens)
            return
    raise KeyError(
        f"Fact {fact_name!r} was not found in case {case_id}."
    )


def main() -> int:
    args = parse_args()
    benchmark = json.loads(
        args.input.read_text(encoding="utf-8-sig")
    )

    if benchmark.get("input_mode") != (
        "frozen_actual_retrieval_outputs"
    ):
        raise RuntimeError(
            "Input must be a frozen actual-retrieval benchmark."
        )
    if len(benchmark.get("cases", [])) != 20:
        raise RuntimeError("Expected exactly 20 cases.")

    cases = {
        str(case["id"]): case
        for case in benchmark["cases"]
    }

    # Arabic morphology and phrasing equivalents observed in the real run.
    add_alternative(
        cases,
        "G06",
        "weekly holiday rate is at least 150 percent",
        ["عطلتك الاسبوعيه", "150%"],
    )
    add_alternative(
        cases,
        "G09",
        "meal and rest time is excluded from work hours",
        ["لا تحسب", "الطعام", "الراحه"],
    )
    add_alternative(
        cases,
        "G14",
        "appointment and termination of union staff",
        ["تعيين", "الموظفين", "انتهاء خدماتهم"],
    )
    add_alternative(
        cases,
        "G15",
        "interpretation removes ambiguity without changing outcome",
        ["لا يخرج القرار", "النتائج"],
    )
    add_alternative(
        cases,
        "G16",
        "indefinite agreement can be ended after two years",
        ["بعد مضي سنتين", "انهائه"],
    )
    add_alternative(
        cases,
        "G20",
        "binding on current and later workers conditionally",
        ["كانوا يعملون", "يوظفون فيما بعد", "اذا ورد"],
    )

    benchmark["benchmark_version"] = "1.2.0"
    benchmark["evaluation_rubric_revision"] = {
        "base_benchmark_version": benchmark.get(
            "benchmark_version",
            "1.1.0",
        ),
        "retrieval_snapshot_changed": False,
        "retrieval_hashes_changed": False,
        "changes": [
            (
                "Added Arabic morphology and phrasing alternatives for "
                "G06, G09, G14, G15, G16, and G20."
            ),
            (
                "No genuine requirement was removed. Extra-article failures "
                "in G10, G16, and G18 remain strict."
            ),
            (
                "The omitted failed-settlement detail in G19 remains a "
                "strict generation failure."
            ),
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            benchmark,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Created: {args.output}")
    print("Retrieval files changed: 0")
    print("Retrieval SHA-256 values changed: 0")
    print("Evaluation rubric version: 1.2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
