from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create evaluation-rubric v1.1 from the frozen real-retrieval "
            "generation benchmark. Retrieval files and their hashes are not "
            "changed."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def add_alternative(
    case: dict[str, Any],
    fact_name: str,
    tokens: list[str],
) -> None:
    for fact in case.get("required_facts", []):
        if fact.get("name") == fact_name:
            alternatives = fact.setdefault("any_of", [])
            if tokens not in alternatives:
                alternatives.append(tokens)
            return
    raise KeyError(
        f"Fact {fact_name!r} was not found in case {case['id']}."
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
            "The input is not a frozen actual-retrieval benchmark."
        )
    if len(benchmark.get("cases", [])) != 20:
        raise RuntimeError("Expected exactly 20 cases.")

    cases = {
        str(case["id"]): case
        for case in benchmark["cases"]
    }

    # G07: the generated answer preserved the condition but used
    # "لا يعزوه إليه" instead of "لا يعزى إليه". This remains a language
    # quality issue, but it must not be counted as a missing legal fact.
    add_alternative(
        cases["G07"],
        "rule requires an unavoidable cause not attributable to employer",
        ["سبب", "لا يعزوه", "صاحب العمل"],
    )

    # G09: Arabic digit and Arabic-word forms are semantically equivalent.
    add_alternative(
        cases["G09"],
        "daily work limit is eight hours",
        ["8 ساعات", "يوم"],
    )

    # G11: accept numbers written as Arabic words and the exact wording used
    # for the prohibition on reducing the fine.
    add_alternative(
        cases["G11"],
        "fine range is 50 to 500 dinars",
        ["خمسين", "خمسمائه", "دينار"],
    )
    add_alternative(
        cases["G11"],
        "fine range is 50 to 500 dinars",
        ["خمسين", "خمسمائة", "دينار"],
    )
    add_alternative(
        cases["G11"],
        "fine may not be reduced below minimum",
        ["لا يجوز", "تخفيض", "حدها الادني"],
    )

    # G16: accept reversed word order and the exact generated formulation.
    # The genuinely missing statement that each party keeps a copy remains
    # required, and Article 41 remains an extra citation failure.
    add_alternative(
        cases["G16"],
        "at least three original copies",
        ["النسخ الثلاث", "اصليه"],
    )
    add_alternative(
        cases["G16"],
        "indefinite agreement can be ended after two years",
        ["غير محدده", "سنتان", "انهائه"],
    )

    benchmark["benchmark_version"] = "1.1.0"
    benchmark["evaluation_rubric_revision"] = {
        "base_benchmark_version": "1.0.0",
        "retrieval_snapshot_changed": False,
        "retrieval_hashes_changed": False,
        "changes": [
            "Arabic digit/word equivalence for G09 and G11.",
            "Arabic morphology and word-order alternatives for G07 and G16.",
            (
                "No genuine legal requirement was removed; G15 and the "
                "remaining G16 omissions are still strict failures."
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
    print("Retrieval JSON files changed: 0")
    print("Retrieval SHA-256 values changed: 0")
    print("Evaluation rubric version: 1.1.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
