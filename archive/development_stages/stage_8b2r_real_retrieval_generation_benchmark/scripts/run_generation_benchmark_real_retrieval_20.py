from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from app.grounded_answer_generator import GroundedAnswerGenerator
from app.retrieval_contract import RetrievalResultV1


ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
DIACRITICS_RE = re.compile(
    r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]"
)
INLINE_CITATION_RE = re.compile(
    r"\[\s*المادة\s+([0-9٠-٩]+)\s*\]"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    value = str(text).translate(ARABIC_DIGITS).lower()
    value = DIACRITICS_RE.sub("", value)
    value = re.sub(r"[إأآٱ]", "ا", value)
    value = (
        value.replace("ى", "ي")
        .replace("ة", "ه")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    value = value.replace("٪", "%")
    value = re.sub(r"\s*%\s*", "%", value)
    value = re.sub(r"[^\u0600-\u06ff0-9a-z%]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokens_match(text: str, tokens: list[str]) -> bool:
    return all(normalize(token) in text for token in tokens)


def expected_status(retrieval: RetrievalResultV1) -> str:
    behavior = retrieval.decision.behavior
    if behavior == "clarify":
        return "clarification_required"
    if behavior == "abstain":
        return "out_of_scope"
    if not retrieval.articles:
        return "insufficient_evidence"
    return "generated"


def style_checks(
    case: dict[str, Any],
    output: dict[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    answer = str(output.get("answer_ar", ""))
    key_points = [str(value) for value in output.get("key_points", [])]
    warnings = [
        str(value).strip()
        for value in output.get("warnings", [])
        if str(value).strip()
    ]
    failures: list[str] = []

    no_source_heading = not bool(
        re.search(r"(^|\n)\s*(المصدر|المراجع)\s*:", answer)
    )
    if not no_source_heading:
        failures.append("style: separate source/references heading")

    no_generic_boilerplate = not any(
        phrase in normalize(" ".join(warnings))
        for phrase in [
            normalize("المعلومات مقتصرة على النص المرفق"),
            normalize("لا يجوز توسيع الاستنتاج خارج النص"),
            normalize("هذه المعلومات لا تغني عن استشارة محام"),
        ]
    )
    if not no_generic_boilerplate:
        failures.append("style: generic limitation/disclaimer")

    policy = case.get("key_points_policy", "optional")
    if policy == "empty":
        key_points_ok = len(key_points) == 0
    elif policy == "recommended":
        key_points_ok = len(key_points) > 0 or "\n" in answer
    else:
        key_points_ok = True
    if not key_points_ok:
        failures.append(f"style: key_points policy is {policy}")

    length_ok = len(answer) <= int(
        case.get("max_answer_chars", 2000)
    )
    if not length_ok:
        failures.append(
            f"style: answer length {len(answer)} exceeds "
            f"{case['max_answer_chars']}"
        )

    passed = all(
        [
            no_source_heading,
            no_generic_boilerplate,
            key_points_ok,
            length_ok,
        ]
    )
    return (
        passed,
        failures,
        {
            "no_source_heading": no_source_heading,
            "no_generic_boilerplate": no_generic_boilerplate,
            "key_points_policy_ok": key_points_ok,
            "length_ok": length_ok,
            "answer_chars": len(answer),
            "warnings": warnings,
        },
    )


def fact_and_claim_checks(
    case: dict[str, Any],
    output: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    answer = str(output.get("answer_ar", ""))
    key_points = [str(value) for value in output.get("key_points", [])]
    combined = normalize(" ".join([answer, *key_points]))
    failures: list[str] = []

    fact_checks = []
    for required in case.get("required_facts", []):
        passed = any(
            tokens_match(combined, alternative)
            for alternative in required["any_of"]
        )
        fact_checks.append(
            {"name": required["name"], "passed": passed}
        )
        if not passed:
            failures.append(f"missing fact: {required['name']}")

    forbidden_checks = []
    for item in case.get("forbidden_patterns", []):
        matched = bool(
            re.search(
                item["pattern"],
                combined,
                flags=re.IGNORECASE,
            )
        )
        forbidden_checks.append(
            {"name": item["name"], "matched": matched}
        )
        if matched:
            failures.append(
                f"forbidden claim: {item['name']}"
            )

    return (
        {
            "facts_passed": sum(
                item["passed"] for item in fact_checks
            ),
            "facts_total": len(fact_checks),
            "fact_checks": fact_checks,
            "forbidden_claims_ok": not any(
                item["matched"] for item in forbidden_checks
            ),
            "forbidden_checks": forbidden_checks,
        },
        failures,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run generation only from a frozen snapshot of 20 actual "
            "retrieval.v1 outputs."
        )
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only a case ID, for example --case G09.",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help=(
            "Exit with code 1 when any generation-evaluable case fails."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = json.loads(
        args.benchmark.read_text(encoding="utf-8-sig")
    )
    if benchmark.get("input_mode") != (
        "frozen_actual_retrieval_outputs"
    ):
        raise RuntimeError(
            "This runner accepts only a benchmark built from actual frozen "
            "retrieval outputs."
        )

    selected = {
        str(value).strip().upper()
        for value in args.case
        if str(value).strip()
    }
    cases = [
        case
        for case in benchmark["cases"]
        if not selected or case["id"].upper() in selected
    ]
    if not cases:
        raise RuntimeError("No benchmark cases selected.")

    generator = GroundedAnswerGenerator()
    case_results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        retrieval_path = (
            args.retrieval_dir / case["retrieval_file"]
        )
        actual_hash = sha256_file(retrieval_path)
        expected_hash = case["retrieval_sha256"]
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Frozen retrieval hash mismatch for {case['id']}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

        retrieval = RetrievalResultV1.model_validate_json(
            retrieval_path.read_text(encoding="utf-8-sig")
        )
        if retrieval.question != case["question"]:
            raise RuntimeError(
                f"Question mismatch in {case['id']}."
            )

        started = time.perf_counter()
        generated = generator.generate(
            retrieval,
            include_debug=args.debug,
        )
        wall_ms = round(
            (time.perf_counter() - started) * 1000
        )
        output = generated.model_dump(mode="json")

        failures: list[str] = []
        expected_output_status = expected_status(retrieval)
        status_ok = output["status"] == expected_output_status
        if not status_ok:
            failures.append(
                "status: expected "
                f"{expected_output_status}, got {output['status']}"
            )

        actual_article_numbers = {
            int(value)
            for value in retrieval.diagnostics.article_numbers
        }
        cited_article_numbers = {
            int(value)
            for value in output.get(
                "cited_article_numbers",
                [],
            )
        }
        citation_subset_ok = cited_article_numbers.issubset(
            actual_article_numbers
        )
        if not citation_subset_ok:
            failures.append(
                "citation safety: generated answer cited an article absent "
                "from the frozen retrieval output"
            )

        inline_numbers = {
            int(value.translate(ARABIC_DIGITS))
            for value in INLINE_CITATION_RE.findall(
                str(output.get("answer_ar", ""))
                + " "
                + " ".join(
                    str(value)
                    for value in output.get("key_points", [])
                )
            )
        }
        inline_subset_ok = inline_numbers.issubset(
            actual_article_numbers
        )
        if not inline_subset_ok:
            failures.append(
                "inline citation safety: answer contains an article absent "
                "from the frozen retrieval output"
            )

        style_ok, style_failures, style_details = style_checks(
            case,
            output,
        )
        failures.extend(style_failures)

        context_state = case["actual_retrieval"][
            "context_state"
        ]
        generation_evaluable = (
            context_state == "complete"
            and retrieval.decision.behavior
            == case.get("expected_behavior", "retrieve")
        )

        fact_details = {
            "facts_passed": 0,
            "facts_total": 0,
            "fact_checks": [],
            "forbidden_claims_ok": True,
            "forbidden_checks": [],
        }
        required_citations_ok: bool | None = None
        strict_generation_pass: bool | None = None

        if generation_evaluable:
            expected_behavior = case.get(
                "expected_behavior",
                "retrieve",
            )
            if expected_behavior == "retrieve":
                required = {
                    int(value)
                    for value in case.get("articles", [])
                }
                required_citations_ok = (
                    output["status"] == "generated"
                    and cited_article_numbers == required
                    and inline_numbers == required
                )
                if not required_citations_ok:
                    failures.append(
                        "required citations: expected exactly "
                        f"{sorted(required)}, got structured="
                        f"{sorted(cited_article_numbers)}, inline="
                        f"{sorted(inline_numbers)}"
                    )

                if output["status"] == "generated":
                    (
                        fact_details,
                        fact_failures,
                    ) = fact_and_claim_checks(case, output)
                    failures.extend(fact_failures)

                strict_generation_pass = all(
                    [
                        status_ok,
                        output["status"] == "generated",
                        output.get("grounded") is True,
                        citation_subset_ok,
                        inline_subset_ok,
                        bool(required_citations_ok),
                        (
                            fact_details["facts_passed"]
                            == fact_details["facts_total"]
                        ),
                        fact_details["forbidden_claims_ok"],
                        style_ok,
                    ]
                )
            else:
                required_citations_ok = True
                strict_generation_pass = all(
                    [
                        status_ok,
                        not output.get(
                            "cited_article_numbers",
                            [],
                        ),
                        style_ok,
                    ]
                )

        classification = (
            "generation_pass"
            if strict_generation_pass is True
            else (
                "generation_failure"
                if strict_generation_pass is False
                else "retrieval_limited"
            )
        )

        case_results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "retrieval_file": case["retrieval_file"],
                "retrieval_sha256": expected_hash,
                "expected_behavior": case.get(
                    "expected_behavior",
                    "retrieve",
                ),
                "actual_retrieval": case["actual_retrieval"],
                "generation_evaluable": generation_evaluable,
                "classification": classification,
                "output": output,
                "evaluation": {
                    "strict_generation_pass": (
                        strict_generation_pass
                    ),
                    "failures": failures,
                    "checks": {
                        "status_ok_given_actual_retrieval": (
                            status_ok
                        ),
                        "citation_subset_of_actual_retrieval": (
                            citation_subset_ok
                        ),
                        "inline_citation_subset_of_actual_retrieval": (
                            inline_subset_ok
                        ),
                        "required_citations_ok": (
                            required_citations_ok
                        ),
                        **fact_details,
                        "style_ok": style_ok,
                        **style_details,
                    },
                },
                "wall_elapsed_ms": wall_ms,
            }
        )

        verdict = (
            "PASS"
            if classification == "generation_pass"
            else (
                "RETRIEVAL-LIMITED"
                if classification == "retrieval_limited"
                else "FAIL"
            )
        )
        print(
            f"[{index:02d}/{len(cases):02d}] "
            f"{case['id']} {verdict} "
            f"retrieved="
            f"{retrieval.diagnostics.article_numbers}"
        )
        for failure in failures:
            print(f"  - {failure}")

    total = len(case_results)
    evaluable = [
        item
        for item in case_results
        if item["generation_evaluable"]
    ]
    retrieval_limited = [
        item
        for item in case_results
        if not item["generation_evaluable"]
    ]
    generation_passed = sum(
        item["evaluation"]["strict_generation_pass"] is True
        for item in evaluable
    )

    complete_count = sum(
        item["actual_retrieval"]["context_state"]
        == "complete"
        for item in case_results
    )
    exact_retrieval_set_count = sum(
        (
            set(item["actual_retrieval"]["article_numbers"])
            == set(
                next(
                    case["articles"]
                    for case in cases
                    if case["id"] == item["id"]
                )
            )
        )
        for item in case_results
        if item["actual_retrieval"]["behavior"] == "retrieve"
    )

    total_facts = sum(
        item["evaluation"]["checks"]["facts_total"]
        for item in evaluable
    )
    passed_facts = sum(
        item["evaluation"]["checks"]["facts_passed"]
        for item in evaluable
    )

    def percent(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return round(100.0 * numerator / denominator, 4)

    metrics = {
        "case_count": total,
        "retrieval_context_complete_cases": complete_count,
        "retrieval_context_complete_rate": percent(
            complete_count,
            total,
        ),
        "retrieval_exact_article_set_cases": (
            exact_retrieval_set_count
        ),
        "retrieval_exact_article_set_rate": percent(
            exact_retrieval_set_count,
            total,
        ),
        "retrieval_limited_cases": len(retrieval_limited),
        "generation_evaluable_cases": len(evaluable),
        "generation_strict_passed": generation_passed,
        "generation_strict_accuracy_on_complete_context": percent(
            generation_passed,
            len(evaluable),
        ),
        "overall_pipeline_usable_rate": percent(
            generation_passed,
            total,
        ),
        "generator_status_accuracy_given_actual_retrieval": percent(
            sum(
                item["evaluation"]["checks"][
                    "status_ok_given_actual_retrieval"
                ]
                for item in case_results
            ),
            total,
        ),
        "citation_safety_accuracy": percent(
            sum(
                item["evaluation"]["checks"][
                    "citation_subset_of_actual_retrieval"
                ]
                and item["evaluation"]["checks"][
                    "inline_citation_subset_of_actual_retrieval"
                ]
                for item in case_results
            ),
            total,
        ),
        "required_fact_coverage_on_complete_context": percent(
            passed_facts,
            total_facts,
        ),
        "average_generation_wall_latency_ms": round(
            sum(item["wall_elapsed_ms"] for item in case_results)
            / total,
            2,
        ),
        "total_input_tokens": sum(
            int(item["output"]["usage"]["input_tokens"])
            for item in case_results
        ),
        "total_output_tokens": sum(
            int(item["output"]["usage"]["output_tokens"])
            for item in case_results
        ),
        "total_tokens": sum(
            int(item["output"]["usage"]["total_tokens"])
            for item in case_results
        ),
        "retrieval_limited_case_ids": [
            item["id"] for item in retrieval_limited
        ],
        "generation_failure_case_ids": [
            item["id"]
            for item in evaluable
            if item["evaluation"][
                "strict_generation_pass"
            ]
            is False
        ],
    }

    result = {
        "benchmark_name": benchmark["benchmark_name"],
        "benchmark_version": benchmark["benchmark_version"],
        "input_mode": benchmark["input_mode"],
        "generator_model": generator.model,
        "metrics": metrics,
        "cases": case_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nReal-retrieval generation benchmark summary")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.output}")

    any_generation_failure = bool(
        metrics["generation_failure_case_ids"]
    )
    if args.strict_exit and any_generation_failure:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
