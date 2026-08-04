from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import time
from datetime import datetime, timezone
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



def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    key_points = [
        str(value) for value in output.get("key_points", [])
    ]
    warnings = [
        str(value).strip()
        for value in output.get("warnings", [])
        if str(value).strip()
    ]
    failures: list[str] = []

    no_source_heading = not bool(
        re.search(
            r"(^|\n)\s*(المصدر|المراجع)\s*:",
            answer,
        )
    )
    if not no_source_heading:
        failures.append("style: separate source heading")

    policy = case.get("key_points_policy", "optional")
    if policy == "empty":
        key_points_ok = not key_points
    elif policy == "recommended":
        key_points_ok = bool(key_points) or "\n" in answer
    else:
        key_points_ok = True

    if not key_points_ok:
        failures.append(
            f"style: key_points policy is {policy}"
        )

    length_ok = len(answer) <= int(
        case.get("max_answer_chars", 2000)
    )
    if not length_ok:
        failures.append(
            f"style: answer length {len(answer)} exceeds "
            f"{case['max_answer_chars']}"
        )

    no_generic_warning = not any(
        phrase in normalize(" ".join(warnings))
        for phrase in [
            normalize("المعلومات مقتصرة على النص المرفق"),
            normalize("لا يجوز توسيع الاستنتاج خارج النص"),
            normalize("هذه المعلومات لا تغني عن استشارة محام"),
        ]
    )
    if not no_generic_warning:
        failures.append("style: generic warning")

    passed = all(
        [
            no_source_heading,
            key_points_ok,
            length_ok,
            no_generic_warning,
        ]
    )
    return (
        passed,
        failures,
        {
            "no_source_heading": no_source_heading,
            "key_points_policy_ok": key_points_ok,
            "length_ok": length_ok,
            "no_generic_warning": no_generic_warning,
            "answer_chars": len(answer),
            "warnings": warnings,
        },
    )


def fact_checks(
    case: dict[str, Any],
    output: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    answer = str(output.get("answer_ar", ""))
    points = [
        str(value) for value in output.get("key_points", [])
    ]
    combined = normalize(" ".join([answer, *points]))
    failures: list[str] = []

    required_results = []
    for required in case.get("required_facts", []):
        passed = any(
            tokens_match(combined, alternative)
            for alternative in required["any_of"]
        )
        required_results.append(
            {
                "name": required["name"],
                "passed": passed,
            }
        )
        if not passed:
            failures.append(
                f"missing fact: {required['name']}"
            )

    forbidden_results = []
    for item in case.get("forbidden_patterns", []):
        matched = bool(
            re.search(
                item["pattern"],
                combined,
                flags=re.IGNORECASE,
            )
        )
        forbidden_results.append(
            {
                "name": item["name"],
                "matched": matched,
            }
        )
        if matched:
            failures.append(
                f"forbidden claim: {item['name']}"
            )

    return (
        {
            "facts_passed": sum(
                item["passed"] for item in required_results
            ),
            "facts_total": len(required_results),
            "fact_checks": required_results,
            "forbidden_claims_ok": not any(
                item["matched"]
                for item in forbidden_results
            ),
            "forbidden_checks": forbidden_results,
        },
        failures,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Perform the first sealed generation run on the fresh unseen "
            "holdout. Retrieval is never executed by this script."
        )
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--allow-rerun",
        action="store_true",
        help=(
            "Permit overwriting the first-run protection. Never use this "
            "for the initial unseen measurement."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seal_path = args.output.with_suffix(
        args.output.suffix + ".seal.json"
    )

    if (
        (args.output.exists() or seal_path.exists())
        and not args.allow_rerun
    ):
        raise RuntimeError(
            "A sealed result already exists. Refusing to overwrite the "
            "fresh unseen first run."
        )

    benchmark = json.loads(
        args.benchmark.read_text(encoding="utf-8-sig")
    )
    if benchmark.get("input_mode") != (
        "fresh_unseen_frozen_actual_retrieval"
    ):
        raise RuntimeError(
            "Benchmark is not a fresh unseen frozen-retrieval snapshot."
        )

    generator = GroundedAnswerGenerator()
    generator_path = Path(
        inspect.getfile(GroundedAnswerGenerator)
    )
    started_at = datetime.now(timezone.utc).isoformat()

    case_results: list[dict[str, Any]] = []

    for index, case in enumerate(
        benchmark["cases"],
        start=1,
    ):
        retrieval_path = (
            args.retrieval_dir / case["retrieval_file"]
        )
        actual_hash = sha256_file(retrieval_path)
        if actual_hash != case["retrieval_sha256"]:
            raise RuntimeError(
                f"Retrieval hash mismatch for {case['id']}."
            )

        retrieval = RetrievalResultV1.model_validate_json(
            retrieval_path.read_text(encoding="utf-8-sig")
        )
        if retrieval.question != case["question"]:
            raise RuntimeError(
                f"Question mismatch for {case['id']}."
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
        status_ok = (
            output["status"] == expected_output_status
        )
        if not status_ok:
            failures.append(
                "status: expected "
                f"{expected_output_status}, "
                f"got {output['status']}"
            )

        actual_articles = {
            int(value)
            for value in retrieval.diagnostics.article_numbers
        }
        structured_citations = {
            int(value)
            for value in output.get(
                "cited_article_numbers",
                [],
            )
        }
        inline_citations = {
            int(value.translate(ARABIC_DIGITS))
            for value in INLINE_CITATION_RE.findall(
                str(output.get("answer_ar", ""))
                + " "
                + " ".join(
                    str(value)
                    for value in output.get(
                        "key_points",
                        [],
                    )
                )
            )
        }

        citation_safety = (
            structured_citations.issubset(actual_articles)
            and inline_citations.issubset(actual_articles)
        )
        if not citation_safety:
            failures.append(
                "citation safety: cited article absent from retrieval"
            )

        style_ok, style_failures, style_details = (
            style_checks(case, output)
        )
        failures.extend(style_failures)

        expected_behavior = str(
            case["expected_behavior"]
        )
        context_state = case["actual_retrieval"][
            "context_state"
        ]
        generation_evaluable = (
            context_state == "complete"
            and retrieval.decision.behavior
            == expected_behavior
        )

        required_citations_ok: bool | None = None
        details = {
            "facts_passed": 0,
            "facts_total": 0,
            "fact_checks": [],
            "forbidden_claims_ok": True,
            "forbidden_checks": [],
        }
        strict_pass: bool | None = None

        if generation_evaluable:
            if expected_behavior == "retrieve":
                required = {
                    int(value)
                    for value in case.get(
                        "articles",
                        [],
                    )
                }
                required_citations_ok = (
                    output["status"] == "generated"
                    and structured_citations == required
                    and inline_citations == required
                )
                if not required_citations_ok:
                    failures.append(
                        "required citations: expected exactly "
                        f"{sorted(required)}, structured="
                        f"{sorted(structured_citations)}, inline="
                        f"{sorted(inline_citations)}"
                    )

                if output["status"] == "generated":
                    details, fact_failures = fact_checks(
                        case,
                        output,
                    )
                    failures.extend(fact_failures)

                strict_pass = all(
                    [
                        status_ok,
                        output["status"] == "generated",
                        output.get("grounded") is True,
                        citation_safety,
                        bool(required_citations_ok),
                        (
                            details["facts_passed"]
                            == details["facts_total"]
                        ),
                        details["forbidden_claims_ok"],
                        style_ok,
                    ]
                )
            else:
                required_citations_ok = (
                    not structured_citations
                    and not inline_citations
                )
                model_not_called = True
                if args.debug:
                    model_not_called = (
                        output.get("debug", {}).get(
                            "model_called"
                        )
                        is False
                    )
                answer_present = bool(
                    str(output.get("answer_ar", "")).strip()
                )

                strict_pass = all(
                    [
                        status_ok,
                        required_citations_ok,
                        output.get("grounded") is False,
                        answer_present,
                        model_not_called,
                        style_ok,
                    ]
                )
                if not model_not_called:
                    failures.append(
                        "routing: model should not be called"
                    )
                if not answer_present:
                    failures.append(
                        "routing: deterministic answer is empty"
                    )

        classification = (
            "generation_pass"
            if strict_pass is True
            else (
                "generation_failure"
                if strict_pass is False
                else "retrieval_limited"
            )
        )

        case_results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_behavior": expected_behavior,
                "required_articles": case.get(
                    "articles",
                    [],
                ),
                "actual_retrieval": (
                    case["actual_retrieval"]
                ),
                "generation_evaluable": (
                    generation_evaluable
                ),
                "classification": classification,
                "output": output,
                "evaluation": {
                    "strict_generation_pass": strict_pass,
                    "failures": failures,
                    "checks": {
                        "status_ok_given_actual_retrieval": (
                            status_ok
                        ),
                        "citation_safety": citation_safety,
                        "required_citations_ok": (
                            required_citations_ok
                        ),
                        **details,
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
            f"[{index:02d}/20] {case['id']} "
            f"{verdict} "
            f"route={retrieval.decision.behavior} "
            f"articles="
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
    passed = sum(
        item["evaluation"]["strict_generation_pass"]
        is True
        for item in evaluable
    )
    retrieval_limited = [
        item
        for item in case_results
        if not item["generation_evaluable"]
    ]

    def percent(value: int, denominator: int) -> float:
        if not denominator:
            return 0.0
        return round(100.0 * value / denominator, 4)

    total_facts = sum(
        item["evaluation"]["checks"]["facts_total"]
        for item in evaluable
    )
    passed_facts = sum(
        item["evaluation"]["checks"]["facts_passed"]
        for item in evaluable
    )

    route_correct = sum(
        item["actual_retrieval"]["behavior"]
        == item["expected_behavior"]
        for item in case_results
    )
    context_complete = sum(
        item["actual_retrieval"]["context_state"]
        == "complete"
        for item in case_results
    )
    citation_safe = sum(
        item["evaluation"]["checks"]["citation_safety"]
        for item in case_results
    )

    metrics = {
        "case_count": total,
        "retrieval_route_accuracy": percent(
            route_correct,
            total,
        ),
        "retrieval_context_complete_cases": (
            context_complete
        ),
        "retrieval_context_complete_rate": percent(
            context_complete,
            total,
        ),
        "retrieval_limited_cases": len(
            retrieval_limited
        ),
        "generation_evaluable_cases": len(evaluable),
        "generation_strict_passed": passed,
        "generation_strict_accuracy_on_complete_context": (
            percent(passed, len(evaluable))
        ),
        "overall_pipeline_usable_rate": percent(
            passed,
            total,
        ),
        "citation_safety_accuracy": percent(
            citation_safe,
            total,
        ),
        "required_fact_coverage_on_complete_context": (
            percent(passed_facts, total_facts)
        ),
        "average_generation_wall_latency_ms": round(
            sum(
                item["wall_elapsed_ms"]
                for item in case_results
            )
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

    completed_at = datetime.now(timezone.utc).isoformat()
    result = {
        "benchmark_name": benchmark["benchmark_name"],
        "benchmark_version": benchmark["benchmark_version"],
        "run_kind": "first_sealed_unseen_generation_run",
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "input_mode": benchmark["input_mode"],
        "generator_model": generator.model,
        "generator_source": str(generator_path),
        "generator_sha256": sha256_file(generator_path),
        "benchmark_sha256": sha256_file(args.benchmark),
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

    retrieval_hashes = {
        case["id"]: case["retrieval_sha256"]
        for case in benchmark["cases"]
    }
    seal = {
        "sealed": True,
        "run_kind": result["run_kind"],
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "output_file": str(args.output),
        "output_sha256": sha256_file(args.output),
        "benchmark_file": str(args.benchmark),
        "benchmark_sha256": result["benchmark_sha256"],
        "generator_source": str(generator_path),
        "generator_sha256": result["generator_sha256"],
        "generator_model": generator.model,
        "retrieval_file_hashes": retrieval_hashes,
        "rule": (
            "Preserve this first-run output unchanged. Do not tune the "
            "generator against this holdout and then report a rerun as "
            "unseen performance."
        ),
    }
    seal_path.write_text(
        json.dumps(seal, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFresh unseen sealed generation summary")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nResult: {args.output}")
    print(f"Seal:   {seal_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
