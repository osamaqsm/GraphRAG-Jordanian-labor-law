from __future__ import annotations

import argparse
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
INLINE_CITATION_RE = re.compile(r"\[\s*المادة\s+(\d+)\s*\]")


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


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    answer = str(result.get("answer_ar", ""))
    key_points = [str(value) for value in result.get("key_points", [])]
    combined = normalize(" ".join([answer, *key_points]))
    failures: list[str] = []

    status_ok = result.get("status") == "generated"
    if not status_ok:
        failures.append(
            f"status: expected generated, got {result.get('status')}"
        )

    grounded_ok = result.get("grounded") is True
    if not grounded_ok:
        failures.append("grounding: grounded must be true")

    expected_citations = [int(value) for value in case["articles"]]
    actual_citations = [
        int(value) for value in result.get("cited_article_numbers", [])
    ]
    citation_exact = (
        set(actual_citations) == set(expected_citations)
        and len(actual_citations) == len(set(actual_citations))
    )
    if not citation_exact:
        failures.append(
            "citation: expected exactly "
            f"{expected_citations}, got {actual_citations}"
        )

    inline_citations = [
        int(value) for value in INLINE_CITATION_RE.findall(answer)
    ]
    for point in key_points:
        inline_citations.extend(
            int(value) for value in INLINE_CITATION_RE.findall(point)
        )
    inline_set_ok = set(inline_citations) == set(expected_citations)
    if not inline_set_ok:
        failures.append(
            "inline citation: expected "
            f"{expected_citations}, got {sorted(set(inline_citations))}"
        )

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
            re.search(item["pattern"], combined, flags=re.IGNORECASE)
        )
        forbidden_checks.append(
            {"name": item["name"], "matched": matched}
        )
        if matched:
            failures.append(f"forbidden claim: {item['name']}")

    warnings = [
        str(value).strip()
        for value in result.get("warnings", [])
        if str(value).strip()
    ]
    warnings_ok = not warnings
    if not warnings_ok:
        failures.append("style: warnings should be empty for complete evidence")

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

    length_ok = len(answer) <= int(case.get("max_answer_chars", 2000))
    if not length_ok:
        failures.append(
            "style: answer length "
            f"{len(answer)} exceeds {case['max_answer_chars']}"
        )

    style_ok = all(
        [
            warnings_ok,
            no_source_heading,
            no_generic_boilerplate,
            key_points_ok,
            length_ok,
        ]
    )
    facts_passed = sum(item["passed"] for item in fact_checks)
    facts_total = len(fact_checks)
    forbidden_ok = not any(item["matched"] for item in forbidden_checks)

    strict_pass = all(
        [
            status_ok,
            grounded_ok,
            citation_exact,
            inline_set_ok,
            facts_passed == facts_total,
            forbidden_ok,
            style_ok,
        ]
    )

    return {
        "strict_pass": strict_pass,
        "failures": failures,
        "checks": {
            "status_ok": status_ok,
            "grounded_ok": grounded_ok,
            "citation_exact": citation_exact,
            "inline_citation_set_ok": inline_set_ok,
            "facts_passed": facts_passed,
            "facts_total": facts_total,
            "fact_checks": fact_checks,
            "forbidden_claims_ok": forbidden_ok,
            "forbidden_checks": forbidden_checks,
            "style_ok": style_ok,
            "warnings_ok": warnings_ok,
            "no_source_heading": no_source_heading,
            "no_generic_boilerplate": no_generic_boilerplate,
            "key_points_policy_ok": key_points_ok,
            "length_ok": length_ok,
            "answer_chars": len(answer),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the generation-only 20-case benchmark from frozen "
            "retrieval.v1 inputs."
        )
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Request generation diagnostics.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only a named case, for example --case G01.",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Return exit code 1 when any strict case fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = json.loads(
        args.benchmark.read_text(encoding="utf-8-sig")
    )
    selected = {
        value.strip().upper()
        for value in args.case
        if value.strip()
    }

    cases = [
        case
        for case in benchmark["cases"]
        if not selected or case["id"].upper() in selected
    ]
    if not cases:
        raise RuntimeError("No benchmark cases selected.")

    generator = GroundedAnswerGenerator()
    case_results = []

    for index, case in enumerate(cases, start=1):
        input_path = args.retrieval_dir / case["retrieval_file"]
        retrieval_payload = json.loads(
            input_path.read_text(encoding="utf-8-sig")
        )
        retrieval = RetrievalResultV1.model_validate(
            retrieval_payload
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
        evaluation = evaluate_case(case, output)

        case_results.append(
            {
                "id": case["id"],
                "source_case_id": case["source_case_id"],
                "question": case["question"],
                "retrieval_file": case["retrieval_file"],
                "required_articles": case["articles"],
                "reference_answer_ar": case["reference_answer_ar"],
                "output": output,
                "evaluation": evaluation,
                "wall_elapsed_ms": wall_ms,
            }
        )

        verdict = "PASS" if evaluation["strict_pass"] else "FAIL"
        print(
            f"[{index:02d}/{len(cases):02d}] "
            f"{case['id']} {verdict}"
        )
        for failure in evaluation["failures"]:
            print(f"  - {failure}")

    count = len(case_results)
    strict_passed = sum(
        item["evaluation"]["strict_pass"]
        for item in case_results
    )
    total_facts = sum(
        item["evaluation"]["checks"]["facts_total"]
        for item in case_results
    )
    passed_facts = sum(
        item["evaluation"]["checks"]["facts_passed"]
        for item in case_results
    )

    def accuracy(key: str) -> float:
        return round(
            100.0
            * sum(
                item["evaluation"]["checks"][key]
                for item in case_results
            )
            / count,
            4,
        )

    total_input_tokens = sum(
        int(item["output"]["usage"]["input_tokens"])
        for item in case_results
    )
    total_output_tokens = sum(
        int(item["output"]["usage"]["output_tokens"])
        for item in case_results
    )
    total_tokens = sum(
        int(item["output"]["usage"]["total_tokens"])
        for item in case_results
    )
    avg_latency = round(
        sum(item["wall_elapsed_ms"] for item in case_results)
        / count,
        2,
    )

    metrics = {
        "case_count": count,
        "strict_passed": strict_passed,
        "strict_case_accuracy": round(
            100.0 * strict_passed / count,
            4,
        ),
        "status_accuracy": accuracy("status_ok"),
        "grounded_accuracy": accuracy("grounded_ok"),
        "citation_exact_accuracy": accuracy("citation_exact"),
        "inline_citation_accuracy": accuracy(
            "inline_citation_set_ok"
        ),
        "required_fact_coverage": round(
            100.0 * passed_facts / total_facts,
            4,
        ),
        "forbidden_claim_safety": accuracy(
            "forbidden_claims_ok"
        ),
        "style_accuracy": accuracy("style_ok"),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "average_wall_latency_ms": avg_latency,
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
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nGeneration-only benchmark summary")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.output}")
    if args.strict_exit and strict_passed != count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
