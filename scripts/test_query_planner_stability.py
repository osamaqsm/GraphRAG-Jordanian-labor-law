from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.legal_query_planner import LegalQueryPlanner


CASES = (
    {
        "id": "R01",
        "expected_decision": "retrieve",
        "question": "إذا منحني عقد العمل حقاً أفضل من الحد الموجود في قانون العمل، ثم تضمن بنداً أتنازل فيه عن هذا الحق، أي حكم يطبق؟",
    },
    {
        "id": "R02",
        "expected_decision": "retrieve",
        "question": "ابتكر عامل منتجاً مرتبطاً بعمل المنشأة مستخدماً أدواتها وخبراتها؛ لمن تكون الملكية الفكرية إذا لم يوجد اتفاق خطي مختلف؟",
    },
    {
        "id": "R03",
        "expected_decision": "retrieve",
        "question": "مؤسسة تشغّل اثني عشر عاملاً، هل يلزمها نظام داخلي للعمل، وما الموضوعات التي يجب أن يتضمنها ومتى يصبح نافذاً؟",
    },
    {
        "id": "R04",
        "expected_decision": "retrieve",
        "question": "هل يمكن تعميم شروط اتفاق عمل جماعي نافذ على بقية منشآت وعمال قطاع معين، ومن يملك إصدار هذا القرار وما الشرط الزمني؟",
    },
    {
        "id": "R05",
        "expected_decision": "retrieve",
        "question": "وقعت ورقه بتقول اني متنازل عن الاجازه السنويه كلها، هل هالتنازل معتَبر ولا باطل؟",
    },
    {
        "id": "R06",
        "expected_decision": "retrieve",
        "question": "في شغل معيّن قالوا ما بقدر أبلّش قبل فحص لياقة طبي؛ مين بحدد الأشغال اللي لازم إلها هالفحص وكيف بنعلن عنها؟",
    },
    {
        "id": "R07",
        "expected_decision": "retrieve",
        "question": "العامل انصاب وهو تحت تأثير مخدر أو بسبب إهمال جسيم منه؛ هل بروح حقه بالتعويض؟ وشو بصير لو الإصابة سببت وفاة أو عجز دائم 30% أو أكثر؟",
    },
    {
        "id": "S01",
        "expected_decision": "clarify",
        "question": "النقابة اتخذت قراراً ضدي، هل قرارها قانوني؟",
    },
    {
        "id": "S02",
        "expected_decision": "clarify",
        "question": "صار نزاع عمالي جماعي في المؤسسة، شو الخطوة القانونية الجاية؟",
    },
    {
        "id": "S03",
        "expected_decision": "clarify",
        "question": "في مخالفة سلامة داخل مكان العمل، مين المسؤول وشو العقوبة؟",
    },
    {
        "id": "S04",
        "expected_decision": "abstain",
        "question": "اشتريت هاتفاً جديداً وظهر فيه عيب بعد أسبوع، هل يحق لي استبداله أو استرداد ثمنه؟",
    },
    {
        "id": "S05",
        "expected_decision": "abstain",
        "question": "أريد تسجيل شركة ذات مسؤولية محدودة وتحديد حصص الشركاء، ما الإجراءات المطلوبة؟",
    },
    {
        "id": "S06",
        "expected_decision": "abstain",
        "question": "الجامعة أوقفت تسجيلي بسبب المعدل، كيف أعترض على القرار الأكاديمي؟",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat the planner routing cases to measure stability."
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--output",
        default="/tmp/query_planner_stability_results.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")

    planner = LegalQueryPlanner(get_settings())
    if not planner.enabled:
        raise SystemExit(
            "Enable OPENAI_QUERY_PLANNER_ENABLED=true before this test."
        )

    results: list[dict] = []
    per_case_decisions: dict[str, Counter] = defaultdict(Counter)

    for repetition in range(1, args.repetitions + 1):
        for case in CASES:
            try:
                plan = planner.plan(case["question"])
                if plan is None:
                    observed = "fallback"
                    payload = None
                else:
                    observed = plan.decision
                    payload = plan.model_dump()

                passed = observed == case["expected_decision"]
                per_case_decisions[case["id"]][observed] += 1
                results.append(
                    {
                        "repetition": repetition,
                        **case,
                        "observed_decision": observed,
                        "pass": passed,
                        "plan": payload,
                        "error": None,
                    }
                )
            except Exception as exc:
                per_case_decisions[case["id"]]["error"] += 1
                results.append(
                    {
                        "repetition": repetition,
                        **case,
                        "observed_decision": "error",
                        "pass": False,
                        "plan": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    total = len(results)
    passed = sum(1 for item in results if item["pass"])
    stable_cases = 0
    case_summary: list[dict] = []

    for case in CASES:
        counts = per_case_decisions[case["id"]]
        expected_count = counts[case["expected_decision"]]
        stable = expected_count == args.repetitions
        stable_cases += int(stable)
        case_summary.append(
            {
                "id": case["id"],
                "expected_decision": case["expected_decision"],
                "decision_counts": dict(counts),
                "stable_and_correct": stable,
            }
        )

    document = {
        "test_name": "Stage 7.7-D Bidirectional Route Stability Test",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "planner_enabled": planner.enabled,
        "planner_model": planner.model,
        "reasoning_effort": planner.reasoning_effort,
        "verify_non_answer": planner.verify_non_answer,
        "verify_low_confidence_retrieve": (
            planner.verify_low_confidence_retrieve
        ),
        "retrieve_verification_below": (
            planner.retrieve_verification_below
        ),
        "route_confidence": planner.route_confidence,
        "route_verification_confidence": (
            planner.non_answer_verification_confidence
        ),
        "summary": {
            "cases": len(CASES),
            "repetitions": args.repetitions,
            "total_plans": total,
            "passed": passed,
            "accuracy": round(passed / total, 6) if total else 0.0,
            "stable_correct_cases": stable_cases,
            "all_cases_stable_and_correct": stable_cases == len(CASES),
        },
        "case_summary": case_summary,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(document["summary"], ensure_ascii=False, indent=2))
    print(f"Saved JSON result to: {output_path}")

    if not document["summary"]["all_cases_stable_and_correct"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
