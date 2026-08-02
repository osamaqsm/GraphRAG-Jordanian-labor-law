from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.legal_query_planner import LegalQueryPlanner


CASES = (
    {
        "id": "F01",
        "question": "إذا منحني عقد العمل حقاً أفضل من الحد الموجود في قانون العمل، ثم تضمن بنداً أتنازل فيه عن هذا الحق، أي حكم يطبق؟",
        "expected_decision": "retrieve",
        "expected_articles": [4],
    },
    {
        "id": "F04",
        "question": "ابتكر عامل منتجاً مرتبطاً بعمل المنشأة مستخدماً أدواتها وخبراتها؛ لمن تكون الملكية الفكرية إذا لم يوجد اتفاق خطي مختلف؟",
        "expected_decision": "retrieve",
        "expected_articles": [20],
    },
    {
        "id": "F05",
        "question": "مؤسسة تشغّل اثني عشر عاملاً، هل يلزمها نظام داخلي للعمل، وما الموضوعات التي يجب أن يتضمنها ومتى يصبح نافذاً؟",
        "expected_decision": "retrieve",
        "expected_articles": [55],
    },
    {
        "id": "F09",
        "question": "هل يمكن تعميم شروط اتفاق عمل جماعي نافذ على بقية منشآت وعمال قطاع معين، ومن يملك إصدار هذا القرار وما الشرط الزمني؟",
        "expected_decision": "retrieve",
        "expected_articles": [43],
    },
    {
        "id": "F11",
        "question": "وقعت ورقه بتقول اني متنازل عن الاجازه السنويه كلها، هل هالتنازل معتَبر ولا باطل؟",
        "expected_decision": "retrieve",
        "expected_articles": [64],
    },
    {
        "id": "F15",
        "question": "في شغل معيّن قالوا ما بقدر أبلّش قبل فحص لياقة طبي؛ مين بحدد الأشغال اللي لازم إلها هالفحص وكيف بنعلن عنها؟",
        "expected_decision": "retrieve",
        "expected_articles": [83],
    },
    {
        "id": "F16",
        "question": "العامل انصاب وهو تحت تأثير مخدر أو بسبب إهمال جسيم منه؛ هل بروح حقه بالتعويض؟ وشو بصير لو الإصابة سببت وفاة أو عجز دائم 30% أو أكثر؟",
        "expected_decision": "retrieve",
        "expected_articles": [94],
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect raw planner decisions for the seven in-scope questions "
            "incorrectly blocked in the Stage 7.7-B regression run."
        )
    )
    parser.add_argument(
        "--output",
        default="/tmp/query_planner_false_blocks.json",
        help="JSON output path inside the container.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    planner = LegalQueryPlanner(get_settings())
    if not planner.enabled:
        raise SystemExit(
            "Enable OPENAI_QUERY_PLANNER_ENABLED=true before this diagnostic."
        )

    results: list[dict] = []
    counts = {
        "retrieve": 0,
        "clarify": 0,
        "abstain": 0,
        "fallback": 0,
    }

    for case in CASES:
        try:
            plan = planner.plan(case["question"])
            if plan is None:
                counts["fallback"] += 1
                results.append(
                    {
                        **case,
                        "planner_status": "fallback",
                        "accepted_by_route_threshold": False,
                        "decision_correct": False,
                        "plan": None,
                        "error": None,
                    }
                )
                continue

            payload = plan.model_dump()
            decision = plan.decision
            counts[decision] += 1

            accepted = plan.confidence >= planner.route_confidence
            decision_correct = decision == case["expected_decision"]

            results.append(
                {
                    **case,
                    "planner_status": "success",
                    "accepted_by_route_threshold": accepted,
                    "decision_correct": decision_correct,
                    "plan": payload,
                    "error": None,
                }
            )
        except Exception as exc:
            counts["fallback"] += 1
            results.append(
                {
                    **case,
                    "planner_status": "error",
                    "accepted_by_route_threshold": False,
                    "decision_correct": False,
                    "plan": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    successful = [
        item for item in results if item["planner_status"] == "success"
    ]

    document = {
        "test_name": "Stage 7.7-B False-Block Planner Diagnostic",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "planner_enabled": planner.enabled,
        "planner_model": planner.model,
        "reasoning_effort": planner.reasoning_effort,
        "route_confidence": planner.route_confidence,
        "retrieve_override_confidence": planner.retrieve_override_confidence,
        "summary": {
            "questions_requested": len(CASES),
            "questions_completed": len(results),
            "successful_plans": len(successful),
            "decision_counts": counts,
            "correct_raw_decisions": sum(
                1 for item in successful if item["decision_correct"]
            ),
            "accepted_plans": sum(
                1 for item in successful
                if item["accepted_by_route_threshold"]
            ),
            "accepted_wrong_non_answers": sum(
                1
                for item in successful
                if item["accepted_by_route_threshold"]
                and not item["decision_correct"]
                and item["plan"]["decision"] in {"clarify", "abstain"}
            ),
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(document, ensure_ascii=False, indent=2))
    print(f"\nSaved JSON result to: {output_path}")


if __name__ == "__main__":
    main()
