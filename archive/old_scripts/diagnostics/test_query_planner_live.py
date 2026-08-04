from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.legal_query_planner import LegalQueryPlanner


QUESTIONS = (
    "ما شروط عقد التدريب ومتى يجوز إنهاؤه إذا أصبح خطراً على صحة المتدرب؟",
    "كم يستطيع صاحب العمل اقتطاعه من الأجر لاسترداد سلفة؟",
    "النقابة اتخذت قراراً ضدي، هل قرارها قانوني؟",
    "صار نزاع عمالي جماعي، شو الخطوة القانونية الجاية؟",
    "اشتريت هاتفاً وظهر فيه عيب، هل أستطيع استبداله؟",
    "الجامعة أوقفت تسجيلي بسبب المعدل، كيف أعترض؟",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the live optional LLM query planner and save valid JSON results."
    )
    parser.add_argument(
        "--output",
        default="/tmp/query_planner_live_results.json",
        help="JSON output path inside the container.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    planner = LegalQueryPlanner(get_settings())

    if not planner.enabled:
        raise SystemExit(
            "Enable OPENAI_QUERY_PLANNER_ENABLED=true before this live test."
        )

    results: list[dict] = []
    decision_counts = {
        "retrieve": 0,
        "clarify": 0,
        "abstain": 0,
        "fallback": 0,
    }

    for index, question in enumerate(QUESTIONS, start=1):
        try:
            plan = planner.plan(question)
            if plan is None:
                decision_counts["fallback"] += 1
                results.append(
                    {
                        "id": f"Q{index}",
                        "question": question,
                        "planner_status": "fallback",
                        "plan": None,
                        "error": None,
                    }
                )
                continue

            payload = plan.model_dump()
            decision = str(payload.get("decision", ""))
            if decision in decision_counts:
                decision_counts[decision] += 1

            results.append(
                {
                    "id": f"Q{index}",
                    "question": question,
                    "planner_status": "success",
                    "plan": payload,
                    "error": None,
                }
            )
        except Exception as exc:  # Keep the remaining live cases running.
            decision_counts["fallback"] += 1
            results.append(
                {
                    "id": f"Q{index}",
                    "question": question,
                    "planner_status": "error",
                    "plan": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    document = {
        "test_name": "Stage 7.7-A Live Query Planner Test",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "planner_enabled": planner.enabled,
        "planner_model": planner.model,
        "reasoning_effort": planner.reasoning_effort,
        "route_confidence": planner.route_confidence,
        "retrieve_override_confidence": planner.retrieve_override_confidence,
        "summary": {
            "questions_requested": len(QUESTIONS),
            "questions_completed": len(results),
            "successful_plans": sum(
                1 for item in results if item["planner_status"] == "success"
            ),
            "fallback_or_error": sum(
                1 for item in results if item["planner_status"] != "success"
            ),
            "decision_counts": decision_counts,
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
