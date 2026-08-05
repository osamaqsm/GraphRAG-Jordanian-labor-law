from __future__ import annotations

import json

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


def main() -> None:
    planner = LegalQueryPlanner(get_settings())

    print("planner_enabled=", planner.enabled)
    print("planner_model=", planner.model)

    if not planner.enabled:
        raise SystemExit(
            "Enable OPENAI_QUERY_PLANNER_ENABLED=true before this live test."
        )

    for index, question in enumerate(QUESTIONS, start=1):
        plan = planner.plan(question)
        print("=" * 80)
        print(f"Q{index}: {question}")
        if plan is None:
            print("planner_result=None (deterministic fallback would be used)")
            continue
        print(
            json.dumps(
                plan.model_dump(),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
