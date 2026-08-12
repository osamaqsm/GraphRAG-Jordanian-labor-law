from __future__ import annotations

from app.config import Settings
from app.legal_query_planner import LegalQueryPlanner


QUESTIONS = [
    "ما مدة الإجازة السنوية للعامل؟",
    "إذا انتهى عقد العامل، ما حقوقه المتعلقة بشهادة الخدمة وإعادة الوثائق؟",
    "ما قيمة مخالفة تجاوز الإشارة الحمراء وكم نقطة مرورية تترتب عليها؟",
]


def main() -> None:
    settings = Settings()
    planner = LegalQueryPlanner(settings)
    print("INITIAL:", planner.diagnostic_state())

    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\n[{index}/3] {question}")
        plan = planner.plan(question)
        if plan is None:
            raise RuntimeError(
                "Planner returned None. In strict evaluation this should never "
                "be accepted as a valid model-comparison run."
            )
        print(plan.model_dump(mode="json"))
        print("DIAGNOSTIC:", planner.diagnostic_state())


if __name__ == "__main__":
    main()