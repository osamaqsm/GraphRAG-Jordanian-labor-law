from __future__ import annotations

import os
from typing import Iterable

from app.legal_query_planner import (
    AtomicLegalIssue,
    LegalQueryPlan,
    LegalQueryPlanner,
)


class DummySettings:
    openai_api_key = ""
    openai_chat_model = "gpt-5-nano"
    openai_reasoning_effort = "low"
    openai_timeout_seconds = 120
    openai_max_retries = 3


def _retrieve_plan(
    *,
    confidence: float = 0.92,
    issue_count: int = 1,
    maximum_articles: int = 1,
) -> LegalQueryPlan:
    issues = [
        AtomicLegalIssue(
            issue_ar=f"مسألة قانونية {index + 1}",
            retrieval_query_ar=f"استعلام قانوني مركز {index + 1}",
            actors=[],
            conditions=[],
            numbers=[],
            requested_result_ar="تحديد الحكم القانوني",
        )
        for index in range(issue_count)
    ]
    return LegalQueryPlan(
        decision="retrieve",
        confidence=confidence,
        decision_reason="السؤال داخل نطاق قانون العمل ومكتمل الوقائع.",
        normalized_question_ar="سؤال قانوني مكتمل",
        legal_domain="قانون العمل الأردني",
        atomic_issues=issues,
        minimum_articles=1,
        maximum_articles=maximum_articles,
        clarification_question_ar="",
    )


def _clarify_plan(*, confidence: float = 0.90) -> LegalQueryPlan:
    return LegalQueryPlan(
        decision="clarify",
        confidence=confidence,
        decision_reason="نوع القرار الواقعي غير محدد.",
        normalized_question_ar="",
        legal_domain="قانون العمل الأردني",
        atomic_issues=[],
        minimum_articles=0,
        maximum_articles=0,
        clarification_question_ar="ما نوع القرار الذي صدر ضدك؟",
    )


class ScriptedPlanner(LegalQueryPlanner):
    def __init__(
        self,
        scripted: Iterable[LegalQueryPlan],
        *,
        verify_non_answer: bool = True,
    ) -> None:
        os.environ["OPENAI_QUERY_PLANNER_ENABLED"] = "false"
        super().__init__(DummySettings())  # type: ignore[arg-type]
        self.enabled = True
        self.client = object()  # type: ignore[assignment]
        self.verify_non_answer = verify_non_answer
        self._scripted = iter(scripted)
        self.calls: list[tuple[dict, str]] = []

    def _request_plan(
        self,
        *,
        payload: dict,
        instructions: str,
        schema_name: str,
    ) -> LegalQueryPlan:
        self.calls.append((payload, schema_name))
        return next(self._scripted)


def main() -> None:
    prompt = LegalQueryPlanner._instructions()
    verifier_prompt = LegalQueryPlanner._non_answer_verifier_instructions()

    required_main_clauses = (
        "A missing decisive fact must be a concrete real-world fact",
        "It must not be the legal classification",
        "Those are retrieval questions, not missing facts.",
        "Do not create one issue for every grammatical clause",
        "Several requested details or retrieval queries may be governed by one article.",
    )
    for clause in required_main_clauses:
        assert clause in prompt, clause

    required_verifier_clauses = (
        "prevent false non-answers",
        "Default to retrieve unless",
        "A legal conclusion is not a missing fact.",
        "If the proposed non-answer asks the user to provide the legal answer itself",
    )
    for clause in required_verifier_clauses:
        assert clause in verifier_prompt, clause

    # Multiple retrieval queries may legitimately map to one governing article.
    two_queries_one_article = _retrieve_plan(
        issue_count=2,
        maximum_articles=1,
    )
    validated = LegalQueryPlanner._validate_plan(two_queries_one_article)
    assert validated.maximum_articles == 1
    assert len(validated.retrieval_queries) == 2

    # A false clarification is corrected by the second-pass verifier.
    recovered = ScriptedPlanner(
        [_clarify_plan(), _retrieve_plan()]
    )
    recovered_plan = recovered.plan(
        "هل التنازل عن الحق العمالي صحيح أم باطل؟"
    )
    assert recovered_plan is not None
    assert recovered_plan.decision == "retrieve"
    assert len(recovered.calls) == 2
    assert recovered.calls[1][1] == (
        "jordan_labor_non_answer_verification"
    )
    assert "proposed_non_answer_plan" in recovered.calls[1][0]

    # A genuine ambiguity remains a clarification after verification.
    confirmed = ScriptedPlanner(
        [_clarify_plan(), _clarify_plan(confidence=0.93)]
    )
    confirmed_plan = confirmed.plan(
        "النقابة اتخذت قراراً ضدي، هل هو قانوني؟"
    )
    assert confirmed_plan is not None
    assert confirmed_plan.decision == "clarify"
    assert len(confirmed.calls) == 2

    # Verifier can be disabled independently.
    disabled = ScriptedPlanner(
        [_clarify_plan()],
        verify_non_answer=False,
    )
    disabled_plan = disabled.plan(
        "هل التنازل عن الحق العمالي صحيح أم باطل؟"
    )
    assert disabled_plan is not None
    assert disabled_plan.decision == "clarify"
    assert len(disabled.calls) == 1

    print("Stage 7.7-C offline checks passed.")
    print("False non-answer verification: enabled by default")
    print("False clarification recovery: clarify -> retrieve")
    print("Genuine ambiguity preservation: clarify -> clarify")
    print("Multiple issue queries may map to one article")


if __name__ == "__main__":
    main()
