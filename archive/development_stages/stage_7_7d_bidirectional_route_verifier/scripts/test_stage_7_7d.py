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
        decision_reason="نوع المخالفة الواقعي غير محدد.",
        normalized_question_ar="",
        legal_domain="قانون العمل الأردني",
        atomic_issues=[],
        minimum_articles=0,
        maximum_articles=0,
        clarification_question_ar="ما نوع المخالفة التي حدثت؟",
    )


def _abstain_plan(*, confidence: float = 0.90) -> LegalQueryPlan:
    return LegalQueryPlan(
        decision="abstain",
        confidence=confidence,
        decision_reason="الموضوع خارج نطاق قانون العمل.",
        normalized_question_ar="",
        legal_domain="نطاق آخر",
        atomic_issues=[],
        minimum_articles=0,
        maximum_articles=0,
        clarification_question_ar="",
    )


class ScriptedPlanner(LegalQueryPlanner):
    def __init__(
        self,
        scripted: Iterable[LegalQueryPlan],
        *,
        verify_non_answer: bool = True,
        verify_low_confidence_retrieve: bool = True,
        retrieve_verification_below: float = 0.90,
    ) -> None:
        os.environ["OPENAI_QUERY_PLANNER_ENABLED"] = "false"
        super().__init__(DummySettings())  # type: ignore[arg-type]
        self.enabled = True
        self.client = object()  # type: ignore[assignment]
        self.verify_non_answer = verify_non_answer
        self.verify_low_confidence_retrieve = (
            verify_low_confidence_retrieve
        )
        self.retrieve_verification_below = retrieve_verification_below
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
    verifier_prompt = LegalQueryPlanner._route_verifier_instructions()

    required_main_clauses = (
        "Do not abstain merely because a labor-law provision overlaps",
        "worker-created inventions or intellectual property",
        "The user does not need to name the job first.",
        "requires clarification because the type of safety violation",
    )
    for clause in required_main_clauses:
        assert clause in prompt, clause

    required_verifier_clauses = (
        "second-pass route verifier",
        "worker-created inventions or intellectual property",
        "without the violation type -> clarify",
        "who determines jobs requiring medical fitness",
    )
    for clause in required_verifier_clauses:
        assert clause in verifier_prompt, clause

    # A false scope abstention is recovered by route verification.
    recovered_scope = ScriptedPlanner(
        [_abstain_plan(confidence=0.88), _retrieve_plan(confidence=0.92)]
    )
    recovered_scope_plan = recovered_scope.plan(
        "ابتكر عامل منتجاً مستخدماً أدوات المنشأة؛ لمن تكون الملكية؟"
    )
    assert recovered_scope_plan is not None
    assert recovered_scope_plan.decision == "retrieve"
    assert len(recovered_scope.calls) == 2
    assert recovered_scope.calls[1][1] == (
        "jordan_labor_route_verification"
    )
    assert "proposed_plan" in recovered_scope.calls[1][0]

    # A low-confidence false retrieve is checked and corrected to clarify.
    recovered_safety = ScriptedPlanner(
        [_retrieve_plan(confidence=0.82), _clarify_plan(confidence=0.91)]
    )
    recovered_safety_plan = recovered_safety.plan(
        "في مخالفة سلامة، مين المسؤول وشو العقوبة؟"
    )
    assert recovered_safety_plan is not None
    assert recovered_safety_plan.decision == "clarify"
    assert len(recovered_safety.calls) == 2
    assert recovered_safety.calls[1][0]["verification_reason"] == (
        "low_confidence_retrieve"
    )

    # A valid low-confidence retrieve is confirmed by the verifier.
    confirmed_retrieve = ScriptedPlanner(
        [_retrieve_plan(confidence=0.85), _retrieve_plan(confidence=0.91)]
    )
    confirmed_retrieve_plan = confirmed_retrieve.plan(
        "من يحدد الأعمال التي تتطلب فحص لياقة طبية؟"
    )
    assert confirmed_retrieve_plan is not None
    assert confirmed_retrieve_plan.decision == "retrieve"
    assert len(confirmed_retrieve.calls) == 2

    # A high-confidence retrieve continues with one call.
    direct_retrieve = ScriptedPlanner([_retrieve_plan(confidence=0.92)])
    direct_retrieve_plan = direct_retrieve.plan(
        "كم يقتطع صاحب العمل من الأجر لاسترداد سلفة؟"
    )
    assert direct_retrieve_plan is not None
    assert direct_retrieve_plan.decision == "retrieve"
    assert len(direct_retrieve.calls) == 1

    # Low-confidence retrieve verification can be disabled independently.
    disabled = ScriptedPlanner(
        [_retrieve_plan(confidence=0.82)],
        verify_low_confidence_retrieve=False,
    )
    disabled_plan = disabled.plan("سؤال استرجاع منخفض الثقة")
    assert disabled_plan is not None
    assert disabled_plan.decision == "retrieve"
    assert len(disabled.calls) == 1

    print("Stage 7.7-D offline checks passed.")
    print("Overlapping labor-law scope preserved")
    print("False abstention recovery: abstain -> retrieve")
    print("Low-confidence retrieve verification: retrieve -> clarify")
    print("High-confidence retrieve fast path: one planner call")


if __name__ == "__main__":
    main()
