from __future__ import annotations

import os

from app.legal_query_planner import (
    AtomicLegalIssue,
    LegalQueryPlan,
    LegalQueryPlanner,
)
from app.legal_question_analysis import analyze_legal_question


class DummySettings:
    openai_api_key = ""
    openai_chat_model = "gpt-5-nano"
    openai_reasoning_effort = "low"
    openai_timeout_seconds = 120
    openai_max_retries = 3


def _planner() -> LegalQueryPlanner:
    os.environ["OPENAI_QUERY_PLANNER_ENABLED"] = "false"
    return LegalQueryPlanner(DummySettings())  # type: ignore[arg-type]


def _retrieve_plan(*, confidence: float = 0.96) -> LegalQueryPlan:
    return LegalQueryPlan(
        decision="retrieve",
        confidence=confidence,
        decision_reason="The question is specific and within labor law.",
        normalized_question_ar=(
            "ما شروط عقد التدريب ومتى يجوز إنهاؤه لأسباب صحية"
        ),
        legal_domain="vocational_training",
        atomic_issues=[
            AtomicLegalIssue(
                issue_ar="شروط عقد التدريب",
                retrieval_query_ar=(
                    "مدة عقد التدريب ومراحله وأجر المتدرب"
                ),
                actors=["المتدرب", "صاحب العمل"],
                conditions=[],
                numbers=[],
                requested_result_ar="تحديد عناصر العقد",
            ),
            AtomicLegalIssue(
                issue_ar="إنهاء عقد التدريب",
                retrieval_query_ar=(
                    "إنهاء عقد التدريب عند تهديد صحة المتدرب"
                ),
                actors=["المتدرب", "صاحب العمل"],
                conditions=["تهديد الصحة"],
                numbers=[],
                requested_result_ar="تحديد حالات الإنهاء",
            ),
        ],
        minimum_articles=2,
        maximum_articles=2,
        clarification_question_ar="",
    )


def main() -> None:
    planner = _planner()
    assert planner.enabled is False
    assert planner.plan("ما حقوق العامل؟") is None

    base = analyze_legal_question(
        "ما الذي يجب تحديده في عقد التدريب ومتى يجوز إنهاؤه؟"
    )
    merged = planner.merge_with_analysis(
        analysis=base,
        plan=_retrieve_plan(),
    )
    assert merged.behavior == "retrieve"
    assert merged.planner_used is True
    assert merged.max_final_articles == 2
    assert len(merged.planner_queries) == 2
    assert "مدة عقد التدريب" in merged.bm25_query
    assert "Structured pre-retrieval issue plan" in merged.reranker_question

    consumer_base = analyze_legal_question(
        "اشتريت هاتفاً وظهر فيه عيب، هل أستطيع استبداله؟"
    )
    abstain_plan = LegalQueryPlan(
        decision="abstain",
        confidence=0.97,
        decision_reason="Consumer law is outside the represented KG.",
        normalized_question_ar="",
        legal_domain="consumer_law",
        atomic_issues=[],
        minimum_articles=0,
        maximum_articles=0,
        clarification_question_ar="",
    )
    consumer_merged = planner.merge_with_analysis(
        analysis=consumer_base,
        plan=abstain_plan,
    )
    assert consumer_merged.behavior == "abstain"
    assert consumer_merged.max_final_articles == 0

    vague_base = analyze_legal_question(
        "النقابة اتخذت قراراً ضدي، هل قرارها قانوني؟"
    )
    clarify_plan = LegalQueryPlan(
        decision="clarify",
        confidence=0.95,
        decision_reason="The type of union decision is missing.",
        normalized_question_ar="",
        legal_domain="trade_unions",
        atomic_issues=[],
        minimum_articles=0,
        maximum_articles=0,
        clarification_question_ar=(
            "ما نوع القرار الذي اتخذته النقابة وما صفتك؟"
        ),
    )
    vague_merged = planner.merge_with_analysis(
        analysis=vague_base,
        plan=clarify_plan,
    )
    assert vague_merged.behavior == "clarify"
    assert vague_merged.clarification_question

    low_confidence = planner.merge_with_analysis(
        analysis=base,
        plan=_retrieve_plan(confidence=0.20),
    )
    assert low_confidence == base

    deterministic_abstain = analyze_legal_question(
        "وصلتني مخالفة سير بسبب السرعة، كيف أعترض؟"
    )
    assert deterministic_abstain.behavior == "abstain"
    protected = planner.merge_with_analysis(
        analysis=deterministic_abstain,
        plan=_retrieve_plan(),
    )
    assert protected.behavior == "abstain"

    print("Stage 7.7-A optional query-planner checks passed.")
    print("Planner disabled fallback: deterministic path preserved")
    print("Planner retrieve issues:", list(merged.planner_queries))
    print("Planner clarify route:", vague_merged.clarification_question)
    print("Planner abstain route: consumer-law question blocked")


if __name__ == "__main__":
    main()
