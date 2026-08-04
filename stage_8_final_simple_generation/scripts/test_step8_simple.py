from __future__ import annotations

import json
from types import SimpleNamespace

from app.grounded_answer_generator import GroundedAnswerGenerator
from app.retrieval_contract import (
    RetrievalDecisionV1,
    RetrievalDiagnosticsV1,
    RetrievalEmbeddingV1,
    RetrievalEvidenceV1,
    RetrievalResultV1,
)


class FakeUsage:
    input_tokens = 100
    output_tokens = 40
    total_tokens = 140


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.output_text = json.dumps(payload, ensure_ascii=False)
        self.usage = FakeUsage()


class FakeResponses:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.payload)


class FakeClient:
    def __init__(self, payload: dict) -> None:
        self.responses = FakeResponses(payload)


def retrieval(behavior: str = "retrieve") -> RetrievalResultV1:
    articles = []
    if behavior == "retrieve":
        articles = [
            RetrievalEvidenceV1(
                uri="http://example.org/article_47",
                local_name="article_47",
                node_kind="Article",
                labels_ar=["المادة 47"],
                labels_en=[],
                article_number=47,
                text="يجوز استرداد السلفة بما لا يزيد على عشرة بالمائة.",
            )
        ]

    return RetrievalResultV1(
        question="كم يقتطع صاحب العمل لاسترداد السلفة؟",
        decision=RetrievalDecisionV1(
            behavior=behavior,
            reason="test reason",
            clarification_question_ar=(
                "ما طبيعة المبلغ؟" if behavior == "clarify" else ""
            ),
        ),
        embedding=RetrievalEmbeddingV1(
            model="test",
            dimensions=0,
            input_tokens=0,
        ),
        articles=articles,
        concepts=[],
        expanded_concepts=[],
        diagnostics=RetrievalDiagnosticsV1(
            article_numbers=[47] if articles else [],
            article_count=len(articles),
        ),
        elapsed_ms=5,
        debug={"previous_step_debug": True},
    )


def main() -> None:
    source_retrieval = retrieval()
    client = FakeClient(
        {
            "answer_ar": "يجوز الاسترداد ضمن الحد القانوني [المادة 47].",
            "cited_article_numbers": [47],
            "limitations": [],
        }
    )
    generator = GroundedAnswerGenerator(
        settings=SimpleNamespace(),
        client=client,
    )
    result = generator.generate(source_retrieval, include_debug=True)

    assert result.status == "generated"
    assert result.grounded is True
    assert len(client.responses.calls) == 1

    call = client.responses.calls[0]
    input_text = call["input"][0]["content"][0]["text"]
    payload = json.loads(input_text)

    assert payload["user_question"] == source_retrieval.question
    assert payload["retrieval_result"] == source_retrieval.model_dump(mode="json")
    assert call["reasoning"] == {"effort": "low"}
    assert call["text"]["verbosity"] == "low"
    assert call["text"]["format"]["strict"] is True
    assert call["max_output_tokens"] == 1800
    assert call["store"] is False
    assert "temperature" not in call
    assert "top_p" not in call
    assert "seed" not in call
    assert "tools" not in call
    assert result.debug["model_calls"] == 1
    assert result.debug["input_included_exact_user_question"] is True
    assert result.debug["input_included_complete_retrieval_result"] is True

    clarify_client = FakeClient({})
    clarify_result = GroundedAnswerGenerator(
        settings=SimpleNamespace(),
        client=clarify_client,
    ).generate(retrieval("clarify"))
    assert clarify_result.status == "clarification_required"
    assert len(clarify_client.responses.calls) == 0

    abstain_client = FakeClient({})
    abstain_result = GroundedAnswerGenerator(
        settings=SimpleNamespace(),
        client=abstain_client,
    ).generate(retrieval("abstain"))
    assert abstain_result.status == "out_of_scope"
    assert len(abstain_client.responses.calls) == 0

    print("Simple Step 8 checks passed.")
    print("Retrieve route uses exactly one model call")
    print("Exact user question is included")
    print("Complete retrieval.v1 output is included")
    print("Strict structured output is enabled")
    print("Temperature, top_p, seed, and tools are not sent")
    print("Clarify and abstain use zero model calls")


if __name__ == "__main__":
    main()
